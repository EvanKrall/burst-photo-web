// Thin WebGPU helpers: device init, buffer management, uniform packing, dispatch.

export async function initGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not available in this browser. Try Chrome/Edge 113+, Safari 26, or Firefox (with WebGPU enabled).');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No suitable GPU adapter found.');
  // The frequency-domain merge works on float planes ~8× the image size (the FT plane of
  // a 24 MP burst is ~200 MB), beyond the 128 MiB default binding limit — request the
  // adapter's maximum buffer limits when they are larger.
  const requiredLimits = {};
  for (const key of ['maxStorageBufferBindingSize', 'maxBufferSize']) {
    const max = adapter.limits[key];
    if (max > 128 * 1024 * 1024) requiredLimits[key] = max;
  }
  const device = await adapter.requestDevice({ requiredLimits });
  return { adapter, device };
}

export class GPUContext {
  constructor(device) {
    this.device = device;
    this.module = null;
    this.pipelines = new Map(); // entryPoint -> GPUComputePipeline
  }

  setShaderModule(wgslSource) {
    this.module = this.device.createShaderModule({ code: wgslSource });
    // Surface compilation info early (async compilation warnings/errors).
    if (this.module.getCompilationInfo) {
      this.module.getCompilationInfo().then((info) => {
        for (const msg of info.messages) {
          if (msg.type === 'error') console.error(`WGSL error @${msg.lineNum}:${msg.linePos}: ${msg.message}`);
          else console.warn(`WGSL ${msg.type} @${msg.lineNum}:${msg.linePos}: ${msg.message}`);
        }
      });
    }
  }

  pipeline(entryPoint) {
    if (!this.pipelines.has(entryPoint)) {
      const p = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: this.module, entryPoint },
      });
      this.pipelines.set(entryPoint, p);
    }
    return this.pipelines.get(entryPoint);
  }

  makeBuffer(label, byteSize, usage) {
    return this.device.createBuffer({ label, size: byteSize, usage });
  }

  // Create a storage buffer initialized from a typed array (aligned copy).
  storageFromData(label, typedArray) {
    const buf = this.makeBuffer(
      label,
      Math.max(16, aligned4(typedArray.byteLength)),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.device.queue.writeBuffer(buf, 0, typedArray.buffer, typedArray.byteOffset, aligned4(typedArray.byteLength));
    return buf;
  }

  zeros(label, byteSize) {
    const buf = this.makeBuffer(
      label,
      byteSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.device.queue.writeBuffer(buf, 0, new Uint8Array(byteSize));
    return buf;
  }

  // GPU-side buffer copy (byteSize must be a multiple of 4) — e.g. seeding the merged
  // frequency-domain accumulator with the reference spectrum.
  copyBuffer(label, src, byteSize) {
    const dst = this.makeBuffer(
      label,
      byteSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    const encoder = this.device.createCommandEncoder({ label: `copy:${label}` });
    encoder.copyBufferToBuffer(src, 0, dst, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);
    return dst;
  }

  /**
   * Dispatch a compute kernel.
   * @param {string} entryPoint kernel name
   * @param {Array<{buffer: GPUBuffer|ArrayBuffer}>} resources in shader binding order;
   *        plain ArrayBuffers are uploaded to transient uniform buffers automatically
   * @param {[number, number, number]} groups workgroup count
   */
  dispatch(entryPoint, resources, groups) {
    const tempUniforms = [];
    const entries = resources.map((r, i) => {
      let buf = r.buffer;
      if (buf instanceof ArrayBuffer) {
        const ub = this.device.createBuffer({
          label: `uniform-${entryPoint}`, size: Math.max(16, buf.byteLength),
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(ub, 0, buf);
        tempUniforms.push(ub);
        buf = ub;
      }
      return { binding: i, resource: { buffer: buf } };
    });
    const bindGroup = this.device.createBindGroup({ layout: this.pipeline(entryPoint).getBindGroupLayout(0), entries });
    const encoder = this.device.createCommandEncoder({ label: `dispatch:${entryPoint}` });
    const pass = encoder.beginComputePass({ label: entryPoint });
    pass.setPipeline(this.pipeline(entryPoint));
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groups[0], groups[1] || 1, groups[2] || 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    // Destroying after submit is legal: enqueued work keeps the memory alive.
    for (const ub of tempUniforms) ub.destroy();
  }

  // Copy a storage buffer's contents to a freshly mapped staging buffer.
  async readBack(buffer, byteSize) {
    const staging = this.device.createBuffer({
      label: 'staging-readback',
      size: Math.max(16, aligned4(byteSize)),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, aligned4(byteSize));
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return copy;
  }
}

export function aligned4(n) {
  return Math.ceil(n / 4) * 4;
}

// Pack scalars into an ArrayBuffer matching a WGSL uniform struct of scalars only.
// Order of entries must match the WGSL struct field order.
export function packUniforms(types, values) {
  let size = 0;
  const offsets = [];
  for (const t of types) {
    offsets.push(size);
    size += 4;
  }
  const buf = new ArrayBuffer(Math.max(16, size));
  const dv = new DataView(buf);
  for (let i = 0; i < values.length; i++) {
    const t = types[i];
    if (t === 'f32') dv.setFloat32(offsets[i], values[i], true);
    else if (t === 'i32') dv.setInt32(offsets[i], values[i], true);
    else if (t === 'u32') dv.setUint32(offsets[i], values[i], true);
    else throw new Error(`unknown uniform type ${t}`);
  }
  return buf;
}
