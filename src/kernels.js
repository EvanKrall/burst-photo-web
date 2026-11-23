// WGSL compute kernels — hand-ported from Burst Photo's Metal sources
// (burstphoto/{align/align.metal, merge/spatial.metal, texture/texture.metal,
// merge/frequency.metal}).
//
// Design notes:
//  - Image planes are storage buffers of array<f32> (row-major, index = y*width + x)
//    instead of textures. The Metal code accesses textures exclusively via explicit
//    .read()/.write() (no sampling hardware), so buffers map 1:1 and avoid WebGPU
//    storage-texture format restrictions and maxTextureDimension limits.
//  - Raw sensor data arrives as a packed array<u32> holding two 16-bit pixels per u32.
//  - Alignment grids are array<vec2<i32>> (Metal used rg16sint textures).
//  - Uniform params are scalar-only structs, packed 1:1 by the JS side.
//  - Where the Metal code relies on out-of-bounds texture reads returning 0, we clamp
//    coordinates instead (only affects border pixels of the warp/upsample stages).
//
// Each kernel below cites the Metal kernel it was ported from.

export const WGSL = /* wgsl */ `

// ---------------------------------------------------------------- constants
const FLOAT16_MIN_VAL: f32 = -65504.0;   // misc/constants.h
const UINT16_MAX_VAL: f32 = 65535.0;

// ---------------------------------------------------------------- utilities
fn clampi(v: i32, lo: i32, hi: i32) -> i32 { return max(lo, min(hi, v)); }
fn clampu(v: i32, n: u32) -> u32 { return u32(clampi(v, 0, i32(n) - 1)); }

// Load a 16-bit sample from the packed raw buffer (2 pixels per u32).
fn load_raw(raw: ptr<storage, array<u32>, read>, idx: u32) -> u32 {
    return ((*raw)[idx >> 1u] >> ((idx & 1u) * 16u)) & 0xffffu;
}

// ============================================================ prepare_bayer
// Port of texture.metal:prepare_texture_bayer for uniform exposure (exposure_diff = 0):
// the Metal expression (v - black)*2^(exposure_diff/100) + black simplifies to v, so
// preparation is black-level-neutral (black is subtracted once at convert_to_u16).
// Hot-pixel correction and exposure bracketing are omitted in the PoC.
// Params: in_w, in_h, pad_left, pad_top
struct PrepareParams { in_w: i32, in_h: i32, pad_left: i32, pad_top: i32, };
@group(0) @binding(0) var<storage, read>       raw_in:    array<u32>;
@group(0) @binding(1) var<storage, read_write> plane_out: array<f32>;
@group(0) @binding(2) var<uniform>             prepare_p: PrepareParams;

@compute @workgroup_size(8, 8)
fn prepare_bayer(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let x = i32(gid3.x);
    let y = i32(gid3.y);
    if (x >= prepare_p.in_w || y >= prepare_p.in_h) { return; }  // exact-dispatch guard
    let idx = y * prepare_p.in_w + x;
    let pixel_value = f32(load_raw(&raw_in, u32(idx)));

    let w_out = prepare_p.in_w + 2 * prepare_p.pad_left;   // padded width
    let o = (y + prepare_p.pad_top) * w_out + (x + prepare_p.pad_left);
    plane_out[o] = max(pixel_value, 0.0);
}

// ==================================================================== blur
// Port of texture.metal:blur_mosaic_texture (separable binomial filter, verbatim).
// Params: w (plane width), h (plane height), size (dim along blur direction), kernel_size, mosaic, direction
struct BlurParams { w: i32, h: i32, size: i32, kernel_size: i32, mosaic: i32, direction: i32, };
@group(0) @binding(0) var<storage, read>       blur_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> blur_out: array<f32>;
@group(0) @binding(2) var<uniform>             blur_p:   BlurParams;

@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= blur_p.w || i32(gid3.y) >= blur_p.h) { return; }  // exact-dispatch guard
    // binomial kernel weights (truncated so removed weights contribute < 0.25%)
    var bw = array<f32, 9>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var kernel_size_trunc = blur_p.kernel_size;
    let ks = blur_p.kernel_size;
    if      (ks == 1)  { bw = array<f32, 9>(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); }
    else if (ks == 2)  { bw = array<f32, 9>(6.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); }
    else if (ks == 3)  { bw = array<f32, 9>(20.0, 15.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0); }
    else if (ks == 4)  { bw = array<f32, 9>(70.0, 56.0, 28.0, 8.0, 1.0, 0.0, 0.0, 0.0, 0.0); }
    else if (ks == 5)  { bw = array<f32, 9>(252.0, 210.0, 120.0, 45.0, 10.0, 0.0, 0.0, 0.0, 0.0); kernel_size_trunc = 4; }
    else if (ks == 6)  { bw = array<f32, 9>(924.0, 792.0, 495.0, 220.0, 66.0, 12.0, 0.0, 0.0, 0.0); kernel_size_trunc = 5; }
    else if (ks == 7)  { bw = array<f32, 9>(3432.0, 3003.0, 2002.0, 1001.0, 364.0, 91.0, 0.0, 0.0, 0.0); kernel_size_trunc = 5; }
    else if (ks == 8)  { bw = array<f32, 9>(12870.0, 11440.0, 8008.0, 4368.0, 1820.0, 560.0, 120.0, 0.0, 0.0); kernel_size_trunc = 6; }
    else if (ks == 16) { bw = array<f32, 9>(601080390.0, 565722720.0, 471435600.0, 347373600.0, 225792840.0, 129024480.0, 64512240.0, 28048800.0, 10518300.0); kernel_size_trunc = 8; }

    var total_intensity = 0.0;
    var total_weight = 0.0;

    let gx = i32(gid3.x);
    let gy = i32(gid3.y);
    // direction == 0: blur along x, direction == 1: blur along y
    let i0 = select(gx, gy, blur_p.direction == 1);
    let step = blur_p.mosaic;   // taps are spaced one mosaic cell apart

    for (var di = -kernel_size_trunc; di <= kernel_size_trunc; di++) {
        let i = i0 + step * di;
        if (i >= 0 && i < blur_p.size) {
            var weight = bw[abs(di)];
            var sx = gx;
            var sy = gy;
            if (blur_p.direction == 0) { sx = i; } else { sy = i; }
            total_intensity += weight * blur_in[sy * blur_p.w + sx];
            total_weight += weight;
        }
    }
    blur_out[gy * blur_p.w + gx] = total_intensity / total_weight;
}

// ================================================================ avg_pool
// Port of align.metal:avg_pool (verbatim; normalization variant not needed for
// the spatial pipeline with uniform exposure).
// Params: in_w, out_w, out_h, scale, black_level
struct PoolParams { in_w: i32, out_w: i32, out_h: i32, scale: i32, black_level: f32, };
@group(0) @binding(0) var<storage, read>       pool_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> pool_out: array<f32>;
@group(0) @binding(2) var<uniform>             pool_p:   PoolParams;

@compute @workgroup_size(8, 8)
fn avg_pool(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= pool_p.out_w || i32(gid3.y) >= pool_p.out_h) { return; }  // exact-dispatch guard
    var out_pixel = 0.0;
    let x0 = i32(gid3.x) * pool_p.scale;
    let y0 = i32(gid3.y) * pool_p.scale;
    for (var dx = 0; dx < pool_p.scale; dx++) {
        for (var dy = 0; dy < pool_p.scale; dy++) {
            out_pixel += pool_in[(y0 + dy) * pool_p.in_w + (x0 + dx)] - pool_p.black_level;
        }
    }
    pool_out[i32(gid3.y) * pool_p.out_w + i32(gid3.x)] = out_pixel / f32(pool_p.scale * pool_p.scale);
}

// ======================================================= upsample_nearest_i32
// Port of texture.metal:upsample_nearest_int for the alignment grid (i32 vec2).
// Params: in_w, in_h, out_w, out_h, scale_x, scale_y
struct UpsNNParams { in_w: i32, in_h: i32, out_w: i32, out_h: i32, scale_x: f32, scale_y: f32, };
@group(0) @binding(0) var<storage, read>       upsnn_in:  array<vec2<i32>>;
@group(0) @binding(1) var<storage, read_write> upsnn_out: array<vec2<i32>>;
@group(0) @binding(2) var<uniform>             upsnn_p:   UpsNNParams;

@compute @workgroup_size(8, 8)
fn upsample_nearest_i32(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let x = i32(gid3.x);
    let y = i32(gid3.y);
    if (x >= upsnn_p.out_w || y >= upsnn_p.out_h) { return; }  // exact-dispatch guard
    let sx = clampi(i32(round(f32(x) / upsnn_p.scale_x)), 0, upsnn_p.in_w - 1);
    let sy = clampi(i32(round(f32(y) / upsnn_p.scale_y)), 0, upsnn_p.in_h - 1);
    upsnn_out[y * upsnn_p.out_w + x] = upsnn_in[sy * upsnn_p.in_w + sx];
}

// ================================================ compute_tile_differences
// Port of align.metal:compute_tile_differences (generic search-distance
// variant; the N=25 optimized kernel is an optimization we skip in the PoC).
// tile_diff is a flat array: index = (tile_y * n_tiles_x + tile_x) * n_pos_2d + pos
// Params: level_w, level_h, n_tiles_x, tile_size, search_dist, downscale, weight_ssd
struct CtdParams {
    level_w: i32, level_h: i32, n_tiles_x: i32, n_tiles_y: i32,
    tile_size: i32, search_dist: i32, downscale: i32, weight_ssd: i32,
};
@group(0) @binding(0) var<storage, read>       ctd_ref:  array<f32>;
@group(0) @binding(1) var<storage, read>       ctd_comp: array<f32>;
@group(0) @binding(2) var<storage, read>       ctd_prev: array<vec2<i32>>;
@group(0) @binding(3) var<storage, read_write> ctd_diff: array<f32>;
@group(0) @binding(4) var<uniform>             ctd_p:    CtdParams;

@compute @workgroup_size(8, 8, 1)
fn compute_tile_differences(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= ctd_p.n_tiles_x || i32(gid3.y) >= ctd_p.n_tiles_y || i32(gid3.z) >= (2 * ctd_p.search_dist + 1) * (2 * ctd_p.search_dist + 1)) { return; }  // exact-dispatch guard
    let n_pos_1d = 2 * ctd_p.search_dist + 1;

    // tile position if previous alignment were 0
    let x0 = i32(gid3.x) * ctd_p.tile_size / 2;
    let y0 = i32(gid3.y) * ctd_p.tile_size / 2;

    // candidate displacement for this invocation
    var dx0 = i32(gid3.z) % n_pos_1d - ctd_p.search_dist;
    var dy0 = i32(gid3.z) / n_pos_1d - ctd_p.search_dist;

    // factor in previous (upsampled) alignment
    let prev = ctd_prev[i32(gid3.y) * ctd_p.n_tiles_x + i32(gid3.x)];
    dx0 += ctd_p.downscale * prev.x;
    dy0 += ctd_p.downscale * prev.y;

    var diff = 0.0;
    for (var dx1 = 0; dx1 < ctd_p.tile_size; dx1++) {
        for (var dy1 = 0; dy1 < ctd_p.tile_size; dy1++) {
            let ref_tile_x = x0 + dx1;
            let ref_tile_y = y0 + dy1;
            let comp_tile_x = ref_tile_x + dx0;
            let comp_tile_y = ref_tile_y + dy0;

            var diff_abs: f32;
            if (comp_tile_x < 0 || comp_tile_y < 0 ||
                comp_tile_x >= ctd_p.level_w || comp_tile_y >= ctd_p.level_h) {
                // out of frame: compare against a large negative value (Metal: 2*FLOAT16_MIN_VAL)
                diff_abs = abs(ctd_ref[ref_tile_y * ctd_p.level_w + ref_tile_x] - 2.0 * FLOAT16_MIN_VAL);
            } else {
                diff_abs = abs(ctd_ref[ref_tile_y * ctd_p.level_w + ref_tile_x]
                             - ctd_comp[comp_tile_y * ctd_p.level_w + comp_tile_x]);
            }
            diff += (1.0 - f32(ctd_p.weight_ssd)) * diff_abs
                  + f32(ctd_p.weight_ssd) * diff_abs * diff_abs;
        }
    }
    let n_pos_2d = n_pos_1d * n_pos_1d;
    ctd_diff[(i32(gid3.y) * ctd_p.n_tiles_x + i32(gid3.x)) * n_pos_2d + i32(gid3.z)] = diff;
}

// ================================================== find_best_tile_alignment
// Port of align.metal:find_best_tile_alignment (verbatim).
// Params: n_tiles_x, n_pos_1d, search_dist, downscale
struct FbtaParams { n_tiles_x: i32, n_tiles_y: i32, n_pos_1d: i32, search_dist: i32, downscale: i32, };
@group(0) @binding(0) var<storage, read>       fbta_diff: array<f32>;
@group(0) @binding(1) var<storage, read>       fbta_prev: array<vec2<i32>>;
@group(0) @binding(2) var<storage, read_write> fbta_out:   array<vec2<i32>>;
@group(0) @binding(3) var<uniform>             fbta_p:    FbtaParams;

@compute @workgroup_size(8, 8)
fn find_best_tile_alignment(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= fbta_p.n_tiles_x || i32(gid3.y) >= fbta_p.n_tiles_y) { return; }  // exact-dispatch guard
    let n_pos_2d = fbta_p.n_pos_1d * fbta_p.n_pos_1d;
    let tile = i32(gid3.y) * fbta_p.n_tiles_x + i32(gid3.x);

    var min_diff_val = 1e20;
    var min_diff_idx = 0;
    for (var i = 0; i < n_pos_2d; i++) {
        let d = fbta_diff[tile * n_pos_2d + i];
        if (d < min_diff_val) { min_diff_val = d; min_diff_idx = i; }
    }

    let prev = fbta_prev[tile] * vec2<i32>(fbta_p.downscale, fbta_p.downscale);
    fbta_out[tile] = vec2<i32>(
        prev.x + (min_diff_idx % fbta_p.n_pos_1d) - fbta_p.search_dist,
        prev.y + (min_diff_idx / fbta_p.n_pos_1d) - fbta_p.search_dist);
}

// ============================================================== warp_bayer
// Port of align.metal:warp_texture_bayer (verbatim; reads clamped to bounds).
// Params: in_w, in_h, half_tile, n_tiles_x, n_tiles_y, downscale
struct WarpParams {
    in_w: i32, in_h: i32, half_tile: i32,
    n_tiles_x: i32, n_tiles_y: i32, downscale: i32,
};
@group(0) @binding(0) var<storage, read>       warp_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> warp_out: array<f32>;
@group(0) @binding(2) var<storage, read>       warp_align: array<vec2<i32>>;
@group(0) @binding(3) var<uniform>             warp_p:   WarpParams;

@compute @workgroup_size(8, 8)
fn warp_bayer(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let x = i32(gid3.x);
    let y = i32(gid3.y);
    if (x >= warp_p.in_w || y >= warp_p.in_h) { return; }  // exact-dispatch guard
    let hts = f32(warp_p.half_tile);

    let x_grid = (f32(x) + 0.5) / hts - 1.0;
    let y_grid = (f32(y) + 0.5) / hts - 1.0;

    let x_grid_floor = i32(max(0.0, floor(x_grid)) + 0.1);
    let y_grid_floor = i32(max(0.0, floor(y_grid)) + 0.1);
    let x_grid_ceil  = i32(min(ceil(x_grid), f32(warp_p.n_tiles_x - 1)) + 0.1);
    let y_grid_ceil  = i32(min(ceil(y_grid), f32(warp_p.n_tiles_y - 1)) + 0.1);

    let weight_x = (f32(x % warp_p.half_tile) + 0.5) / (2.0 * hts);
    let weight_y = (f32(y % warp_p.half_tile) + 0.5) / (2.0 * hts);

    let a0 = warp_align[y_grid_floor * warp_p.n_tiles_x + x_grid_floor] * vec2<i32>(warp_p.downscale, warp_p.downscale);
    let a1 = warp_align[y_grid_floor * warp_p.n_tiles_x + x_grid_ceil ] * vec2<i32>(warp_p.downscale, warp_p.downscale);
    let a2 = warp_align[y_grid_ceil  * warp_p.n_tiles_x + x_grid_floor] * vec2<i32>(warp_p.downscale, warp_p.downscale);
    let a3 = warp_align[y_grid_ceil  * warp_p.n_tiles_x + x_grid_ceil ] * vec2<i32>(warp_p.downscale, warp_p.downscale);

    var pixel_value  = (1.0 - weight_x) * (1.0 - weight_y) * warp_in[clampu(y + a0.y, u32(warp_p.in_h)) * u32(warp_p.in_w) + clampu(x + a0.x, u32(warp_p.in_w))];
    var total_weight = (1.0 - weight_x) * (1.0 - weight_y);
    pixel_value  += weight_x * (1.0 - weight_y) * warp_in[clampu(y + a1.y, u32(warp_p.in_h)) * u32(warp_p.in_w) + clampu(x + a1.x, u32(warp_p.in_w))];
    total_weight += weight_x * (1.0 - weight_y);
    pixel_value  += (1.0 - weight_x) * weight_y * warp_in[clampu(y + a2.y, u32(warp_p.in_h)) * u32(warp_p.in_w) + clampu(x + a2.x, u32(warp_p.in_w))];
    total_weight += (1.0 - weight_x) * weight_y;
    pixel_value  += weight_x * weight_y * warp_in[clampu(y + a3.y, u32(warp_p.in_h)) * u32(warp_p.in_w) + clampu(x + a3.x, u32(warp_p.in_w))];
    total_weight += weight_x * weight_y;

    warp_out[y * warp_p.in_w + x] = pixel_value / total_weight;
}

// ========================================================= color_difference
// Port of merge/spatial.metal:color_difference (verbatim).
// Params: w (plane width), out_h, mosaic
struct CdParams { w: i32, out_h: i32, mosaic: i32, };
@group(0) @binding(0) var<storage, read>       cd_t1:  array<f32>;
@group(0) @binding(1) var<storage, read>       cd_t2:  array<f32>;
@group(0) @binding(2) var<storage, read_write> cd_out: array<f32>;
@group(0) @binding(3) var<uniform>             cd_p:   CdParams;

@compute @workgroup_size(8, 8)
fn color_difference(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.y) >= cd_p.out_h) { return; }  // exact-dispatch guard
    var total_diff = 0.0;
    let x0 = i32(gid3.x) * cd_p.mosaic;
    let y0 = i32(gid3.y) * cd_p.mosaic;
    for (var dx = 0; dx < cd_p.mosaic; dx++) {
        for (var dy = 0; dy < cd_p.mosaic; dy++) {
            let x = x0 + dx;
            let y = y0 + dy;
            total_diff += abs(cd_t1[y * cd_p.w + x] - cd_t2[y * cd_p.w + x]);
        }
    }
    cd_out[i32(gid3.y) * (cd_p.w / cd_p.mosaic) + i32(gid3.x)] = total_diff;
}

// ===================================================== compute_merge_weight
// Port of merge/spatial.metal:compute_merge_weight (verbatim; noise_sd passed
// as a uniform scalar instead of a buffer).
// Params: w (diff-plane width), h, noise_sd, robustness
struct MwParams { w: i32, h: i32, noise_sd: f32, robustness: f32, };
@group(0) @binding(0) var<storage, read>       mw_diff:   array<f32>;
@group(0) @binding(1) var<storage, read_write> mw_weight: array<f32>;
@group(0) @binding(2) var<uniform>             mw_p:      MwParams;

@compute @workgroup_size(8, 8)
fn compute_merge_weight(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= mw_p.w || i32(gid3.y) >= mw_p.h) { return; }  // exact-dispatch guard
    let i = i32(gid3.y) * mw_p.w + i32(gid3.x);
    let diff = mw_diff[i];
    var weight: f32;
    if (mw_p.robustness == 0.0) {
        weight = 1.0;
    } else {
        let max_diff = mw_p.noise_sd / mw_p.robustness;
        weight = clamp(1.0 - diff / max_diff, 0.0, 1.0);
    }
    mw_weight[i] = weight;
}

// ========================================================= upsample_bilinear
// Port of texture.metal:upsample_bilinear_float (verbatim; reads clamped).
// Params: in_w, in_h, out_w, out_h, scale_x, scale_y
struct UpsBLParams { in_w: i32, in_h: i32, out_w: i32, out_h: i32, scale_x: f32, scale_y: f32, };
@group(0) @binding(0) var<storage, read>       upsbl_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> upsbl_out: array<f32>;
@group(0) @binding(2) var<uniform>             upsbl_p:   UpsBLParams;

@compute @workgroup_size(8, 8)
fn upsample_bilinear(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= upsbl_p.out_w || i32(gid3.y) >= upsbl_p.out_h) { return; }  // exact-dispatch guard
    let x = f32(i32(gid3.x)) / upsbl_p.scale_x;
    let y = f32(i32(gid3.y)) / upsbl_p.scale_y;
    let epsilon = 1e-5;
    let in_w = upsbl_p.in_w;

    var i1: f32;
    var i2: f32;
    if (abs(x - round(x)) < epsilon) {
        let xr = clampu(i32(round(x)), u32(in_w));
        i1 = upsbl_in[clampu(i32(floor(y)), u32(upsbl_p.in_h)) * u32(in_w) + xr];
        i2 = upsbl_in[clampu(i32(ceil(y)),  u32(upsbl_p.in_h)) * u32(in_w) + xr];
    } else {
        let x0 = clampu(i32(floor(x)), u32(in_w));
        let x1 = clampu(i32(ceil(x)),  u32(in_w));
        let y0 = clampu(i32(floor(y)), u32(upsbl_p.in_h));
        let y1 = clampu(i32(ceil(y)),  u32(upsbl_p.in_h));
        let i11 = upsbl_in[y0 * u32(in_w) + x0];
        let i12 = upsbl_in[y1 * u32(in_w) + x0];
        let i21 = upsbl_in[y0 * u32(in_w) + x1];
        let i22 = upsbl_in[y1 * u32(in_w) + x1];
        i1 = (ceil(x) - x) * i11 + (x - floor(x)) * i21;
        i2 = (ceil(x) - x) * i12 + (x - floor(x)) * i22;
    }

    var i: f32;
    if (abs(y - round(y)) < epsilon) {
        i = i1;
    } else {
        i = (ceil(y) - y) * i1 + (y - floor(y)) * i2;
    }
    upsbl_out[i32(gid3.y) * upsbl_p.out_w + i32(gid3.x)] = i;
}

// ======================================================= add_texture_weighted
// Port of texture.metal:add_texture_weighted (verbatim).
// Params: w, h
struct AtwParams { w: i32, h: i32, };
@group(0) @binding(0) var<storage, read>       atw_t1: array<f32>;
@group(0) @binding(1) var<storage, read>       atw_t2: array<f32>;
@group(0) @binding(2) var<storage, read>       atw_wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> atw_out: array<f32>;
@group(0) @binding(4) var<uniform>             atw_p:  AtwParams;

@compute @workgroup_size(8, 8)
fn add_texture_weighted(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= atw_p.w || i32(gid3.y) >= atw_p.h) { return; }  // exact-dispatch guard
    let i = i32(gid3.y) * atw_p.w + i32(gid3.x);
    let intensity1 = atw_t1[i];
    let intensity2 = atw_t2[i];
    let weight = atw_wt[i];
    atw_out[i] = weight * intensity2 + (1.0 - weight) * intensity1;
}

// ================================================================ accumulate
// Port of texture.metal:add_texture: out += in / n_textures.
// Params: w, h, n
struct AccParams { w: i32, h: i32, n: f32, };
@group(0) @binding(0) var<storage, read>       acc_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> acc_out: array<f32>;
@group(0) @binding(2) var<uniform>             acc_p:   AccParams;

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= acc_p.w || i32(gid3.y) >= acc_p.h) { return; }  // exact-dispatch guard
    let i = i32(gid3.y) * acc_p.w + i32(gid3.x);
    acc_out[i] = acc_out[i] + acc_in[i] / acc_p.n;
}

// ============================================================ convert_to_u16
// Port of texture.metal:convert_float_to_uint16 (verbatim, with factor_16bit).
// Params: w, h, white_level, factor_16bit, mosaic
struct ConvParams { w: i32, h: i32, white_level: i32, factor_16bit: i32, mosaic: i32, };
@group(0) @binding(0) var<storage, read>       conv_in:    array<f32>;
@group(0) @binding(1) var<storage, read>       conv_black: array<f32>;
@group(0) @binding(2) var<storage, read_write> conv_out:   array<u32>;
@group(0) @binding(3) var<uniform>             conv_p:     ConvParams;

@compute @workgroup_size(8, 8)
fn convert_to_u16(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let x = i32(gid3.x);
    let y = i32(gid3.y);
    if (x >= conv_p.w || y >= conv_p.h) { return; }  // exact-dispatch guard
    let black_level = conv_black[u32((y % conv_p.mosaic) + conv_p.mosaic * (x % conv_p.mosaic))];
    var out_value = i32(round(f32(conv_p.factor_16bit) * (conv_in[y * conv_p.w + x] - black_level) + black_level));
    out_value = clampi(out_value, 0, min(conv_p.white_level, i32(UINT16_MAX_VAL)));
    conv_out[y * conv_p.w + x] = u32(out_value);
}

// ================================================================ crop
// Port of texture.metal:crop_texture (verbatim).
// Params: in_w, out_w, out_h, pad_left, pad_top
struct CropParams { in_w: i32, out_w: i32, out_h: i32, pad_left: i32, pad_top: i32, };
@group(0) @binding(0) var<storage, read>       crop_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> crop_out: array<f32>;
@group(0) @binding(2) var<uniform>             crop_p:   CropParams;

@compute @workgroup_size(8, 8)
fn crop(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= crop_p.out_w || i32(gid3.y) >= crop_p.out_h) { return; }  // exact-dispatch guard
    let x = i32(gid3.x) + crop_p.pad_left;
    let y = i32(gid3.y) + crop_p.pad_top;
    crop_out[i32(gid3.y) * crop_p.out_w + i32(gid3.x)] = crop_in[y * crop_p.in_w + x];
}

// ============================================================ area_pool_f32
// Area-average (fractional box) resampling of a plane to an arbitrary out_w × out_h.
// JS counterpart: viz.downsampleValues (used for small CPU-side thumbnail math).
// Params: in_w, in_h, out_w, out_h
struct PoolAParams { in_w: i32, in_h: i32, out_w: i32, out_h: i32, };
@group(0) @binding(0) var<storage, read>       pa_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> pa_out: array<f32>;
@group(0) @binding(2) var<uniform>             pa_p:   PoolAParams;

@compute @workgroup_size(8, 8)
fn area_pool_f32(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= pa_p.out_w || i32(gid3.y) >= pa_p.out_h) { return; }  // exact-dispatch guard
    let ox = i32(gid3.x);
    let oy = i32(gid3.y);
    let x0 = i32(floor(f32(ox) * f32(pa_p.in_w) / f32(pa_p.out_w)));
    let x1 = max(i32(floor(f32(ox + 1) * f32(pa_p.in_w) / f32(pa_p.out_w))), x0 + 1);
    let y0 = i32(floor(f32(oy) * f32(pa_p.in_h) / f32(pa_p.out_h)));
    let y1 = max(i32(floor(f32(oy + 1) * f32(pa_p.in_h) / f32(pa_p.out_h))), y0 + 1);
    var sum = 0.0;
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            sum += pa_in[y * pa_p.in_w + x];
        }
    }
    pa_out[oy * pa_p.out_w + ox] = sum / f32((y1 - y0) * (x1 - x0));
}

// ================================================================= diff_rgba8
// Port of viz.metal:signed_difference_rgba8 + a pooled readback path: tonemap the SIGNED
// difference (gain*comp + offset - ref) to red/blue (equality = black), area-average down to
// out_w × out_h, and pack as RGBA8 u32 (r | g<<8 | b<<16 | a<<24) for a cheap readback.
// Params: in_w, in_h, out_w, out_h, gain, offset, white_point
struct Diff8Params {
    in_w: i32, in_h: i32, out_w: i32, out_h: i32,
    gain: f32, offset: f32, white_point: f32,
};
@group(0) @binding(0) var<storage, read>       d8_ref:  array<f32>;
@group(0) @binding(1) var<storage, read>       d8_comp: array<f32>;
@group(0) @binding(2) var<storage, read_write> d8_out:  array<u32>;
@group(0) @binding(3) var<uniform>             d8_p:    Diff8Params;

@compute @workgroup_size(8, 8)
fn diff_rgba8(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= d8_p.out_w || i32(gid3.y) >= d8_p.out_h) { return; }  // exact-dispatch guard
    let ox = i32(gid3.x);
    let oy = i32(gid3.y);
    let x0 = i32(floor(f32(ox) * f32(d8_p.in_w) / f32(d8_p.out_w)));
    let x1 = max(i32(floor(f32(ox + 1) * f32(d8_p.in_w) / f32(d8_p.out_w))), x0 + 1);
    let y0 = i32(floor(f32(oy) * f32(d8_p.in_h) / f32(d8_p.out_h)));
    let y1 = max(i32(floor(f32(oy + 1) * f32(d8_p.in_h) / f32(d8_p.out_h))), y0 + 1);

    // average the signed difference over the source box (averaging the tonemapped
    // channels instead would let opposite-sign residuals cancel imperfectly)
    var sum = 0.0;
    let n = f32((y1 - y0) * (x1 - x0));
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            let i = y * d8_p.in_w + x;
            sum += (d8_p.gain * d8_comp[i] + d8_p.offset) - d8_ref[i];
        }
    }
    let difference = sum / n;
    let white = max(d8_p.white_point, 1e-9);
    let red = clamp(difference / white, 0.0, 1.0);
    let blue = clamp(-difference / white, 0.0, 1.0);
    let r = u32(red * 255.0 + 0.5);
    let g = 0u;
    let b = u32(blue * 255.0 + 0.5);
    d8_out[oy * d8_p.out_w + ox] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}

// ================================================================ gray_rgba8
// Port of viz.metal:gray_to_rgba8 + a pooled readback path: tonemap a single-channel plane to
// gray (value / white_point), area-average down to out_w × out_h, pack as RGBA8 u32.
// Averaging happens on the tonemapped values, exactly like gray_to_rgba8 + CPU box downsample
// of the resulting bitmap would (display-space averaging — matches the Swift pipeline).
// Params: in_w, in_h, out_w, out_h, white_point
struct Gray8Params { in_w: i32, in_h: i32, out_w: i32, out_h: i32, white_point: f32, };
@group(0) @binding(0) var<storage, read>       g8_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> g8_out: array<u32>;
@group(0) @binding(2) var<uniform>             g8_p:   Gray8Params;

@compute @workgroup_size(8, 8)
fn gray_rgba8(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= g8_p.out_w || i32(gid3.y) >= g8_p.out_h) { return; }  // exact-dispatch guard
    let ox = i32(gid3.x);
    let oy = i32(gid3.y);
    let x0 = i32(floor(f32(ox) * f32(g8_p.in_w) / f32(g8_p.out_w)));
    let x1 = max(i32(floor(f32(ox + 1) * f32(g8_p.in_w) / f32(g8_p.out_w))), x0 + 1);
    let y0 = i32(floor(f32(oy) * f32(g8_p.in_h) / f32(g8_p.out_h)));
    let y1 = max(i32(floor(f32(oy + 1) * f32(g8_p.in_h) / f32(g8_p.out_h))), y0 + 1);

    var sum = 0.0;
    let n = f32((y1 - y0) * (x1 - x0));
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            sum += clamp(g8_in[y * g8_p.in_w + x] / max(g8_p.white_point, 1e-9), 0.0, 1.0);
        }
    }
    let u = u32(clamp(sum / n, 0.0, 1.0) * 255.0 + 0.5);
    g8_out[oy * g8_p.out_w + ox] = u | (u << 8u) | (u << 16u) | (255u << 24u);
}

// =================================================================== diff_f32
// Port of viz.metal:abs_difference (incl. brightness-match gain/offset): full-resolution
// per-pixel absolute difference |ref - (gain*comp + offset)| as an f32 plane, for render-time
// statistics (mean |diff| via the sum kernels) and re-use in other render paths.
// Params: w, h, gain, offset
struct DiffFParams { w: i32, h: i32, gain: f32, offset: f32, };
@group(0) @binding(0) var<storage, read>       df_ref:  array<f32>;
@group(0) @binding(1) var<storage, read>       df_comp: array<f32>;
@group(0) @binding(2) var<storage, read_write> df_out:  array<f32>;
@group(0) @binding(3) var<uniform>             df_p:    DiffFParams;

@compute @workgroup_size(8, 8)
fn diff_f32(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= df_p.w || i32(gid3.y) >= df_p.h) { return; }  // exact-dispatch guard
    let i = i32(gid3.y) * df_p.w + i32(gid3.x);
    df_out[i] = abs(df_ref[i] - (df_p.gain * df_comp[i] + df_p.offset));
}

// ============================================== sum_partials / sum_final
// Two-stage reduction replacing texture_mean()'s sum_rect_columns + sum_row +
// divide_buffer chain. Sums an array<f32> down to a single scalar.
struct SumParams { n: u32, };
@group(0) @binding(0) var<storage, read>       sum_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> sum_out: array<f32>;
@group(0) @binding(2) var<uniform>             sum_p:   SumParams;

var<workgroup> reduce_aux: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_partials(@builtin(global_invocation_id) gid3: vec3<u32>,
                @builtin(local_invocation_id)  lid:  vec3<u32>,
                @builtin(workgroup_id)         wid:  vec3<u32>,
                @builtin(num_workgroups)       nw:   vec3<u32>) {
    let n = sum_p.n;
    let chunk = (n + nw.x - 1u) / nw.x;
    let start = wid.x * chunk;
    let end = min(start + chunk, n);
    var s = 0.0;
    for (var i = start + lid.x; i < end; i += 256u) { s += sum_in[i]; }
    reduce_aux[lid.x] = s;
    workgroupBarrier();
    for (var stride = 128u; stride >= 1u; stride = stride >> 1u) {
        if (lid.x < stride) { reduce_aux[lid.x] += reduce_aux[lid.x + stride]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { sum_out[wid.x] = reduce_aux[0]; }
}

@compute @workgroup_size(256)
fn sum_final(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(global_invocation_id) gid3: vec3<u32>) {
    let n = sum_p.n;
    reduce_aux[lid.x] = 0.0;
    var s = 0.0;
    for (var i = lid.x; i < n; i += 256u) { s += sum_in[i]; }
    reduce_aux[lid.x] = s;
    workgroupBarrier();
    for (var stride = 128u; stride >= 1u; stride = stride >> 1u) {
        if (lid.x < stride) { reduce_aux[lid.x] += reduce_aux[lid.x + stride]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { sum_out[0] = reduce_aux[0]; }
}
// ===========================================================================
// ==        FREQUENCY-DOMAIN MERGE (port of burstphoto/merge/frequency)    ==
// ===========================================================================
// Buffer layouts (all planes row-major, index = y * width + x):
//  - bayer canvas (f32):     the padded frame embedded at the per-iteration offset;
//                            exactly n_tiles_merge*16 bayer px wide/high
//  - rgba plane (vec4<f32>): 2×2 bayer cells packed into one vec4 (convert_to_rgba);
//                            dims = canvas dims / 2 = n_tiles_merge * 8
//  - FT plane (vec4<f32>):   per 8×8-rgba merge tile a 16-column strip of interleaved
//                            complex vec4s — Re at even x, Im at odd x — rows = tile
//                            rows; dims (canvas_w, canvas_h/2), i.e. 2× rgba width
// This port handles uniform-exposure bursts only, so the bracketed-exposure pieces of
// the original are omitted: exposure_corr1/corr2 are 1, calculate_highlights_norm
// writes 1.0 (→ dropped), and calculate_mismatch's exposure_factor is 1.

const PI: f32 = 3.14159265358979;

// ================================================================== place_f32
// Embed (or crop) a padded frame into a larger/smaller zero-filled canvas at the given
// offset — the per-iteration placement that the Swift implementation achieves by
// preparing each frame with asymmetric padding (prepare_texture pad_left/right/top/
// bottom). Keeping the alignment warp on the shared alignment padding instead means
// inspection alignment grids stay bit-identical and are reused as-is.
// Params: src_w, src_h, out_w, out_h, ox, oy
struct PlaceParams { src_w: i32, src_h: i32, out_w: i32, out_h: i32, ox: i32, oy: i32, };
@group(0) @binding(0) var<storage, read>       place_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> place_out: array<f32>;
@group(0) @binding(2) var<uniform>             place_p:   PlaceParams;

@compute @workgroup_size(8, 8)
fn place_f32(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let x = i32(gid3.x);
    let y = i32(gid3.y);
    if (x >= place_p.out_w || y >= place_p.out_h) { return; }  // exact-dispatch guard
    let sx = x - place_p.ox;
    let sy = y - place_p.oy;
    if (sx >= 0 && sx < place_p.src_w && sy >= 0 && sy < place_p.src_h) {
        place_out[y * place_p.out_w + x] = place_in[sy * place_p.src_w + sx];
    } else {
        place_out[y * place_p.out_w + x] = 0.0;
    }
}

// ================================================================== crop_rect
// Asymmetric crop of an f32 plane (out_w × out_h at offset (x0, y0)) — used for the
// final per-iteration canvas crop (pad_merge + shift margins) and the mismatch-mean
// tile rectangle (Swift: crop_texture + texture_mean).
// Params: in_w, out_w, out_h, x0, y0
struct CropRParams { in_w: i32, out_w: i32, out_h: i32, x0: i32, y0: i32, };
@group(0) @binding(0) var<storage, read>       cropr_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> cropr_out: array<f32>;
@group(0) @binding(2) var<uniform>             cropr_p:   CropRParams;

@compute @workgroup_size(8, 8)
fn crop_rect(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= cropr_p.out_w || i32(gid3.y) >= cropr_p.out_h) { return; }  // exact-dispatch guard
    let x = i32(gid3.x) + cropr_p.x0;
    let y = i32(gid3.y) + cropr_p.y0;
    cropr_out[i32(gid3.y) * cropr_p.out_w + i32(gid3.x)] = cropr_in[y * cropr_p.in_w + x];
}

// ======================================================== convert_to_rgba (freq)
// Port of texture.metal:convert_to_rgba for the merge canvas: pack each 2×2 bayer cell
// into one vec4. The Swift version's crop_merge trim is already applied by the canvas
// placement, so this is a pure repack.
// Params: in_w (canvas width), out_w, out_h
struct C4Params { in_w: i32, out_w: i32, out_h: i32, };
@group(0) @binding(0) var<storage, read>       c4_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> c4_out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             c4_p:   C4Params;

@compute @workgroup_size(8, 8)
fn convert_to_rgba(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= c4_p.out_w || i32(gid3.y) >= c4_p.out_h) { return; }  // exact-dispatch guard
    let x = i32(gid3.x) * 2;
    let y = i32(gid3.y) * 2;
    let i0 = y * c4_p.in_w + x;
    c4_out[i32(gid3.y) * c4_p.out_w + i32(gid3.x)] =
        vec4<f32>(c4_in[i0], c4_in[i0 + 1], c4_in[i0 + c4_p.in_w], c4_in[i0 + c4_p.in_w + 1]);
}

// ======================================================== convert_to_bayer (freq)
// Inverse of convert_to_rgba: spread each vec4 back over its 2×2 bayer cell.
// Params: out_w (bayer canvas width)
struct CbParams { out_w: i32, };
@group(0) @binding(0) var<storage, read>       cb_in:  array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cb_out: array<f32>;
@group(0) @binding(2) var<uniform>             cb_p:   CbParams;

@compute @workgroup_size(8, 8)
fn convert_to_bayer(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let x = i32(gid3.x) * 2;
    let y = i32(gid3.y) * 2;
    let color = cb_in[i32(gid3.y) * (cb_p.out_w / 2) + i32(gid3.x)];
    cb_out[y * cb_p.out_w + x] = color[0];
    cb_out[y * cb_p.out_w + x + 1] = color[1];
    cb_out[(y + 1) * cb_p.out_w + x] = color[2];
    cb_out[(y + 1) * cb_p.out_w + x + 1] = color[3];
}

// ========================================================= calculate_rms_rgba
// Port of frequency.metal:calculate_rms_rgba — per-merge-tile noise estimate of the
// reference (shot-noise proxy; verbatim including the unusual normalization).
// Params: rgba_w, n_tiles_x, n_tiles_y
struct RmsParams { rgba_w: i32, n_tiles_x: i32, n_tiles_y: i32, };
@group(0) @binding(0) var<storage, read>       rms_in:  array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> rms_out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             rms_p:   RmsParams;

@compute @workgroup_size(8, 8)
fn calculate_rms_rgba(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= rms_p.n_tiles_x || i32(gid3.y) >= rms_p.n_tiles_y) { return; }  // exact-dispatch guard
    let tile_size = 8;
    let x0 = i32(gid3.x) * tile_size;
    let y0 = i32(gid3.y) * tile_size;
    var noise_est = vec4<f32>(0.0);
    for (var dy = 0; dy < tile_size; dy++) {
        for (var dx = 0; dx < tile_size; dx++) {
            let d = rms_in[(y0 + dy) * rms_p.rgba_w + x0 + dx];
            noise_est += d * d;
        }
    }
    noise_est = 0.25 * sqrt(noise_est) / f32(tile_size);
    rms_out[i32(gid3.y) * rms_p.n_tiles_x + i32(gid3.x)] = noise_est;
}

// ====================================================== calculate_abs_diff_rgba
// Port of frequency.metal:calculate_abs_diff_rgba (element-wise).
// Params: w, h (rgba plane dims)
struct AdParams { w: i32, h: i32, };
@group(0) @binding(0) var<storage, read>       ad_in1: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       ad_in2: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> ad_out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             ad_p:   AdParams;

@compute @workgroup_size(8, 8)
fn calculate_abs_diff_rgba(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= ad_p.w || i32(gid3.y) >= ad_p.h) { return; }  // exact-dispatch guard
    let i = gid3.y * u32(ad_p.w) + gid3.x;
    ad_out[i] = abs(ad_in1[i] - ad_in2[i]);
}

// ==================================================== calculate_mismatch_rgba
// Port of frequency.metal:calculate_mismatch_rgba — motion mismatch per merge tile:
// raised-cosine-windowed absolute difference over a 2× tile-size support, normalized by
// the tile noise estimate (uniform exposure: exposure_factor = 1).
// Params: rgba_w, rgba_h, n_tiles_x
struct MmParams { rgba_w: i32, rgba_h: i32, n_tiles_x: i32, };
@group(0) @binding(0) var<storage, read>       mm_absdiff:  array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       mm_rms:      array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> mm_mismatch: array<f32>;
@group(0) @binding(3) var<uniform>             mm_p:        MmParams;

@compute @workgroup_size(8, 8)
fn calculate_mismatch_rgba(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= mm_p.n_tiles_x || i32(gid3.y) >= (mm_p.rgba_h + 7) / 8) { return; }  // exact-dispatch guard
    let tile_size = 8;
    let x0 = i32(gid3.x) * tile_size;
    let y0 = i32(gid3.y) * tile_size;
    let noise_est = mm_rms[i32(gid3.y) * mm_p.n_tiles_x + i32(gid3.x)];

    // spatial support twice the merge tile size, clamped at the frame borders
    let x_start = max(0, x0 - tile_size / 2);
    let y_start = max(0, y0 - tile_size / 2);
    let x_end = min(mm_p.rgba_w - 1, x0 + tile_size * 3 / 2) + 1;
    let y_end = min(mm_p.rgba_h - 1, y0 + tile_size * 3 / 2) + 1;
    let x_shift = -(x0 - tile_size / 2);
    let y_shift = -(y0 - tile_size / 2);

    let angle = -2.0 * PI / f32(tile_size);
    var tile_diff = vec4<f32>(0.0);
    var n_total = 0.0;
    for (var dy = y_start; dy < y_end; dy++) {
        for (var dx = x_start; dx < x_end; dx++) {
            let norm_cosine =
                (0.5 - 0.17 * cos(-angle * (f32(dx + x_shift) + 0.5))) *
                (0.5 - 0.17 * cos(-angle * (f32(dy + y_shift) + 0.5)));
            tile_diff += norm_cosine * mm_absdiff[dy * mm_p.rgba_w + dx];
            n_total += norm_cosine;
        }
    }
    tile_diff /= n_total;

    // mismatch ratio (quadratic Wiener terms deliberately not used — see the original's
    // comment); uniform exposure: sqrt(0.5*n + 0.5*n/1 + 1)
    let mismatch4 = tile_diff / sqrt(0.5 * noise_est + 0.5 * noise_est + vec4<f32>(1.0));
    let mismatch = 0.25 * (mismatch4[0] + mismatch4[1] + mismatch4[2] + mismatch4[3]);
    mm_mismatch[i32(gid3.y) * mm_p.n_tiles_x + i32(gid3.x)] = mismatch;
}

// ========================================================= normalize_mismatch
// Port of frequency.metal:normalize_mismatch — scale so the interior mean maps to 0.12
// and clamp to [0, 1]. mean_mismatch is computed on the interior tile rectangle (CPU).
// Params: n (plane elements), mean_mismatch
struct NmParams { n: u32, mean_mismatch: f32, };
@group(0) @binding(0) var<storage, read_write> nm_mismatch: array<f32>;
@group(0) @binding(1) var<uniform>             nm_p:       NmParams;

@compute @workgroup_size(64)
fn normalize_mismatch(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let i = gid3.x;
    if (i >= nm_p.n) { return; }  // exact-dispatch guard
    var m = nm_mismatch[i];
    m *= 0.12 / (nm_p.mean_mismatch + 1e-12);
    nm_mismatch[i] = clamp(m, 0.0, 1.0);
}

// ================================================================ forward_fft
// Port of frequency.metal:forward_fft (8×8 real forward DFT with raised-cosine tile
// windowing; exploits conjugate symmetry to transform rows 0..4 and mirror the rest).
// One thread per merge tile. Params: rgba_w, ft_w, n_tiles_x, n_tiles_y
struct FftParams { rgba_w: i32, ft_w: i32, n_tiles_x: i32, n_tiles_y: i32, norm: f32, };
@group(0) @binding(0) var<storage, read>       ff_in:  array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> ff_out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             ff_p:   FftParams;

@compute @workgroup_size(8, 8)
fn forward_fft(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= ff_p.n_tiles_x || i32(gid3.y) >= ff_p.n_tiles_y) { return; }  // exact-dispatch guard
    let tile_size = 8;
    let m0 = i32(gid3.x) * tile_size;
    let n0 = i32(gid3.y) * tile_size;
    let ts14 = tile_size / 4;
    let ts24 = tile_size / 2;
    let ts34 = ts14 * 3;
    let angle = -2.0 * PI / f32(tile_size);

    var tmp_data: array<vec4<f32>, 16>;
    var tmp_tile: array<vec4<f32>, 80>;
    let zeros4 = vec4<f32>(0.0);

    // column-wise one-dimensional DFT along y; the input is real, so only rows 0..tile/2
    // have to be computed (the rest follows from conjugate symmetry)
    for (var dm = 0; dm < tile_size; dm += 2) {
        let m = m0 + dm;
        for (var dn = 0; dn < tile_size; dn++) {
            tmp_data[2 * dn + 0] = ff_in[(n0 + dn) * ff_p.rgba_w + m];
            tmp_data[2 * dn + 1] = ff_in[(n0 + dn) * ff_p.rgba_w + m + 1];
        }
        for (var dn = 0; dn <= tile_size / 2; dn++) {
            let n_tmp = dn * 2 * tile_size;
            var Re0 = zeros4; var Im0 = zeros4;
            var Re1 = zeros4; var Im1 = zeros4;
            for (var dy = 0; dy < tile_size; dy++) {
                // modified raised cosine window for tile blending (overlapped tiles)
                let norm_cosine0 = (0.5 - 0.5 * cos(-angle * (f32(dm) + 0.5))) * (0.5 - 0.5 * cos(-angle * (f32(dy) + 0.5)));
                let norm_cosine1 = (0.5 - 0.5 * cos(-angle * (f32(dm) + 1.5))) * (0.5 - 0.5 * cos(-angle * (f32(dy) + 0.5)));
                let coefRe = cos(angle * f32(dn * dy));
                let coefIm = sin(angle * f32(dn * dy));
                let dataRe0 = norm_cosine0 * tmp_data[2 * dy + 0];
                Re0 += coefRe * dataRe0;
                Im0 += coefIm * dataRe0;
                let dataRe1 = norm_cosine1 * tmp_data[2 * dy + 1];
                Re1 += coefRe * dataRe1;
                Im1 += coefIm * dataRe1;
            }
            tmp_tile[n_tmp + 2 * dm + 0] = Re0;
            tmp_tile[n_tmp + 2 * dm + 1] = Im0;
            tmp_tile[n_tmp + 2 * dm + 2] = Re1;
            tmp_tile[n_tmp + 2 * dm + 3] = Im1;
        }
    }

    // row-wise FFT along x (rows 0..tile/2 only); four stride-4 sub-DFTs + two butterfly
    // stages. Note the deliberate Re1/Re2 swap of the sub-DFT accumulators, matching the
    // Metal original's output bin mapping.
    for (var dn = 0; dn <= tile_size / 2; dn++) {
        let n = n0 + dn;
        for (var dm = 0; dm < tile_size; dm++) {
            tmp_data[2 * dm + 0] = tmp_tile[dn * 2 * tile_size + 2 * dm + 0];
            tmp_data[2 * dm + 1] = tmp_tile[dn * 2 * tile_size + 2 * dm + 1];
        }
        for (var dm = 0; dm < tile_size / 4; dm++) {
            let m = 2 * (m0 + dm);
            var Re0 = zeros4; var Im0 = zeros4;
            var Re1 = zeros4; var Im1 = zeros4;
            var Re2 = zeros4; var Im2 = zeros4;
            var Re3 = zeros4; var Im3 = zeros4;
            var Re00: vec4<f32>; var Re11: vec4<f32>; var Re22: vec4<f32>; var Re33: vec4<f32>;
            var Im00: vec4<f32>; var Im11: vec4<f32>; var Im22: vec4<f32>; var Im33: vec4<f32>;
            for (var dx = 0; dx < tile_size; dx += 4) {
                let coefRe = cos(angle * f32(dm * dx));
                let coefIm = sin(angle * f32(dm * dx));
                var dataRe: vec4<f32>; var dataIm: vec4<f32>;
                // DFT0
                dataRe = tmp_data[2 * dx + 0]; dataIm = tmp_data[2 * dx + 1];
                Re0 += (coefRe * dataRe - coefIm * dataIm);
                Im0 += (coefIm * dataRe + coefRe * dataIm);
                // DFT1
                dataRe = tmp_data[2 * dx + 2]; dataIm = tmp_data[2 * dx + 3];
                Re2 += (coefRe * dataRe - coefIm * dataIm);
                Im2 += (coefIm * dataRe + coefRe * dataIm);
                // DFT2
                dataRe = tmp_data[2 * dx + 4]; dataIm = tmp_data[2 * dx + 5];
                Re1 += (coefRe * dataRe - coefIm * dataIm);
                Im1 += (coefIm * dataRe + coefRe * dataIm);
                // DFT3
                dataRe = tmp_data[2 * dx + 6]; dataIm = tmp_data[2 * dx + 7];
                Re3 += (coefRe * dataRe - coefIm * dataIm);
                Im3 += (coefIm * dataRe + coefRe * dataIm);
            }

            // first butterfly
            var bRe = cos(angle * 2.0 * f32(dm));
            var bIm = sin(angle * 2.0 * f32(dm));
            Re00 = Re0 + bRe * Re1 - bIm * Im1;
            Im00 = Im0 + bIm * Re1 + bRe * Im1;
            Re22 = Re2 + bRe * Re3 - bIm * Im3;
            Im22 = Im2 + bIm * Re3 + bRe * Im3;
            bRe = cos(angle * 2.0 * f32(dm + ts14));
            bIm = sin(angle * 2.0 * f32(dm + ts14));
            Re11 = Re0 + bRe * Re1 - bIm * Im1;
            Im11 = Im0 + bIm * Re1 + bRe * Im1;
            Re33 = Re2 + bRe * Re3 - bIm * Im3;
            Im33 = Im2 + bIm * Re3 + bRe * Im3;

            // second butterfly
            bRe = cos(angle * f32(dm)); bIm = sin(angle * f32(dm));
            Re0 = Re00 + bRe * Re22 - bIm * Im22;
            Im0 = Im00 + bIm * Re22 + bRe * Im22;
            bRe = cos(angle * f32(dm + ts24)); bIm = sin(angle * f32(dm + ts24));
            Re2 = Re00 + bRe * Re22 - bIm * Im22;
            Im2 = Im00 + bIm * Re22 + bRe * Im22;
            bRe = cos(angle * f32(dm + ts14)); bIm = sin(angle * f32(dm + ts14));
            Re1 = Re11 + bRe * Re33 - bIm * Im33;
            Im1 = Im11 + bIm * Re33 + bRe * Im33;
            bRe = cos(angle * f32(dm + ts34)); bIm = sin(angle * f32(dm + ts34));
            Re3 = Re11 + bRe * Re33 - bIm * Im33;
            Im3 = Im11 + bIm * Re33 + bRe * Im33;

            ff_out[n * ff_p.ft_w + m + 0] = Re0;
            ff_out[n * ff_p.ft_w + m + 1] = Im0;
            ff_out[n * ff_p.ft_w + m + ts24 + 0] = Re1;
            ff_out[n * ff_p.ft_w + m + ts24 + 1] = Im1;
            ff_out[n * ff_p.ft_w + m + tile_size + 0] = Re2;
            ff_out[n * ff_p.ft_w + m + tile_size + 1] = Im2;
            ff_out[n * ff_p.ft_w + m + ts24 * 3 + 0] = Re3;
            ff_out[n * ff_p.ft_w + m + ts24 * 3 + 1] = Im3;

            // conjugate-symmetric rows
            if (dn > 0 && dn != tile_size / 2) {
                let n2 = n0 + tile_size - dn;
                let m20 = 2 * (m0 + min(dm, 1) * (tile_size - dm));
                let m21 = 2 * (m0 + tile_size - dm - ts14);
                let m22 = 2 * (m0 + tile_size - dm - ts24);
                let m23 = 2 * (m0 + tile_size - dm - ts14 * 3);
                ff_out[n2 * ff_p.ft_w + m20 + 0] = Re0;
                ff_out[n2 * ff_p.ft_w + m20 + 1] = -Im0;
                ff_out[n2 * ff_p.ft_w + m21 + 0] = Re1;
                ff_out[n2 * ff_p.ft_w + m21 + 1] = -Im1;
                ff_out[n2 * ff_p.ft_w + m22 + 0] = Re2;
                ff_out[n2 * ff_p.ft_w + m22 + 1] = -Im2;
                ff_out[n2 * ff_p.ft_w + m23 + 0] = Re3;
                ff_out[n2 * ff_p.ft_w + m23 + 1] = -Im3;
            }
        }
    }
}

// ================================================================ backward_fft
// Port of frequency.metal:backward_fft — exact inverse of forward_fft (up to the
// 1/(n_textures·tile²) normalization (the norm field), so each iteration's output is the
// burst average including the reference, exactly like Swift's backward_ft).
@group(0) @binding(0) var<storage, read>       bf_in:  array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> bf_out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             bf_p:   FftParams;

@compute @workgroup_size(8, 8)
fn backward_fft(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= bf_p.n_tiles_x || i32(gid3.y) >= bf_p.n_tiles_y) { return; }  // exact-dispatch guard
    let tile_size = 8;
    let m0 = i32(gid3.x) * tile_size;
    let n0 = i32(gid3.y) * tile_size;
    let ts14 = tile_size / 4;
    let ts24 = tile_size / 2;
    let ts34 = ts14 * 3;
    let angle = -2.0 * PI / f32(tile_size);

    var tmp_data: array<vec4<f32>, 16>;
    var tmp_tile: array<vec4<f32>, 128>;
    let zeros4 = vec4<f32>(0.0);
    let norm_factor = bf_p.norm;  // n_textures·tile_size²

    // row-wise inverse FFT along x (all tile rows)
    for (var dn = 0; dn < tile_size; dn++) {
        let n_tmp = dn * 2 * tile_size;
        for (var dm = 0; dm < tile_size; dm++) {
            tmp_data[2 * dm + 0] = bf_in[(n0 + dn) * bf_p.ft_w + 2 * (m0 + dm) + 0];
            tmp_data[2 * dm + 1] = bf_in[(n0 + dn) * bf_p.ft_w + 2 * (m0 + dm) + 1];
        }
        for (var dm = 0; dm < tile_size / 4; dm++) {
            var Re0 = zeros4; var Im0 = zeros4;
            var Re1 = zeros4; var Im1 = zeros4;
            var Re2 = zeros4; var Im2 = zeros4;
            var Re3 = zeros4; var Im3 = zeros4;
            var Re00: vec4<f32>; var Re11: vec4<f32>; var Re22: vec4<f32>; var Re33: vec4<f32>;
            var Im00: vec4<f32>; var Im11: vec4<f32>; var Im22: vec4<f32>; var Im33: vec4<f32>;
            for (var dx = 0; dx < tile_size; dx += 4) {
                let coefRe = cos(angle * f32(dm * dx));
                let coefIm = sin(angle * f32(dm * dx));
                var dataRe: vec4<f32>; var dataIm: vec4<f32>;
                // DFT0
                dataRe = tmp_data[2 * dx + 0]; dataIm = tmp_data[2 * dx + 1];
                Re0 += (coefRe * dataRe + coefIm * dataIm);
                Im0 += (coefIm * dataRe - coefRe * dataIm);
                // DFT1
                dataRe = tmp_data[2 * dx + 2]; dataIm = tmp_data[2 * dx + 3];
                Re2 += (coefRe * dataRe + coefIm * dataIm);
                Im2 += (coefIm * dataRe - coefRe * dataIm);
                // DFT2
                dataRe = tmp_data[2 * dx + 4]; dataIm = tmp_data[2 * dx + 5];
                Re1 += (coefRe * dataRe + coefIm * dataIm);
                Im1 += (coefIm * dataRe - coefRe * dataIm);
                // DFT3
                dataRe = tmp_data[2 * dx + 6]; dataIm = tmp_data[2 * dx + 7];
                Re3 += (coefRe * dataRe + coefIm * dataIm);
                Im3 += (coefIm * dataRe - coefRe * dataIm);
            }

            // first butterfly
            var bRe = cos(angle * 2.0 * f32(dm));
            var bIm = sin(angle * 2.0 * f32(dm));
            Re00 = Re0 + bRe * Re1 - bIm * Im1;
            Im00 = Im0 + bIm * Re1 + bRe * Im1;
            Re22 = Re2 + bRe * Re3 - bIm * Im3;
            Im22 = Im2 + bIm * Re3 + bRe * Im3;
            bRe = cos(angle * 2.0 * f32(dm + ts14));
            bIm = sin(angle * 2.0 * f32(dm + ts14));
            Re11 = Re0 + bRe * Re1 - bIm * Im1;
            Im11 = Im0 + bIm * Re1 + bRe * Im1;
            Re33 = Re2 + bRe * Re3 - bIm * Im3;
            Im33 = Im2 + bIm * Re3 + bRe * Im3;

            // second butterfly
            bRe = cos(angle * f32(dm)); bIm = sin(angle * f32(dm));
            Re0 = Re00 + bRe * Re22 - bIm * Im22;
            Im0 = Im00 + bIm * Re22 + bRe * Im22;
            bRe = cos(angle * f32(dm + ts24)); bIm = sin(angle * f32(dm + ts24));
            Re2 = Re00 + bRe * Re22 - bIm * Im22;
            Im2 = Im00 + bIm * Re22 + bRe * Im22;
            bRe = cos(angle * f32(dm + ts14)); bIm = sin(angle * f32(dm + ts14));
            Re1 = Re11 + bRe * Re33 - bIm * Im33;
            Im1 = Im11 + bIm * Re33 + bRe * Im33;
            bRe = cos(angle * f32(dm + ts34)); bIm = sin(angle * f32(dm + ts34));
            Re3 = Re11 + bRe * Re33 - bIm * Im33;
            Im3 = Im11 + bIm * Re33 + bRe * Im33;

            tmp_tile[n_tmp + 2 * dm + 0] = Re0;
            tmp_tile[n_tmp + 2 * dm + 1] = -Im0;
            tmp_tile[n_tmp + 2 * dm + ts24 + 0] = Re1;
            tmp_tile[n_tmp + 2 * dm + ts24 + 1] = -Im1;
            tmp_tile[n_tmp + 2 * dm + tile_size + 0] = Re2;
            tmp_tile[n_tmp + 2 * dm + tile_size + 1] = -Im2;
            tmp_tile[n_tmp + 2 * dm + ts24 * 3 + 0] = Re3;
            tmp_tile[n_tmp + 2 * dm + ts24 * 3 + 1] = -Im3;
        }
    }

    // column-wise inverse DFT along y (real output: Im discarded)
    for (var dm = 0; dm < tile_size; dm++) {
        let m = m0 + dm;
        for (var dn = 0; dn < tile_size; dn++) {
            tmp_data[2 * dn + 0] = tmp_tile[dn * 2 * tile_size + 2 * dm + 0];
            tmp_data[2 * dn + 1] = tmp_tile[dn * 2 * tile_size + 2 * dm + 1];
        }
        for (var dn = 0; dn < tile_size / 4; dn++) {
            let n = n0 + dn;
            var Re0 = zeros4; var Im0 = zeros4;
            var Re1 = zeros4; var Im1 = zeros4;
            var Re2 = zeros4; var Im2 = zeros4;
            var Re3 = zeros4; var Im3 = zeros4;
            var Re00: vec4<f32>; var Re11: vec4<f32>; var Re22: vec4<f32>; var Re33: vec4<f32>;
            var Im00: vec4<f32>; var Im11: vec4<f32>; var Im22: vec4<f32>; var Im33: vec4<f32>;
            for (var dy = 0; dy < tile_size; dy += 4) {
                let coefRe = cos(angle * f32(dn * dy));
                let coefIm = sin(angle * f32(dn * dy));
                var dataRe: vec4<f32>; var dataIm: vec4<f32>;
                // DFT0
                dataRe = tmp_data[2 * dy + 0]; dataIm = tmp_data[2 * dy + 1];
                Re0 += (coefRe * dataRe + coefIm * dataIm);
                Im0 += (coefIm * dataRe - coefRe * dataIm);
                // DFT1
                dataRe = tmp_data[2 * dy + 2]; dataIm = tmp_data[2 * dy + 3];
                Re2 += (coefRe * dataRe + coefIm * dataIm);
                Im2 += (coefIm * dataRe - coefRe * dataIm);
                // DFT2
                dataRe = tmp_data[2 * dy + 4]; dataIm = tmp_data[2 * dy + 5];
                Re1 += (coefRe * dataRe + coefIm * dataIm);
                Im1 += (coefIm * dataRe - coefRe * dataIm);
                // DFT3
                dataRe = tmp_data[2 * dy + 6]; dataIm = tmp_data[2 * dy + 7];
                Re3 += (coefRe * dataRe + coefIm * dataIm);
                Im3 += (coefIm * dataRe - coefRe * dataIm);
            }

            // first butterfly
            var bRe = cos(angle * 2.0 * f32(dn));
            var bIm = sin(angle * 2.0 * f32(dn));
            Re00 = Re0 + bRe * Re1 - bIm * Im1;
            Im00 = Im0 + bIm * Re1 + bRe * Im1;
            Re22 = Re2 + bRe * Re3 - bIm * Im3;
            Im22 = Im2 + bIm * Re3 + bRe * Im3;
            bRe = cos(angle * 2.0 * f32(dn + ts14));
            bIm = sin(angle * 2.0 * f32(dn + ts14));
            Re11 = Re0 + bRe * Re1 - bIm * Im1;
            Im11 = Im0 + bIm * Re1 + bRe * Im1;
            Re33 = Re2 + bRe * Re3 - bIm * Im3;
            Im33 = Im2 + bIm * Re3 + bRe * Im3;

            // second butterfly
            bRe = cos(angle * f32(dn)); bIm = sin(angle * f32(dn));
            Re0 = Re00 + bRe * Re22 - bIm * Im22;
            bRe = cos(angle * f32(dn + ts24)); bIm = sin(angle * f32(dn + ts24));
            Re2 = Re00 + bRe * Re22 - bIm * Im22;
            bRe = cos(angle * f32(dn + ts14)); bIm = sin(angle * f32(dn + ts14));
            Re1 = Re11 + bRe * Re33 - bIm * Im33;
            bRe = cos(angle * f32(dn + ts34)); bIm = sin(angle * f32(dn + ts34));
            Re3 = Re11 + bRe * Re33 - bIm * Im33;

            let r0 = Re0 / norm_factor;
            let r1 = Re1 / norm_factor;
            let r2 = Re2 / norm_factor;
            let r3 = Re3 / norm_factor;
            bf_out[n * bf_p.rgba_w + m] = r0;
            bf_out[(n + ts14) * bf_p.rgba_w + m] = r1;
            bf_out[(n + ts24) * bf_p.rgba_w + m] = r2;
            bf_out[(n + ts34) * bf_p.rgba_w + m] = r3;
        }
    }
}

// =================================================== merge_frequency_domain
// Port of frequency.metal:merge_frequency_domain — the core pairwise Wiener merge with
// Fourier-shift-theorem sub-pixel alignment (7×7 candidate shifts per merge tile).
// uniform exposure: uniform_exposure == 1 and highlights_norm == 1 (dropped).
// Params: ft_w, n_tiles_x, n_tiles_y, robustness_norm, read_noise, max_motion_norm
struct MfParams {
    ft_w: i32, n_tiles_x: i32, n_tiles_y: i32,
    robustness_norm: f32, read_noise: f32, max_motion_norm: f32,
};
@group(0) @binding(0) var<storage, read>       mf_ref:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       mf_aligned:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> mf_out:      array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       mf_rms:      array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       mf_mismatch: array<f32>;
@group(0) @binding(5) var<uniform>             mf_p:        MfParams;

@compute @workgroup_size(8, 8)
fn merge_frequency_domain(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= mf_p.n_tiles_x || i32(gid3.y) >= mf_p.n_tiles_y) { return; }  // exact-dispatch guard
    let tile_size = 8;
    let m0 = i32(gid3.x) * tile_size;
    let n0 = i32(gid3.y) * tile_size;
    let tile = i32(gid3.y) * mf_p.n_tiles_x + i32(gid3.x);

    // combine estimated shot noise and read noise; normalize with tile size + robustness
    let noise_est = mf_rms[tile] + vec4<f32>(mf_p.read_noise);
    let noise_norm = noise_est * f32(tile_size) * f32(tile_size) * mf_p.robustness_norm;

    // motion norm from the mismatch texture (Liba 2019 Fig. 9f-style linear ramp)
    let mismatch = mf_mismatch[tile];
    let mismatch_weight = clamp(1.0 - 10.0 * (mismatch - 0.2), 0.0, 1.0);
    let motion_norm = clamp(mf_p.max_motion_norm - (mismatch - 0.02) * (mf_p.max_motion_norm - 1.0) / 0.15, 1.0, mf_p.max_motion_norm);

    let angle = -2.0 * PI / f32(tile_size);
    let shift_step_size = 1.0 / 6.0;

    var total_diff: array<f32, 49>;
    for (var i = 0; i < 49; i++) { total_diff[i] = 0.0; }

    // sub-pixel alignment via the Fourier shift theorem: test shifts between -0.5 and
    // +0.5 px on the per-color-channel pixel scale (-1..+1 px on the sensor scale)
    for (var dn = 0; dn < tile_size; dn++) {
        for (var dm = 0; dm < tile_size; dm++) {
            let m = 2 * (m0 + dm);
            let n = n0 + dn;
            let refRe = mf_ref[n * mf_p.ft_w + m];
            let refIm = mf_ref[n * mf_p.ft_w + m + 1];
            let alignedRe = mf_aligned[n * mf_p.ft_w + m];
            let alignedIm = mf_aligned[n * mf_p.ft_w + m + 1];
            for (var i = 0; i < 49; i++) {
                let shift_x = -0.5 + f32(i % 7) * shift_step_size;
                let shift_y = -0.5 + f32(i / 7) * shift_step_size;
                let coefRe = cos(angle * (f32(dm) * shift_x + f32(dn) * shift_y));
                let coefIm = sin(angle * (f32(dm) * shift_x + f32(dn) * shift_y));
                let aRe2 = refRe - (coefRe * alignedRe - coefIm * alignedIm);
                let aIm2 = refIm - (coefIm * alignedRe + coefRe * alignedIm);
                let w4 = aRe2 * aRe2 + aIm2 * aIm2;
                total_diff[i] += w4[0] + w4[1] + w4[2] + w4[3];
            }
        }
    }

    var best_diff = 1e20;
    var best_i = 0;
    for (var i = 0; i < 49; i++) {
        if (total_diff[i] < best_diff) { best_diff = total_diff[i]; best_i = i; }
    }
    let best_shift_x = -0.5 + f32(best_i % 7) * shift_step_size;
    let best_shift_y = -0.5 + f32(best_i / 7) * shift_step_size;

    // pairwise merge of the two tiles (Hasinoff 2016 eq. 6 / Monod 2021 eq. 10)
    for (var dn = 0; dn < tile_size; dn++) {
        for (var dm = 0; dm < tile_size; dm++) {
            let m = 2 * (m0 + dm);
            let n = n0 + dn;
            let refRe = mf_ref[n * mf_p.ft_w + m];
            let refIm = mf_ref[n * mf_p.ft_w + m + 1];
            let alignedRe = mf_aligned[n * mf_p.ft_w + m];
            let alignedIm = mf_aligned[n * mf_p.ft_w + m + 1];

            let coefRe = cos(angle * (f32(dm) * best_shift_x + f32(dn) * best_shift_y));
            let coefIm = sin(angle * (f32(dm) * best_shift_x + f32(dn) * best_shift_y));
            let aRe2 = coefRe * alignedRe - coefIm * alignedIm;
            let aIm2 = coefIm * alignedRe + coefRe * alignedIm;

            // sharper tiles (larger frequency magnitude relative to the reference) merge
            // with more weight (Delbracio 2015 eq. 3); skipped at zero frequency and for
            // mismatched tiles (uniform_exposure == 1, so the boost is active)
            var magnitude_norm = 1.0;
            if (dm + dn > 0 && mismatch < 0.3) {
                let refMag = sqrt(refRe * refRe + refIm * refIm);
                let alignedMag2 = sqrt(aRe2 * aRe2 + aIm2 * aIm2);
                let ratio_mag = (alignedMag2[0] + alignedMag2[1] + alignedMag2[2] + alignedMag2[3])
                              / (refMag[0] + refMag[1] + refMag[2] + refMag[3]);
                magnitude_norm = mismatch_weight * clamp(ratio_mag * ratio_mag * ratio_mag * ratio_mag, 0.5, 3.0);
            }

            // Wiener shrinkage weight; per-channel weights → mean of the two central
            // values (a slight robustness tweak over the publications' max)
            var weight4 = (refRe - aRe2) * (refRe - aRe2) + (refIm - aIm2) * (refIm - aIm2);
            weight4 = weight4 / (weight4 + magnitude_norm * motion_norm * noise_norm);
            let min_weight = min(weight4[0], min(weight4[1], min(weight4[2], weight4[3])));
            let max_weight = max(weight4[0], max(weight4[1], max(weight4[2], weight4[3])));
            let weight = clamp(0.5 * (weight4[0] + weight4[1] + weight4[2] + weight4[3] - min_weight - max_weight), 0.0, 1.0);

            let idx_re = n * mf_p.ft_w + m;
            mf_out[idx_re] = mf_out[idx_re] + (1.0 - weight) * aRe2 + weight * refRe;
            mf_out[idx_re + 1] = mf_out[idx_re + 1] + (1.0 - weight) * aIm2 + weight * refIm;
        }
    }
}

// ============================================ deconvolute_frequency_domain
// Port of frequency.metal:deconvolute_frequency_domain — mild frequency-dependent gain
// to correct blur from sub-pixel misalignment; weighted down for mismatched tiles and
// strong frequencies. Fixed gains for the 8×8 merge tile (the 16-table in the original
// is unreachable: tile_size_merge is pinned to 8).
@group(0) @binding(0) var<storage, read_write> dc_ft:       array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       dc_mismatch: array<f32>;
@group(0) @binding(2) var<uniform>             dc_p:        MfParams;

@compute @workgroup_size(8, 8)
fn deconvolute_frequency_domain(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= dc_p.n_tiles_x || i32(gid3.y) >= dc_p.n_tiles_y) { return; }  // exact-dispatch guard
    let tile_size = 8;
    let m0 = i32(gid3.x) * tile_size;
    let n0 = i32(gid3.y) * tile_size;
    let tile = i32(gid3.y) * dc_p.n_tiles_x + i32(gid3.x);
    var cw = array<f32, 16>(0.00, 0.02, 0.04, 0.08, 0.04, 0.08, 0.04, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    let mismatch = dc_mismatch[tile];
    let mismatch_weight = clamp(1.0 - 10.0 * (mismatch - 0.2), 0.0, 1.0);

    let convRe0 = dc_ft[n0 * dc_p.ft_w + 2 * m0];
    let convIm0 = dc_ft[n0 * dc_p.ft_w + 2 * m0 + 1];
    let convMag0 = sqrt(convRe0 * convRe0 + convIm0 * convIm0);
    let magnitude_zero = convMag0[0] + convMag0[1] + convMag0[2] + convMag0[3];

    for (var dn = 0; dn < tile_size; dn++) {
        for (var dm = 0; dm < tile_size; dm++) {
            if (dm + dn > 0 && mismatch < 0.3) {
                let m = 2 * (m0 + dm);
                let n = n0 + dn;
                let convRe = dc_ft[n * dc_p.ft_w + m];
                let convIm = dc_ft[n * dc_p.ft_w + m + 1];
                let convMag = sqrt(convRe * convRe + convIm * convIm);
                let magnitude = convMag[0] + convMag[1] + convMag[2] + convMag[3];
                // weight 0 for ratio >= 0.05, 1 for ratio <= 0.01
                let weight = mismatch_weight * clamp(1.25 - 25.0 * magnitude / magnitude_zero, 0.0, 1.0);
                let gain = (1.0 + weight * cw[dm]) * (1.0 + weight * cw[dn]);
                let idx = n * dc_p.ft_w + m;
                dc_ft[idx] = gain * convRe;
                dc_ft[idx + 1] = gain * convIm;
            }
        }
    }
}

// ============================================ reduce_artifacts_tile_border
// Port of frequency.metal:reduce_artifacts_tile_border — clamp negative (sub-black)
// values and blend tile-border pixels with the reference via the raised cosine window.
// Params: rgba_w, n_tiles_x, n_tiles_y, black0..3 (reference black levels)
struct RabParams { rgba_w: i32, n_tiles_x: i32, n_tiles_y: i32, black0: f32, black1: f32, black2: f32, black3: f32, };
@group(0) @binding(0) var<storage, read_write> rab_out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       rab_ref: array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             rab_p:   RabParams;

@compute @workgroup_size(8, 8)
fn reduce_artifacts_tile_border(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (i32(gid3.x) >= rab_p.n_tiles_x || i32(gid3.y) >= rab_p.n_tiles_y) { return; }  // exact-dispatch guard
    let tile_size = 8;
    let x0 = i32(gid3.x) * tile_size;
    let y0 = i32(gid3.y) * tile_size;
    let min_values = vec4<f32>(rab_p.black0 - 1.0, rab_p.black1 - 1.0, rab_p.black2 - 1.0, rab_p.black3 - 1.0);
    let max_values = vec4<f32>(UINT16_MAX_VAL);
    let angle = -2.0 * PI / f32(tile_size);

    for (var dy = 0; dy < tile_size; dy++) {
        for (var dx = 0; dx < tile_size; dx++) {
            let x = x0 + dx;
            let y = y0 + dy;
            let norm_cosine = (0.5 - 0.5 * cos(-angle * (f32(dx) + 0.5))) * (0.5 - 0.5 * cos(-angle * (f32(dy) + 0.5)));
            let idx = y * rab_p.rgba_w + x;
            var pixel_value = rab_out[idx];
            pixel_value = clamp(pixel_value, norm_cosine * min_values, max_values);
            if (dx == 0 || dx == tile_size - 1 || dy == 0 || dy == tile_size - 1) {
                pixel_value = 0.5 * (norm_cosine * rab_ref[idx] + pixel_value);
            }
            rab_out[idx] = pixel_value;
        }
    }
}

`;
