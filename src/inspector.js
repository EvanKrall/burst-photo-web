// Alignment inspector UI: interactive pre-merge visualization of the alignment fields and
// difference images of a burst (see src/inspect.js for the underlying InspectionSession).
// Port of the Swift visualize branch's AlignmentInspectorView.swift, restructured for the DOM:
//
// - The "stage" pane switches to this view after decoding when "Inspect alignment first" is on;
//   the Merge button runs the full pipeline on the frames still checked in the frame list.
// - All heavy rendering (warp, difference, tonemap) happens on demand for the selected frame
//   only, asynchronously on the GPU; requests are queued and coalesced, and stale requests are
//   dropped when control parameters change (scheduleRender/pump, port of the Swift render queue).
// - The alignment color scale and the difference white point are SHARED ACROSS ALL FRAMES of a
//   burst (median-based, computed in the session's calibration pass), so images are comparable
//   between frames. Until calibration completes, difference renders fall back to per-frame scale.
// - Clicking the image sets an ANCHOR POINT: all alignment/flow visualizations (main view and
//   thumbnails) subtract the shift of the anchored tile of their own frame, so the anchor reads
//   zero and the field shows the deformation RELATIVE to it. With an anchor set, the Difference
//   mode fits the brightness match locally and robustly around it instead of globally.
// - Dragging on the image marks the SUBJECT AREA (region of interest). Both scores in the frame
//   list — mean brightness-matched difference and alignment-field variation — are computed inside
//   it; the list can be sorted by either. The default subject area is the centered half of the
//   frame (the Swift app additionally reads the reference frame's AF point from Nikon maker
//   notes, which flashraw does not expose).
// - The frame list supports checking/unchecking images (excluded from the merge), and multi-
//   selection with cmd-click (toggle one) and shift-click (toggle a range).
// - Arrow keys scrub through frames IN LOCKSTEP with the preview: while a render is in flight,
//   repeated steps coalesce into one pending move that is applied when the render finishes.

import { renderAlignmentField, grayBitmapFromValues, roiIsDegenerate, normalizeRoi } from './viz.js';

const MODES = ['alignment', 'flow', 'difference', 'original', 'warped', 'reference'];
const MODE_IS_ALIGNMENT = (m) => m === 'alignment' || m === 'flow';
const MODE_KEEP_THUMBS = (m) => m === 'original' || m === 'reference'; // thumbnails keep previous mode

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

export class InspectorUI {
  /**
   * @param {InspectionSession} session from performInspection()
   * @param {{ onMerge: (includedFrames: Array) => void }} handlers
   */
  constructor(session, { onMerge }) {
    this.session = session;
    this.onMerge = onMerge;

    const $ = (id) => document.getElementById(id);
    this.el = {
      pane: $('inspectorPane'),
      modes: $('inspModes'),
      alignControls: $('inspAlignControls'),
      alignShared: $('inspAlignShared'),
      alignScaleLabel: $('inspAlignScaleLabel'),
      alignScale: $('inspAlignScale'),
      diffControls: $('inspDiffControls'),
      matchExposure: $('inspMatchExposure'),
      diffShared: $('inspDiffShared'),
      diffGainLabel: $('inspDiffGainLabel'),
      diffGain: $('inspDiffGain'),
      clearAnchor: $('inspClearAnchor'),
      sort: $('inspSort'),
      roiHint: $('inspRoiHint'),
      clearRoi: $('inspClearRoi'),
      frameList: $('inspFrameList'),
      imageWrap: $('inspImageWrap'),
      holder: $('inspCanvasHolder'),
      canvas: $('inspCanvas'),
      overlay: $('inspOverlay'),
      zoomBadge: $('inspZoomBadge'),
      spinner: $('inspSpinner'),
      statsLine: $('inspStatsLine'),
      infoLine: $('inspInfoLine'),
      mergeText: $('inspMergeText'),
      mergeBtn: $('inspMerge'),
    };

    // ---- UI state (port of InspectorContentView's @State vars) ----
    this.state = {
      selectedIdx: session.comparisonFrames()[0]?.idx ?? session.refIdx,
      selection: new Set(),
      lastClicked: 0,
      excluded: new Set(),
      mode: 'alignment',
      alignShared: true,
      alignManualScale: 32,
      diffShared: true,
      diffGain: 4.0,
      matchExposure: true,
      anchor: null, // {u, v, tileX, tileY} | null
      anchorManual: false,
      roi: { x0: 0.25, y0: 0.25, x1: 0.75, y1: 0.75 }, // default subject area: centered half
      sortMode: 0,
      zoom: 1,
      renderedInfo: '',
    };
    const first = this.state.selectedIdx;
    this.state.selection = new Set([first]);
    this.state.lastClicked = first;
    // anchor starts at the subject-area midpoint (still automatic; a manual click or
    // "Clear anchor" hands control to the user)
    this.state.anchor = this.sceneAnchor((this.state.roi.x0 + this.state.roi.x1) / 2, (this.state.roi.y0 + this.state.roi.y1) / 2);

    // render queue (port of the Swift render_queue/pump_renders lockstep machinery)
    this.renderQueue = [];
    this.renderBusy = false;
    this.pendingMoves = 0;
    this.lastQueuedKey = null;
    this.lastThumbKey = null;

    this.rows = new Map(); // idx -> row element
    this.thumbs = new Map(); // idx -> ImageBitmap-ish {rgba, width, height}

    this.buildFrameList();
    this.bindEvents();
    this.refreshToolbar();
    this.refreshScores();
    this.updateMergeBar();
    this.scheduleRender();

    // background calibration pass (thumbnails, brightness matches, shared scales)
    this.startCalibration();
  }

  // ------------------------------------------------------------- session helpers

  sceneAnchor(u, v) {
    const ti = this.session.tileInfo;
    const downscale = ti.downscale;
    const halfTile = ti.tileSize / 2;
    const gx = ((u * this.session.width + this.session.padX) / downscale + 0.5) / halfTile - 1;
    const gy = ((v * this.session.height + this.session.padY) / downscale + 0.5) / halfTile - 1;
    return { u, v, tileX: clamp(Math.round(gx), 0, ti.w - 1), tileY: clamp(Math.round(gy), 0, ti.h - 1) };
  }

  /** Frame indices in current display order (capture order, or score-sorted, ref pinned top). */
  displayIndices() {
    const scores = this.state.sortMode === 2 ? this.alignScores : this.diffScores;
    if (this.state.sortMode === 0 || !scores || !scores.size) {
      return this.session.frameList.map((f) => f.idx);
    }
    const rank = (f) => (f.isRef ? 0 : scores.has(f.idx) ? 1 : 2);
    return [...this.session.frameList]
      .sort((a, b) => {
        const ra = rank(a), rb = rank(b);
        if (ra !== rb) return ra - rb;
        if (ra === 1 && scores.get(a.idx) !== scores.get(b.idx)) return scores.get(a.idx) - scores.get(b.idx);
        return a.idx - b.idx;
      })
      .map((f) => f.idx);
  }

  includedFrames() {
    return this.session.frameList.filter((f) => !this.state.excluded.has(f.idx));
  }

  // ---------------------------------------------------------------- frame list

  buildFrameList() {
    const list = this.el.frameList;
    list.textContent = '';
    this.rows.clear();
    for (const f of this.session.frameList) {
      const row = document.createElement('div');
      row.className = 'insp-row';
      row.dataset.idx = f.idx;

      const check = document.createElement('div');
      check.className = 'insp-check';
      check.title = 'Included in the merge (click to exclude)';

      const thumb = document.createElement('canvas');
      thumb.className = 'insp-thumb';
      thumb.width = 44;
      thumb.height = 30;

      const meta = document.createElement('div');
      meta.className = 'insp-meta';
      const name = document.createElement('div');
      name.className = 'insp-name';
      name.textContent = `frame ${String(f.idx).padStart(2, '0')}`;
      if (f.isRef) {
        const badge = document.createElement('span');
        badge.className = 'insp-refbadge';
        badge.textContent = 'REF';
        name.appendChild(document.createTextNode(' '));
        name.appendChild(badge);
      }
      const file = document.createElement('div');
      file.className = 'insp-file';
      file.textContent = f.name;
      file.title = f.name;
      meta.append(name, file);

      const diffScore = document.createElement('div');
      diffScore.className = 'insp-score none';
      diffScore.title = 'Mean difference to the reference inside the subject area (lower = more similar)';
      const alignScore = document.createElement('div');
      alignScore.className = 'insp-score none';
      alignScore.title = 'Variation of the alignment field inside the subject area, in level-0 pixels (lower = cleaner mapping)';

      row.append(check, thumb, meta, diffScore, alignScore);
      list.appendChild(row);
      this.rows.set(f.idx, { row, check, thumb, diffScore, alignScore });
    }
    this.refreshSelection();
    this.refreshIncluded();
  }

  refreshSelection() {
    for (const [idx, row] of this.rows) {
      row.row.classList.toggle('selected', this.state.selection.has(idx));
    }
    const el = this.rows.get(this.state.selectedIdx);
    el?.row.scrollIntoView({ block: 'nearest' });
  }

  refreshIncluded() {
    for (const [idx, row] of this.rows) {
      row.check.classList.toggle('included', !this.state.excluded.has(idx));
    }
    this.updateMergeBar();
  }

  refreshScores() {
    this.diffScores = new Map();
    this.alignScores = new Map();
    for (const f of this.session.comparisonFrames()) {
      if (this.state.roi) {
        const d = this.session.roiDiffScore(f.idx, this.state.roi);
        if (d != null) this.diffScores.set(f.idx, d);
        const a = this.session.alignmentVariation(f.idx, this.state.roi);
        if (a != null) this.alignScores.set(f.idx, a);
      }
    }
    for (const [idx, row] of this.rows) {
      const d = this.diffScores.get(idx);
      const a = this.alignScores.get(idx);
      row.diffScore.textContent = d != null ? d.toFixed(1) : '–';
      row.diffScore.classList.toggle('none', d == null);
      row.alignScore.textContent = a != null ? a.toFixed(2) : '–';
      row.alignScore.classList.toggle('none', a == null);
    }
    this.rebuildOrder();
  }

  rebuildOrder() {
    const order = this.displayIndices();
    for (const idx of order) {
      const row = this.rows.get(idx);
      this.el.frameList.appendChild(row.row);
    }
  }

  updateMergeBar() {
    const included = this.includedFrames().length;
    const total = this.session.frameList.length;
    this.el.mergeBtn.textContent = `Merge ${included} of ${total} images`;
    this.el.mergeBtn.disabled = included < 2;
    this.el.mergeText.textContent = included === total
      ? 'Merging runs the full alignment + merge pipeline on the checked images.'
      : 'Unchecked images are excluded from the merge (it picks its own reference from the included images).';
  }

  // ------------------------------------------------------------------ events

  bindEvents() {
    const el = this.el;

    // toolbar: mode + per-mode controls
    el.modes.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-mode]');
      if (btn) this.setMode(btn.dataset.mode);
    });
    el.alignShared.addEventListener('change', () => {
      this.state.alignShared = el.alignShared.checked;
      this.refreshToolbar();
      this.scheduleRender();
    });
    el.alignScale.addEventListener('input', () => {
      this.state.alignManualScale = Number(el.alignScale.value);
      this.refreshToolbar();
      this.scheduleRender();
    });
    el.matchExposure.addEventListener('change', () => {
      this.state.matchExposure = el.matchExposure.checked;
      el.diffShared.disabled = !this.state.matchExposure;
      this.scheduleRender();
    });
    el.diffShared.addEventListener('change', () => {
      this.state.diffShared = el.diffShared.checked;
      this.scheduleRender();
    });
    el.diffGain.addEventListener('input', () => {
      this.state.diffGain = Number(el.diffGain.value);
      el.diffGainLabel.textContent = `gain ${this.state.diffGain.toFixed(1)}`;
      this.scheduleRender();
    });
    el.clearAnchor.addEventListener('click', () => {
      this.state.anchor = null;
      this.state.anchorManual = true; // manual removal: ROI changes no longer re-place the anchor
      this.refreshToolbar();
      this.scheduleRender();
    });
    el.sort.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-sort]');
      if (!btn) return;
      this.state.sortMode = Number(btn.dataset.sort);
      for (const b of el.sort.querySelectorAll('button')) b.classList.toggle('active', b === btn);
      this.rebuildOrder();
    });
    el.clearRoi.addEventListener('click', () => this.setRoi(null));

    // frame list rows: click = select, cmd-click = toggle in selection, shift-click = range,
    // checkbox = include/exclude (multi-selection follows)
    el.frameList.addEventListener('click', (e) => {
      const rowEl = e.target.closest('.insp-row');
      if (!rowEl) return;
      const idx = Number(rowEl.dataset.idx);
      if (e.target.closest('.insp-check')) {
        this.toggleIncluded(idx);
        return;
      }
      this.rowClicked(idx, e.metaKey || e.ctrlKey, e.shiftKey);
    });

    // image area: drag = mark subject area, plain click = anchor, double-click = reset zoom
    let drag = null; // {startX, startY, isRoi, liveRect, clickCount}
    let lastClickTime = 0;
    const overlay = el.overlay;
    overlay.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      overlay.setPointerCapture(e.pointerId);
      const rect = this.fittedRect();
      const pos = this.overlayPos(e);
      drag = { startX: pos.x, startY: pos.y, isRoi: false, liveRect: null, inside: rect.contains(pos.x, pos.y) };
      e.preventDefault();
    });
    overlay.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const rect = this.fittedRect();
      const pos = this.overlayPos(e);
      if (!drag.isRoi) {
        if (Math.hypot(pos.x - drag.startX, pos.y - drag.startY) < 4) return;
        drag.isRoi = true;
      }
      drag.liveRect = this.clampToRect(rect, drag.startX, drag.startY, pos.x, pos.y);
      this.drawOverlay(drag.liveRect);
    });
    overlay.addEventListener('pointerup', (e) => {
      if (!drag) return;
      const wasDrag = drag.isRoi;
      const live = drag.liveRect;
      const inside = drag.inside;
      drag = null;
      this.drawOverlay();
      if (wasDrag) {
        // finished a subject-area drag; ignore degenerate rects (>4 px but still tiny)
        if (live && !roiIsDegenerate(live)) this.setRoi(live);
        return;
      }
      if (!inside) return;
      const now = performance.now();
      const isDoubleClick = now - lastClickTime < 400;
      lastClickTime = now;
      if (isDoubleClick) {
        this.setZoom(1);
        return;
      }
      const rect = this.fittedRect();
      const pos = this.overlayPos(e);
      this.setAnchor((pos.x - rect.x) / rect.w, (pos.y - rect.y) / rect.h);
    });
    overlay.addEventListener('pointercancel', () => { drag = null; this.drawOverlay(); });

    // wheel: zoom centered on the cursor (ctrl+wheel = trackpad pinch; browser zoom is blocked)
    el.imageWrap.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * 0.0015);
      this.setZoom(clamp(this.state.zoom * factor, 1, 16), e);
    }, { passive: false });

    // merge
    el.mergeBtn.addEventListener('click', () => {
      const included = this.includedFrames();
      if (included.length < 2) return;
      this.onMerge(included.map((f) => f.idx));
    });

    // keyboard navigation (only while the inspector is visible and no text input has focus)
    this.keyHandler = (e) => {
      if (this.el.pane.hidden) return;
      const tag = document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === 'Escape') {
        if (this.state.roi) { this.setRoi(null); e.preventDefault(); }
        return;
      }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        this.requestSelectionMove(e.key === 'ArrowDown' ? 1 : -1);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        this.moveMode(e.key === 'ArrowRight' ? 1 : -1);
      } else if (e.key === ' ') {
        e.preventDefault();
        this.toggleIncluded(this.state.selectedIdx);
      }
    };
    window.addEventListener('keydown', this.keyHandler);
  }

  destroy() {
    window.removeEventListener('keydown', this.keyHandler);
    this.aborted = true;
  }

  // ------------------------------------------------------------- state mutations

  setMode(mode) {
    if (!MODES.includes(mode) || mode === this.state.mode) return;
    this.state.mode = mode;
    this.refreshToolbar();
    this.scheduleRender();
  }

  moveMode(delta) {
    const next = clamp(MODES.indexOf(this.state.mode) + delta, 0, MODES.length - 1);
    this.setMode(MODES[next]);
  }

  setRoi(roi) {
    this.state.roi = roi;
    this.el.clearRoi.hidden = !roi;
    this.el.roiHint.hidden = !!roi;
    // while the user has not taken manual control of the anchor, it follows the subject area
    if (!this.state.anchorManual) {
      this.state.anchor = roi ? this.sceneAnchor((roi.x0 + roi.x1) / 2, (roi.y0 + roi.y1) / 2) : null;
      this.refreshToolbar();
    }
    this.refreshScores();
    this.drawOverlay();
    this.scheduleRender();
  }

  setAnchor(u, v) {
    const uC = clamp(u, 0, 1), vC = clamp(v, 0, 1);
    const ti = this.session.tileInfo;
    let tileX, tileY;
    if (MODE_IS_ALIGNMENT(this.state.mode)) {
      // click on the tile-grid visualization: one bitmap pixel = one tile
      tileX = clamp(Math.round(uC * ti.w - 0.5), 0, ti.w - 1);
      tileY = clamp(Math.round(vC * ti.h - 0.5), 0, ti.h - 1);
    } else {
      const scene = this.sceneAnchor(uC, vC);
      tileX = scene.tileX;
      tileY = scene.tileY;
    }
    this.state.anchor = { u: uC, v: vC, tileX, tileY };
    this.state.anchorManual = true;
    this.refreshToolbar();
    this.scheduleRender();
  }

  rowClicked(idx, cmd, shift) {
    const selection = this.state.selection;
    if (cmd) {
      if (selection.has(idx)) selection.delete(idx);
      else selection.add(idx);
    } else if (shift) {
      const lower = Math.min(this.state.lastClicked, idx);
      const upper = Math.max(this.state.lastClicked, idx);
      const range = new Set();
      for (let i = lower; i <= upper; i++) range.add(i);
      if (selection.has(idx)) for (const i of range) selection.delete(i);
      else for (const i of range) selection.add(i);
    } else {
      selection.clear();
      selection.add(idx);
    }
    this.state.selectedIdx = idx;
    this.state.lastClicked = idx;
    this.refreshSelection();
    this.scheduleRender();
  }

  toggleIncluded(idx) {
    const makeExcluded = !this.state.excluded.has(idx);
    if (this.state.selection.has(idx) && this.state.selection.size > 1) {
      // the row belongs to a multi-selection: all selected rows follow
      for (const i of this.state.selection) {
        if (makeExcluded) this.state.excluded.add(i);
        else this.state.excluded.delete(i);
      }
    } else if (makeExcluded) {
      this.state.excluded.add(idx);
    } else {
      this.state.excluded.delete(idx);
    }
    this.refreshIncluded();
  }

  requestSelectionMove(delta) {
    // coalesce into the pending move while a render is in flight — applied when it finishes,
    // so the selected row and the preview advance in lockstep at render speed
    if (this.renderBusy) {
      this.pendingMoves = delta;
      return;
    }
    this.applySelectionMove(delta);
  }

  applySelectionMove(delta) {
    const order = this.displayIndices();
    const pos = order.indexOf(this.state.selectedIdx);
    if (pos < 0) return false;
    const next = order[clamp(pos + delta, 0, order.length - 1)];
    if (next === this.state.selectedIdx) return false;
    this.state.selectedIdx = next;
    this.state.selection = new Set([next]);
    this.state.lastClicked = next;
    this.refreshSelection();
    this.scheduleRender();
    return true;
  }

  // ------------------------------------------------------------------ toolbar

  refreshToolbar() {
    const el = this.el;
    const mode = this.state.mode;
    for (const b of el.modes.querySelectorAll('button')) b.classList.toggle('active', b.dataset.mode === mode);
    el.alignControls.hidden = !MODE_IS_ALIGNMENT(mode);
    el.diffControls.hidden = mode !== 'difference';
    el.clearAnchor.hidden = !this.state.anchor;
    el.alignScaleLabel.hidden = this.state.alignShared;
    el.alignScale.hidden = this.state.alignShared;
    el.alignScaleLabel.textContent = `±${Math.round(this.state.alignManualScale)} px`;
  }

  // ---------------------------------------------------------------- rendering

  /** Display rect of the fitted image inside the scroll container (CSS px). */
  fittedRect() {
    const bmp = this.currentBitmap;
    if (!bmp) return { x: 0, y: 0, w: 0, h: 0 };
    const w = bmp.width * this.displayScale();
    const h = bmp.height * this.displayScale();
    return { x: 0, y: 0, w, h };
  }

  /** CSS px per bitmap px: the canvas element is sized bitmap*fit*zoom, so this is fit*zoom. */
  displayScale() {
    return this.baseScale * this.state.zoom;
  }

  overlayPos(e) {
    const r = this.el.overlay.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  clampToRect(rect, x0, y0, x1, y1) {
    const ax = clamp(Math.min(x0, x1), rect.x, rect.x + rect.w);
    const ay = clamp(Math.min(y0, y1), rect.y, rect.y + rect.h);
    const bx = clamp(Math.max(x0, x1), rect.x, rect.x + rect.w);
    const by = clamp(Math.max(y0, y1), rect.y, rect.y + rect.h);
    return normalizeRoi({ x0: (ax - rect.x) / rect.w, y0: (ay - rect.y) / rect.h, x1: (bx - rect.x) / rect.w, y1: (by - rect.y) / rect.h });
  }

  setZoom(zoom, anchorEvent = null) {
    const wrap = this.el.imageWrap;
    const prevScale = this.displayScale();
    // content point under the cursor (or center) stays put
    const cx = anchorEvent ? this.overlayPos(anchorEvent).x + wrap.scrollLeft : wrap.clientWidth / 2 + wrap.scrollLeft;
    const cy = anchorEvent ? this.overlayPos(anchorEvent).y + wrap.scrollTop : wrap.clientHeight / 2 + wrap.scrollTop;
    this.state.zoom = zoom;
    this.applyCanvasSize();
    const scale = this.displayScale();
    wrap.scrollLeft = cx * (scale / prevScale) - (anchorEvent ? this.overlayPos(anchorEvent).x : wrap.clientWidth / 2);
    wrap.scrollTop = cy * (scale / prevScale) - (anchorEvent ? this.overlayPos(anchorEvent).y : wrap.clientHeight / 2);
    this.el.zoomBadge.hidden = false;
    this.el.zoomBadge.textContent = `×${zoom.toFixed(1)}  (double-click to reset)`;
    this.drawOverlay();
  }

  applyCanvasSize() {
    const bmp = this.currentBitmap;
    const holder = this.el.holder;
    if (!bmp) {
      holder.style.width = '100%';
      holder.style.height = '100%';
      holder.style.position = 'static';
      return;
    }
    holder.style.position = 'relative';
    const w = Math.round(bmp.width * this.displayScale());
    const h = Math.round(bmp.height * this.displayScale());
    holder.style.width = `${w}px`;
    holder.style.height = `${h}px`;
    this.el.canvas.style.width = `${w}px`;
    this.el.canvas.style.height = `${h}px`;
    this.el.overlay.style.width = `${w}px`;
    this.el.overlay.style.height = `${h}px`;
    this.el.overlay.width = w;
    this.el.overlay.height = h;
  }

  drawOverlay(liveRect = null) {
    const overlay = this.el.overlay;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const rect = this.fittedRect();
    if (!rect.w) return;

    // committed subject area (darken everything outside it)
    const roi = this.state.roi;
    if (roi && !liveRect) {
      const x0 = rect.x + roi.x0 * rect.w, y0 = rect.y + roi.y0 * rect.h;
      const x1 = rect.x + roi.x1 * rect.w, y1 = rect.y + roi.y1 * rect.h;
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(0, 0, overlay.width, y0);
      ctx.fillRect(0, y1, overlay.width, overlay.height - y1);
      ctx.fillRect(0, y0, x0, y1 - y0);
      ctx.fillRect(x1, y0, overlay.width - x1, y1 - y0);
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
    }
    // in-progress drag rectangle (solid, slightly brighter)
    if (liveRect) {
      const x0 = rect.x + liveRect.x0 * rect.w, y0 = rect.y + liveRect.y0 * rect.h;
      const x1 = rect.x + liveRect.x1 * rect.w, y1 = rect.y + liveRect.y1 * rect.h;
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(0, 0, overlay.width, y0);
      ctx.fillRect(0, y1, overlay.width, overlay.height - y1);
      ctx.fillRect(0, y0, x0, y1 - y0);
      ctx.fillRect(x1, y0, overlay.width - x1, y1 - y0);
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }

    // circular anchor marker
    const anchor = this.state.anchor;
    if (anchor) {
      const cx = rect.x + anchor.u * rect.w;
      const cy = rect.y + anchor.v * rect.h;
      const r = 6.5;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(255,255,255,0.95)';
      ctx.stroke();
    }
  }

  // -------------------------------------------------------------- render queue

  baseScaleFor(bmp) {
    const wrap = this.el.imageWrap;
    const availW = Math.max(wrap.clientWidth - 24, 100);
    const availH = Math.max(wrap.clientHeight - 24, 100);
    // no upper cap: alignment-mode bitmaps are tile-grid-sized (a few px) and must fill the
    // pane like the Swift inspector's fitted view (upscaled nearest-neighbor)
    return Math.min(availW / bmp.width, availH / bmp.height);
  }

  renderTargetSize() {
    // pooled GPU render size for the photo modes: roughly pane-sized, aspect-preserved
    const wrap = this.el.imageWrap;
    const aspect = this.session.width / this.session.height;
    let w = Math.min(wrap.clientWidth - 24, 1440);
    let h = Math.round(w / aspect);
    if (h > wrap.clientHeight - 24) {
      h = Math.max(wrap.clientHeight - 24, 100);
      w = Math.round(h * aspect);
    }
    return { outWidth: clamp(Math.round(w / 4) * 4, 64, 4096), outHeight: clamp(Math.round(h / 4) * 4, 64, 4096) };
  }

  currentRenderParams() {
    const s = this.state;
    const target = this.renderTargetSize();
    return {
      mode: s.mode,
      alignmentScale: s.alignShared ? this.session.alignmentScale : Math.max(Math.round(s.alignManualScale), 1),
      anchor: s.anchor,
      diffWhiteOverride: (s.mode === 'difference' && s.diffShared && s.matchExposure && this.session.calibrated)
        ? s.diffGain * this.session.medianDiffMean : null,
      diffGain: s.diffGain,
      matchExposure: s.matchExposure,
      ...target,
    };
  }

  scheduleRender() {
    this.updateThumbnails();
    const params = this.currentRenderParams();
    const key = JSON.stringify(params);
    // a parameter change invalidates queued requests so control changes respond instantly;
    // repeated identical requests coalesce
    if (key !== this.lastQueuedKey) {
      this.renderQueue.length = 0;
      this.lastQueuedKey = key;
    }
    const req = { idx: this.state.selectedIdx, params, key };
    const last = this.renderQueue[this.renderQueue.length - 1];
    if (!last || last.idx !== req.idx || last.key !== req.key) {
      this.renderQueue.push(req);
      if (this.renderQueue.length > 64) this.renderQueue.shift();
    }
    this.pumpRenders();
    this.refreshStatsBar();
  }

  async pumpRenders() {
    if (this.renderBusy) return;
    const req = this.renderQueue.shift();
    if (!req) return;
    this.renderBusy = true;
    this.el.spinner.hidden = false;
    try {
      const s = this.state;
      // per-frame anchor offset is computed at render time (level-0 px)
      let offsetX = 0, offsetY = 0;
      if (MODE_IS_ALIGNMENT(req.params.mode) && req.params.anchor) {
        [offsetX, offsetY] = this.session.texelOffset(req.idx, req.params.anchor.tileX, req.params.anchor.tileY);
      }
      const { bitmap, info } = await this.session.renderFrame(req.idx, { ...req.params, offsetX, offsetY });
      this.currentBitmap = bitmap;
      this.state.renderedInfo = info;
      const ctx = this.el.canvas.getContext('2d');
      this.el.canvas.width = bitmap.width;
      this.el.canvas.height = bitmap.height;
      ctx.putImageData(new ImageData(bitmap.rgba, bitmap.width, bitmap.height), 0, 0);
      this.el.holder.classList.toggle('pixelated', MODE_IS_ALIGNMENT(req.params.mode));
      if (this.state.zoom === 1) {
        this.baseScale = this.baseScaleFor(bitmap);
        this.applyCanvasSize();
        this.el.zoomBadge.hidden = true;
      } else {
        this.applyCanvasSize();
      }
      this.drawOverlay();
      this.refreshStatsBar();
    } catch (err) {
      console.error('inspector render failed:', err);
      this.state.renderedInfo = `render failed: ${err.message}`;
      this.refreshStatsBar();
    } finally {
      this.renderBusy = false;
      this.el.spinner.hidden = true;
      let more = this.renderQueue.length > 0;
      if (this.pendingMoves !== 0) {
        const delta = this.pendingMoves;
        this.pendingMoves = 0;
        // the selection change schedules the next scrub render; hitting the end is a no-op
        more = this.applySelectionMove(delta) || more;
      }
      if (more) this.pumpRenders();
    }
  }

  // -------------------------------------------------------------- thumbnails

  /** Regenerate the frame-list thumbnails if the parameters they depend on changed. */
  updateThumbnails() {
    if (MODE_KEEP_THUMBS(this.state.mode)) return;
    const s = this.state;
    const scale = s.alignShared ? this.session.alignmentScale : Math.max(Math.round(s.alignManualScale), 1);
    // calibration progress is part of the key so difference/warped thumbnails are regenerated
    // periodically while the calibration pass fills in the per-frame data
    const key = [
      s.mode, scale,
      s.anchor ? `${s.anchor.tileX},${s.anchor.tileY}` : 'none',
      s.mode === 'difference' ? s.diffGain : 0,
      this.session.calibrated,
      (this.session.calibrationProgress / 10) | 0,
    ].join('|');
    if (key === this.lastThumbKey) return;
    this.lastThumbKey = key;

    for (const f of this.session.frameList) {
      const thumb = this.rows.get(f.idx)?.thumb;
      if (!thumb) continue;
      const ctx = thumb.getContext('2d');
      let bitmap = null;
      if (MODE_IS_ALIGNMENT(s.mode)) {
        const texels = f.isRef ? new Int32Array(this.session.tileInfo.w * this.session.tileInfo.h * 2) : f.texels;
        let ox = 0, oy = 0;
        if (s.anchor && !f.isRef) [ox, oy] = this.session.texelOffset(f.idx, s.anchor.tileX, s.anchor.tileY);
        bitmap = renderAlignmentField(texels, this.session.tileInfo.w, this.session.tileInfo.h, {
          mode: s.mode === 'flow' ? 'flow' : 'rgb', scale, offsetX: ox, offsetY: oy,
        });
      } else if (s.mode === 'difference') {
        const data = this.session.frameData.get(f.idx);
        if (f.isRef) {
          const d = this.session.refThumbnailData;
          bitmap = d && grayBitmapFromValues(new Float32Array(d.values.length), d.width, d.height, 1);
        } else if (data) {
          const white = (this.session.calibrated ? this.session.medianDiffMean : data.diffMean) * s.diffGain;
          bitmap = grayBitmapFromValues(data.diff48.values, data.diff48.width, data.diff48.height, Math.max(white, 1e-6));
        }
      } else if (s.mode === 'warped') {
        if (f.isRef) {
          const d = this.session.refThumbnailData;
          bitmap = d && grayBitmapFromValues(d.values, d.width, d.height, Math.max(d.max, 1e-6));
        } else {
          const data = this.session.frameData.get(f.idx);
          if (data) bitmap = grayBitmapFromValues(data.warped48.values, data.warped48.width, data.warped48.height, Math.max(data.warpedMax, 1e-6));
        }
      }
      if (!bitmap) continue; // keep the previous mode's thumbnail until data exists
      const tmp = new OffscreenCanvas(bitmap.width, bitmap.height);
      tmp.getContext('2d').putImageData(new ImageData(bitmap.rgba, bitmap.width, bitmap.height), 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, thumb.width, thumb.height);
      // preserve aspect inside the 44x30 box
      const scaleFit = Math.min(thumb.width / bitmap.width, thumb.height / bitmap.height);
      const w = bitmap.width * scaleFit, h = bitmap.height * scaleFit;
      ctx.drawImage(tmp, (thumb.width - w) / 2, (thumb.height - h) / 2, w, h);
    }
  }

  // -------------------------------------------------------------- calibration

  async startCalibration() {
    this.calibrating = true;
    try {
      await this.session.calibrate(({ frame, total }) => {
        if (this.aborted) return true; // cancel the pass (session is being disposed)
        if (frame % 5 === 0 || frame === total) {
          this.refreshScores(); // diff scores appear frame by frame
          this.el.infoLine.textContent = `measuring shared scale… ${frame}/${total}`;
          this.scheduleRender();
        }
        return false;
      });
    } catch (err) {
      console.error('calibration failed:', err);
    } finally {
      this.calibrating = false;
    }
    if (this.aborted) return;
    // calibration complete: shared scales are now available — refresh everything
    this.lastThumbKey = null;
    this.refreshScores();
    this.scheduleRender();
  }

  // -------------------------------------------------------------- stats bar

  refreshStatsBar() {
    const s = this.state;
    const f = this.session.frame(s.selectedIdx);
    let stats;
    if (f.isRef) {
      stats = `frame ${f.idx} — ${f.name} — reference frame`;
    } else if (f.stats) {
      const st = f.stats;
      stats = `frame ${f.idx} — ${f.name} · mean |shift| ${st.meanMagnitude.toFixed(1)} l0-px (${(st.meanMagnitude * this.session.tileInfo.downscale).toFixed(1)} sensor px) · max ${st.maxAbsShift} l0-px`;
    } else {
      stats = `frame ${f.idx} — ${f.name}`;
    }
    let anchorText = '';
    if (s.anchor) {
      const [dx, dy] = this.session.texelOffset(s.selectedIdx, s.anchor.tileX, s.anchor.tileY);
      const factor = this.session.tileInfo.downscale;
      anchorText = ` · anchor tile (${s.anchor.tileX}, ${s.anchor.tileY}): shift (${dx}, ${dy}) l0-px = (${(dx * factor).toFixed(0)}, ${(dy * factor).toFixed(0)}) sensor px`;
    }
    const calibrating = s.mode === 'difference' && s.diffShared && s.matchExposure && !this.session.calibrated;
    const calibText = calibrating ? ` · measuring shared scale… ${this.session.calibrationProgress}/${this.session.comparisonFrames().length}` : '';
    this.el.statsLine.textContent = stats + calibText;
    this.el.infoLine.textContent = [s.renderedInfo, anchorText].filter(Boolean).join('')
      + '  — ↑/↓ frame · ←/→ mode · space = check · click = anchor · wheel = zoom, double-click reset';
  }
}
