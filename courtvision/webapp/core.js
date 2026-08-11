/* CourtVision core — the full line-calling pipeline in plain JavaScript.
 *
 * A faithful port of the Python engine (court geometry, homography, Kalman
 * ball tracking, bounce-kink detection, ITF line rules, match stats) so the
 * whole system runs on a phone, in the browser, with no server at all.
 * Verified in Node against fixtures exported from the Python pipeline: same
 * candidates in -> same calls out.
 *
 * Everything below works in pixel units of the *processing* frame; the
 * `scale` config adapts speed/size thresholds to the processing resolution
 * (they were tuned at 540p).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.CourtVisionCore = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------------------------------------------------------------- court
  const COURT = {
    DOUBLES_WIDTH: 10.97,
    SINGLES_WIDTH: 8.23,
    LENGTH: 23.77,
    NET_Y: 11.885,
    ALLEY: 1.37,
    CENTER_X: 5.485,
    SERVICE_NEAR_Y: 5.485,
    SERVICE_FAR_Y: 18.285,
    BALL_RADIUS: 0.033,
  };

  COURT.DOUBLES_CORNERS = [
    [0, 0],
    [COURT.DOUBLES_WIDTH, 0],
    [COURT.DOUBLES_WIDTH, COURT.LENGTH],
    [0, COURT.LENGTH],
  ];

  COURT.LINES = [
    [[0, 0], [COURT.DOUBLES_WIDTH, 0]],
    [[0, COURT.LENGTH], [COURT.DOUBLES_WIDTH, COURT.LENGTH]],
    [[0, 0], [0, COURT.LENGTH]],
    [[COURT.DOUBLES_WIDTH, 0], [COURT.DOUBLES_WIDTH, COURT.LENGTH]],
    [[COURT.ALLEY, 0], [COURT.ALLEY, COURT.LENGTH]],
    [[COURT.ALLEY + COURT.SINGLES_WIDTH, 0], [COURT.ALLEY + COURT.SINGLES_WIDTH, COURT.LENGTH]],
    [[COURT.ALLEY, COURT.SERVICE_NEAR_Y], [COURT.ALLEY + COURT.SINGLES_WIDTH, COURT.SERVICE_NEAR_Y]],
    [[COURT.ALLEY, COURT.SERVICE_FAR_Y], [COURT.ALLEY + COURT.SINGLES_WIDTH, COURT.SERVICE_FAR_Y]],
    [[COURT.CENTER_X, COURT.SERVICE_NEAR_Y], [COURT.CENTER_X, COURT.SERVICE_FAR_Y]],
    [[COURT.CENTER_X, 0], [COURT.CENTER_X, 0.15]],
    [[COURT.CENTER_X, COURT.LENGTH - 0.15], [COURT.CENTER_X, COURT.LENGTH]],
  ];

  function rectSignedDistance(r, x, y) {
    const dx = Math.max(r.x0 - x, 0, x - r.x1);
    const dy = Math.max(r.y0 - y, 0, y - r.y1);
    if (dx > 0 || dy > 0) return Math.hypot(dx, dy);
    return -Math.min(x - r.x0, r.x1 - x, y - r.y0, r.y1 - y);
  }

  const SINGLES_COURT = { x0: COURT.ALLEY, y0: 0, x1: COURT.ALLEY + COURT.SINGLES_WIDTH, y1: COURT.LENGTH };
  const DOUBLES_COURT = { x0: 0, y0: 0, x1: COURT.DOUBLES_WIDTH, y1: COURT.LENGTH };

  function courtZone(x, y) {
    const half = y < COURT.NET_Y ? "near" : "far";
    const d = Math.abs(y - COURT.NET_Y);
    const depth = d <= 6.4 ? "short" : d <= 9.4 ? "mid" : "deep";
    const third = COURT.SINGLES_WIDTH / 3;
    const lane = x < COURT.ALLEY + third ? "left" : x < COURT.ALLEY + 2 * third ? "center" : "right";
    return `${half}-${depth}-${lane}`;
  }

  // ------------------------------------------------------------- linalg
  function solveLinear(A, b) {
    // Gaussian elimination with partial pivoting.  A: n x n (array of rows).
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      if (Math.abs(M[piv][col]) < 1e-12) return null;
      [M[col], M[piv]] = [M[piv], M[col]];
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col] / M[col][col];
        for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
      }
    }
    return M.map((row, i) => row[n] / M[i][i]);
  }

  function homographyFromPoints(src, dst) {
    // Exact 4-point homography (h33 = 1): src pixels -> dst meters.
    const A = [], b = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i], [X, Y] = dst[i];
      A.push([x, y, 1, 0, 0, 0, -X * x, -X * y]); b.push(X);
      A.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]); b.push(Y);
    }
    const h = solveLinear(A, b);
    if (!h) return null;
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
  }

  function invert3x3(m) {
    const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-12) return null;
    return [
      [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
      [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
      [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
    ];
  }

  function applyH(H, x, y) {
    const w = H[2][0] * x + H[2][1] * y + H[2][2];
    return [
      (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
      (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
    ];
  }

  // -------------------------------------------------------- calibration
  function Calibration(imageCorners) {
    // imageCorners: 4 [x,y] pixels — near-left, near-right, far-right, far-left.
    const H = homographyFromPoints(imageCorners, COURT.DOUBLES_CORNERS);
    if (!H) return null;
    const Hinv = invert3x3(H);
    return {
      imageToCourt: (x, y) => applyH(H, x, y),
      courtToImage: (x, y) => applyH(Hinv, x, y),
      metersPerPixel(x, y) {
        const c0 = applyH(H, x, y), cx = applyH(H, x + 1, y), cy = applyH(H, x, y + 1);
        return Math.max(Math.hypot(cx[0] - c0[0], cx[1] - c0[1]), Math.hypot(cy[0] - c0[0], cy[1] - c0[1]));
      },
    };
  }

  // ------------------------------------------------------------ tracker
  class Tracker {
    constructor(opts = {}) {
      const scale = opts.scale || 1;
      this.gate = (opts.gatePx || 40) * scale;
      this.maxMissed = opts.maxMissed || 6;
      this.q = opts.processNoise || 8;
      this.r = opts.measurementNoise || 2;
      this.state = null; // [x, y, vx, vy]
      this.P = null;
      this.missed = 0;
      this.pending = [];
      this.points = []; // {frame, x, y, observed}
    }

    _predict() {
      const [x, y, vx, vy] = this.state;
      this.state = [x + vx, y + vy, vx, vy];
      // P = F P F' + Q with F = [[I, I], [0, I]]
      const P = this.P;
      const N = [
        [P[0][0] + P[0][2] + P[2][0] + P[2][2], P[0][1] + P[0][3] + P[2][1] + P[2][3], P[0][2] + P[2][2], P[0][3] + P[2][3]],
        [P[1][0] + P[1][2] + P[3][0] + P[3][2], P[1][1] + P[1][3] + P[3][1] + P[3][3], P[1][2] + P[3][2], P[1][3] + P[3][3]],
        [P[2][0] + P[2][2], P[2][1] + P[2][3], P[2][2], P[2][3]],
        [P[3][0] + P[3][2], P[3][1] + P[3][3], P[3][2], P[3][3]],
      ];
      const Q = [0.25 * this.q, 0.25 * this.q, this.q, this.q];
      for (let i = 0; i < 4; i++) N[i][i] += Q[i];
      this.P = N;
    }

    _update(zx, zy) {
      const P = this.P;
      const S00 = P[0][0] + this.r, S01 = P[0][1], S10 = P[1][0], S11 = P[1][1] + this.r;
      const det = S00 * S11 - S01 * S10;
      const I00 = S11 / det, I01 = -S01 / det, I10 = -S10 / det, I11 = S00 / det;
      const K = [];
      for (let i = 0; i < 4; i++) {
        K.push([P[i][0] * I00 + P[i][1] * I10, P[i][0] * I01 + P[i][1] * I11]);
      }
      const yx = zx - this.state[0], yy = zy - this.state[1];
      for (let i = 0; i < 4; i++) this.state[i] += K[i][0] * yx + K[i][1] * yy;
      const NP = [];
      for (let i = 0; i < 4; i++) {
        NP.push([]);
        for (let j = 0; j < 4; j++) {
          NP[i][j] = P[i][j] - (K[i][0] * P[0][j] + K[i][1] * P[1][j]);
        }
      }
      this.P = NP;
    }

    step(frame, candidates) {
      if (this.state === null) {
        if (candidates.length) {
          const b = candidates[0];
          this.state = [b.x, b.y, 0, 0];
          this.P = [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100]];
          this.missed = 0;
          this.points.push({ frame, x: b.x, y: b.y, observed: true });
        }
        return;
      }
      this._predict();
      const px = this.state[0], py = this.state[1];
      const gate = this.gate * (1 + 0.5 * this.missed);
      let best = null, bestD = gate;
      for (const c of candidates) {
        const d = Math.hypot(c.x - px, c.y - py);
        if (d < bestD) { best = c; bestD = d; }
      }
      if (best) {
        this._update(best.x, best.y);
        this.missed = 0;
        this.points.push(...this.pending);
        this.pending = [];
        this.points.push({ frame, x: this.state[0], y: this.state[1], observed: true });
      } else {
        this.missed += 1;
        if (this.missed > this.maxMissed) {
          this.state = null; this.P = null; this.pending = [];
        } else {
          this.pending.push({ frame, x: px, y: py, observed: false });
        }
      }
    }
  }

  // --------------------------------------------------- events on a track
  function polyfit2(pts) {
    // least-squares y = a t^2 + b t + c over {frame, y}
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, sy = 0, sty = 0, st2y = 0;
    for (const p of pts) {
      const t = p.frame, y = p.y;
      s0 += 1; s1 += t; s2 += t * t; s3 += t * t * t; s4 += t * t * t * t;
      sy += y; sty += t * y; st2y += t * t * y;
    }
    return solveLinear([[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]], [st2y, sty, sy]);
  }

  function refineContact(points, i, win) {
    // Time is centered on the peak frame before fitting: raw frame numbers
    // in the normal equations (t^4 ~ 1e10) lose the precision that decides
    // centimeters.
    const p = points[i];
    const t0 = p.frame;
    const centered = (pts) => pts.map((q) => ({ frame: q.frame - t0, y: q.y }));
    const pre = points.slice(Math.max(0, i - 2 * win), i + 1);
    const post = points.slice(i, i + 2 * win + 1);
    if (pre.length < 3 || post.length < 3) return [p.x, p.y];
    const fa = polyfit2(centered(pre)), fb = polyfit2(centered(post));
    if (!fa || !fb) return [p.x, p.y];
    const A = fa[0] - fb[0], B = fa[1] - fb[1], C = fa[2] - fb[2];
    let roots = [];
    if (Math.abs(A) < 1e-9) {
      if (Math.abs(B) > 1e-9) roots = [-C / B];
    } else {
      const disc = B * B - 4 * A * C;
      if (disc >= 0) {
        const s = Math.sqrt(disc);
        roots = [(-B + s) / (2 * A), (-B - s) / (2 * A)];
      }
    }
    if (!roots.length) return [p.x, p.y];
    let tc = roots[0];
    for (const r of roots) if (Math.abs(r) < Math.abs(tc)) tc = r;
    if (Math.abs(tc) > win) return [p.x, p.y];
    const yc = fa[0] * tc * tc + fa[1] * tc + fa[2];
    // linear interp of x around the contact time (still centered)
    const nb = points.slice(Math.max(0, i - win), i + win + 1);
    let xc = p.x;
    for (let k = 0; k + 1 < nb.length; k++) {
      const fk = nb[k].frame - t0, fk1 = nb[k + 1].frame - t0;
      if (fk <= tc && tc <= fk1) {
        const f = (tc - fk) / Math.max(fk1 - fk, 1e-9);
        xc = nb[k].x + f * (nb[k + 1].x - nb[k].x);
        break;
      }
    }
    return [xc, yc];
  }

  function detectBounces(points, opts = {}) {
    const scale = opts.scale || 1;
    const win = opts.window || 4;
    const minSpeed = (opts.minSpeed || 1.0) * scale;
    const minKink = (opts.minKink || 3.0) * scale;
    const minSep = opts.minSeparation || 5;
    const n = points.length;
    if (n < 2 * win + 1) return [];
    const kink = new Array(n).fill(-Infinity);
    for (let i = win; i < n - win; i++) {
      const vin = (points[i].y - points[i - win].y) / Math.max(points[i].frame - points[i - win].frame, 1);
      const vout = (points[i + win].y - points[i].y) / Math.max(points[i + win].frame - points[i].frame, 1);
      const k = vin - vout;
      if (vin > minSpeed && k > minKink && k > 0.5 * vin) kink[i] = k;
    }
    const order = kink.map((k, i) => [k, i]).filter(([k]) => isFinite(k)).sort((a, b) => b[0] - a[0]);
    const events = [];
    for (const [k, i] of order) {
      const frame = points[i].frame;
      if (events.some((e) => Math.abs(frame - e.frame) < minSep)) continue;
      const [x, y] = refineContact(points, i, win);
      events.push({ frame, x, y, strength: k });
    }
    events.sort((a, b) => a.frame - b.frame);
    return events;
  }

  function detectHits(points, bounces, opts = {}) {
    const scale = opts.scale || 1;
    const win = opts.window || 4;
    const minSpeed = (opts.minSpeed || 1.5) * scale;
    const minSep = opts.minSeparation || 8;
    const n = points.length;
    if (n < 2 * win + 1) return [];
    const bounceFrames = bounces.map((b) => b.frame);
    const events = [];
    let last = -1e9;
    for (let i = win; i < n - win; i++) {
      const f = points[i].frame;
      if (bounceFrames.some((bf) => Math.abs(f - bf) < win + 2)) continue;
      const vin = (points[i].y - points[i - win].y) / Math.max(f - points[i - win].frame, 1);
      const vout = (points[i + win].y - points[i].y) / Math.max(points[i + win].frame - f, 1);
      const rev = (vin > minSpeed && vout < -minSpeed) || (vin < -minSpeed && vout > minSpeed);
      if (rev && f - last >= minSep) {
        events.push({ frame: f, x: points[i].x, y: points[i].y });
        last = f;
      }
    }
    return events;
  }

  function reversesHorizontal(seg, event, opts = {}) {
    const scale = opts.scale || 1;
    const horizon = 6;
    const minTravel = 8 * scale;
    const byFrame = new Map(seg.map((p) => [p.frame, p.x]));
    let before = null, after = null;
    for (let f = event.frame - horizon; f < event.frame; f++) if (byFrame.has(f)) { before = byFrame.get(f); break; }
    for (let f = event.frame + horizon; f > event.frame; f--) if (byFrame.has(f)) { after = byFrame.get(f); break; }
    if (before === null || after === null) return false;
    const dxin = event.x - before, dxout = after - event.x;
    return Math.abs(dxin) > minTravel && Math.abs(dxout) > minTravel && Math.sign(dxin) !== Math.sign(dxout);
  }

  function isTrueBounce(seg, event, calib, opts = {}) {
    const [cx, cy] = calib.imageToCourt(event.x, event.y);
    if (rectSignedDistance(DOUBLES_COURT, cx, cy) > 2.0) return false;
    return !reversesHorizontal(seg, event, opts);
  }

  // ----------------------------------------------------------- line calls
  function makeCall(frame, event, calib, mode) {
    const [cx, cy] = calib.imageToCourt(event.x, event.y);
    const region = mode === "doubles" ? DOUBLES_COURT : SINGLES_COURT;
    const sd = rectSignedDistance(region, cx, cy);
    const margin = COURT.BALL_RADIUS - sd;
    const mpp = calib.metersPerPixel(event.x, event.y);
    const sigma = Math.sqrt(0.03 * 0.03 + (2 * mpp) * (2 * mpp));
    let conf = 1 / (1 + Math.exp(-Math.abs(margin) / Math.max(sigma, 1e-6)));
    conf = (conf - 0.5) * 2;
    return {
      frame,
      imageXY: [event.x, event.y],
      courtXY: [cx, cy],
      decision: margin >= 0 ? "IN" : "OUT",
      marginM: margin,
      confidence: conf,
      zone: courtZone(cx, cy),
    };
  }

  // -------------------------------------------------------------- engine
  const DEFAULTS = {
    mode: "singles",
    scale: 1, // processing-height / 540
    rallyGapFrames: 30,
    segMaxGap: 6,
    segMaxJumpPerFrame: 45,
    ballMinPoints: 5,
    ballMinPathPx: 40,
    ballMinNetPx: 50,
    dedupFrames: 5,
    directionHorizon: 6,
  };

  class Engine {
    constructor(fps, opts = {}) {
      this.cfg = Object.assign({}, DEFAULTS, opts);
      const s = this.cfg.scale;
      this.cfg.segMaxJumpPerFrame *= s;
      this.cfg.ballMinPathPx *= s;
      this.cfg.ballMinNetPx *= s;
      this.fps = fps;
      this.tracker = new Tracker({ scale: s });
      this.calib = null;
      this.frameIdx = -1;
      this.consumed = 0;
      this.seg = []; this.segPath = 0;
      this.rallyOpen = false;
      this.rallyPoints = []; this.rallyBounces = []; this.rallyHits = []; this.rallyCalls = [];
      this.processed = [];
      this.lastBallFrame = null;
      this.stats = { rallies: [], calls: [], zones: {}, speeds: [] };
    }

    setCalibration(imageCorners) {
      this.calib = Calibration(imageCorners);
      return this.calib !== null;
    }

    setFps(fps) { this.fps = fps; }

    segBallLike() {
      const s = this.seg;
      if (s.length < this.cfg.ballMinPoints || this.segPath < this.cfg.ballMinPathPx) return false;
      const net = Math.hypot(s[s.length - 1].x - s[0].x, s[s.length - 1].y - s[0].y);
      return net >= this.cfg.ballMinNetPx;
    }

    step(candidates) {
      this.frameIdx += 1;
      const events = [];
      this.tracker.step(this.frameIdx, candidates);
      const pts = this.tracker.points;
      for (let i = this.consumed; i < pts.length; i++) this._ingest(pts[i], events);
      this.consumed = pts.length;
      if (this.rallyOpen && this.lastBallFrame !== null &&
          this.frameIdx - this.lastBallFrame > this.cfg.rallyGapFrames) {
        this._closeRally(events);
      }
      return events;
    }

    finish() {
      const events = [];
      if (this.rallyOpen) this._closeRally(events);
      return events;
    }

    _ingest(p, events) {
      if (this.seg.length) {
        const prev = this.seg[this.seg.length - 1];
        const df = p.frame - prev.frame;
        const jump = Math.hypot(p.x - prev.x, p.y - prev.y);
        if (df > this.cfg.segMaxGap || jump > this.cfg.segMaxJumpPerFrame * Math.max(df, 1)) {
          this._closeSegment(events);
        }
      }
      if (this.seg.length) this.segPath += Math.hypot(p.x - this.seg[this.seg.length - 1].x, p.y - this.seg[this.seg.length - 1].y);
      this.seg.push(p);
      if (this.segBallLike()) {
        this.lastBallFrame = p.frame;
        if (!this.rallyOpen) {
          this.rallyOpen = true;
          events.push({ type: "rally_start", frame: this.seg[0].frame });
        }
        this._scan(false, events);
      }
    }

    _scan(final, events) {
      if (!this.calib) return;
      const seg = this.seg;
      const tail = seg.slice(-60);
      for (const b of detectBounces(tail, { scale: this.cfg.scale })) {
        if (!final && b.frame + this.cfg.directionHorizon > this.frameIdx) continue;
        if (this.processed.some((f) => Math.abs(b.frame - f) < this.cfg.dedupFrames)) continue;
        this.processed.push(b.frame);
        if (isTrueBounce(seg, b, this.calib, { scale: this.cfg.scale })) {
          this.rallyBounces.push(b);
          const call = makeCall(b.frame, b, this.calib, this.cfg.mode);
          this.rallyCalls.push(call);
          this.stats.calls.push(call);
          this.stats.zones[call.zone] = (this.stats.zones[call.zone] || 0) + 1;
          events.push({ type: "call", frame: this.frameIdx, call });
        } else {
          this.rallyHits.push({ frame: b.frame, x: b.x, y: b.y });
        }
      }
    }

    _closeSegment(events) {
      if (this.segBallLike()) {
        this._scan(true, events);
        const seg = this.seg;
        const segBounces = this.rallyBounces.filter(
          (b) => seg[0].frame <= b.frame && b.frame <= seg[seg.length - 1].frame
        );
        this.rallyHits.push(...detectHits(seg, segBounces, { scale: this.cfg.scale }));
        this.rallyPoints.push(...seg);
      }
      this.seg = []; this.segPath = 0;
    }

    _closeRally(events) {
      this._closeSegment(events);
      this.rallyOpen = false;
      if (this.rallyPoints.length && this.rallyBounces.length) {
        const pts = this.rallyPoints;
        const contacts = this.rallyCalls
          .map((c) => ({ frame: c.frame, xy: c.courtXY }))
          .sort((a, b) => a.frame - b.frame);
        const speeds = [];
        for (let i = 0; i + 1 < contacts.length; i++) {
          const dt = (contacts[i + 1].frame - contacts[i].frame) / this.fps;
          if (dt <= 0) continue;
          const dist = Math.hypot(
            contacts[i + 1].xy[0] - contacts[i].xy[0],
            contacts[i + 1].xy[1] - contacts[i].xy[1]
          );
          const v = (dist / dt) * 3.6;
          if (v > 10 && v < 260) speeds.push(v);
        }
        const rally = {
          index: this.stats.rallies.length + 1,
          startFrame: pts[0].frame,
          endFrame: pts[pts.length - 1].frame,
          shots: Math.max(this.rallyHits.length + 1, this.rallyBounces.length),
          bounces: this.rallyBounces.length,
          durationS: (pts[pts.length - 1].frame - pts[0].frame) / this.fps,
          terminalCall: this.rallyCalls.length ? this.rallyCalls[this.rallyCalls.length - 1].decision : null,
          avgSpeedKmh: speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null,
        };
        this.stats.rallies.push(rally);
        this.stats.speeds.push(...speeds);
        events.push({ type: "rally_end", frame: this.frameIdx, rally });
      }
      this.rallyPoints = []; this.rallyBounces = []; this.rallyHits = []; this.rallyCalls = [];
      this.processed = [];
    }

    summary() {
      const calls = this.stats.calls;
      const nIn = calls.filter((c) => c.decision === "IN").length;
      const speeds = this.stats.speeds;
      return {
        rallies: this.stats.rallies.length,
        calls: calls.length,
        in: nIn,
        out: calls.length - nIn,
        closeCalls: calls.filter((c) => c.confidence < 0.6).length,
        longestRallyShots: this.stats.rallies.reduce((m, r) => Math.max(m, r.shots), 0),
        avgSpeedKmh: speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null,
        maxSpeedKmh: speeds.length ? Math.max(...speeds) : null,
      };
    }
  }

  // ------------------------------------------------------------ detector
  // Works on raw RGBA frames (ImageData.data).  EMA background model with
  // an adaptive threshold, shape + color filters, exclusion zones around
  // large moving objects (players), and static-cell suppression — the same
  // defense stack as the Python detector, tuned for phone cameras.
  class Detector {
    constructor(width, height, opts = {}) {
      this.w = width; this.h = height;
      const scale = height / 540;
      this.scale = scale;
      this.opts = Object.assign(
        {
          minArea: Math.max(3, 4 * scale * scale),
          maxArea: 400 * scale * scale,
          minFill: 0.4,          // area / bbox-area; a disc is ~0.78
          maxAspect: 3.0,
          useColorPrior: true,
          hueRange: [36, 110],   // degrees; optic yellow ~50-70
          minSat: 0.23,
          minVal: 0.35,
          baseThreshold: 22,     // luma delta
          noiseK: 4.0,           // adaptive: threshold >= noiseK * frame noise
          bgAlpha: 0.03,         // background learn rate
          fgAlpha: 0.004,        // learn rate where foreground (don't absorb the ball)
          warmupFrames: 20,
          maxFgFraction: 0.2,    // exposure change guard
          playerMinArea: 3000 * scale * scale,
          playerMarginPx: 24 * scale,
          staticCellPx: Math.max(16, Math.round(32 * scale)),
          staticWindow: 60,
          staticThreshold: 25,
        },
        opts
      );
      const n = width * height;
      this.bg = new Float32Array(n);
      this.mask = new Uint8Array(n);
      this.labels = new Int32Array(n);
      this.frames = 0;
      this.cellHistory = [];
      this.cellCounts = new Map();
    }

    _luma(data, i4) {
      return 0.299 * data[i4] + 0.587 * data[i4 + 1] + 0.114 * data[i4 + 2];
    }

    process(data) {
      const { w, h } = this, o = this.opts, n = w * h;
      this.frames += 1;
      const warm = this.frames <= o.warmupFrames;

      // background diff + noise estimate (sampled)
      let noiseSum = 0, noiseCnt = 0;
      const mask = this.mask;
      const alphaWarm = warm ? 0.25 : o.bgAlpha;
      // First pass: compute diffs, update noise sample
      for (let i = 0, i4 = 0; i < n; i++, i4 += 4) {
        const l = this._luma(data, i4);
        const d = l - this.bg[i];
        if ((i & 127) === 0) { noiseSum += Math.abs(d); noiseCnt++; }
        mask[i] = 0;
        // store diff sign later; update bg after fg decision (two-pass cheap way:
        // use previous threshold decision via current diff)
        const ad = Math.abs(d);
        const thr = this._thr || o.baseThreshold;
        const fg = ad > thr;
        if (fg && !warm) {
          this.bg[i] += o.fgAlpha * d;
          mask[i] = 1;
        } else {
          this.bg[i] += alphaWarm * d;
        }
      }
      const noise = noiseCnt ? noiseSum / noiseCnt : 0;
      this._thr = Math.max(o.baseThreshold, o.noiseK * noise);
      if (warm) return [];

      // exposure-change guard
      let fgCount = 0;
      for (let i = 0; i < n; i++) fgCount += mask[i];
      if (fgCount > o.maxFgFraction * n) {
        // refresh the background quickly and skip the frame
        for (let i = 0, i4 = 0; i < n; i++, i4 += 4) {
          this.bg[i] += 0.5 * (this._luma(data, i4) - this.bg[i]);
        }
        return [];
      }

      // connected components (4-neighborhood, stack flood fill)
      const labels = this.labels;
      labels.fill(0);
      const comps = [];
      const stack = [];
      let nextLabel = 0;
      for (let i = 0; i < n; i++) {
        if (!mask[i] || labels[i]) continue;
        nextLabel += 1;
        let area = 0, sx = 0, sy = 0;
        let minX = w, maxX = 0, minY = h, maxY = 0;
        stack.length = 0; stack.push(i); labels[i] = nextLabel;
        while (stack.length) {
          const p = stack.pop();
          const px = p % w, py = (p / w) | 0;
          area++; sx += px; sy += py;
          if (px < minX) minX = px; if (px > maxX) maxX = px;
          if (py < minY) minY = py; if (py > maxY) maxY = py;
          if (px > 0 && mask[p - 1] && !labels[p - 1]) { labels[p - 1] = nextLabel; stack.push(p - 1); }
          if (px < w - 1 && mask[p + 1] && !labels[p + 1]) { labels[p + 1] = nextLabel; stack.push(p + 1); }
          if (py > 0 && mask[p - w] && !labels[p - w]) { labels[p - w] = nextLabel; stack.push(p - w); }
          if (py < h - 1 && mask[p + w] && !labels[p + w]) { labels[p + w] = nextLabel; stack.push(p + w); }
        }
        comps.push({ area, cx: sx / area, cy: sy / area, minX, maxX, minY, maxY });
      }

      // player exclusion zones: cluster sizeable fragments by proximity, keep
      // clusters whose total area is player-sized
      const bigFrag = comps.filter((c) => c.area > o.playerMinArea / 12);
      const zones = this._clusterZones(bigFrag, o);

      const cells = new Set();
      const out = [];
      for (const c of comps) {
        if (c.area < o.minArea || c.area > o.maxArea) continue;
        const bw = c.maxX - c.minX + 1, bh = c.maxY - c.minY + 1;
        const aspect = bw / bh;
        if (aspect > o.maxAspect || aspect < 1 / o.maxAspect) continue;
        if (c.area / (bw * bh) < o.minFill) continue;
        if (zones.some((z) => c.cx >= z.x0 && c.cx <= z.x1 && c.cy >= z.y0 && c.cy <= z.y1)) continue;
        if (o.useColorPrior && !this._isBallColored(data, c.cx | 0, c.cy | 0)) continue;
        const cell = `${(c.cx / o.staticCellPx) | 0},${(c.cy / o.staticCellPx) | 0}`;
        cells.add(cell);
        if ((this.cellCounts.get(cell) || 0) >= o.staticThreshold) continue;
        out.push({ x: c.cx, y: c.cy, area: c.area });
      }

      // update static map
      this.cellHistory.push(cells);
      for (const cell of cells) this.cellCounts.set(cell, (this.cellCounts.get(cell) || 0) + 1);
      if (this.cellHistory.length > o.staticWindow) {
        for (const cell of this.cellHistory.shift()) {
          const v = (this.cellCounts.get(cell) || 1) - 1;
          if (v <= 0) this.cellCounts.delete(cell); else this.cellCounts.set(cell, v);
        }
      }
      return out;
    }

    _clusterZones(frags, o) {
      // merge overlapping expanded bboxes; keep player-sized clusters
      const boxes = frags.map((c) => ({
        x0: c.minX - o.playerMarginPx, x1: c.maxX + o.playerMarginPx,
        y0: c.minY - o.playerMarginPx, y1: c.maxY + o.playerMarginPx,
        area: c.area,
      }));
      let merged = true;
      while (merged) {
        merged = false;
        outer: for (let i = 0; i < boxes.length; i++) {
          for (let j = i + 1; j < boxes.length; j++) {
            const a = boxes[i], b = boxes[j];
            if (a.x0 <= b.x1 && b.x0 <= a.x1 && a.y0 <= b.y1 && b.y0 <= a.y1) {
              a.x0 = Math.min(a.x0, b.x0); a.x1 = Math.max(a.x1, b.x1);
              a.y0 = Math.min(a.y0, b.y0); a.y1 = Math.max(a.y1, b.y1);
              a.area += b.area;
              boxes.splice(j, 1);
              merged = true;
              break outer;
            }
          }
        }
      }
      return boxes.filter((b) => b.area > this.opts.playerMinArea);
    }

    _isBallColored(data, x, y) {
      const { w, h } = this, o = this.opts;
      let r = 0, g = 0, b = 0, cnt = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const px = x + dx, py = y + dy;
          if (px < 0 || px >= w || py < 0 || py >= h) continue;
          const i4 = (py * w + px) * 4;
          r += data[i4]; g += data[i4 + 1]; b += data[i4 + 2]; cnt++;
        }
      }
      r /= cnt * 255; g /= cnt * 255; b /= cnt * 255;
      const max = Math.max(r, g, b), min = Math.min(r, g, b);
      const v = max, s = max === 0 ? 0 : (max - min) / max;
      let hDeg = 0;
      if (max !== min) {
        if (max === r) hDeg = 60 * (((g - b) / (max - min)) % 6);
        else if (max === g) hDeg = 60 * ((b - r) / (max - min) + 2);
        else hDeg = 60 * ((r - g) / (max - min) + 4);
        if (hDeg < 0) hDeg += 360;
      }
      return hDeg >= o.hueRange[0] && hDeg <= o.hueRange[1] && s >= o.minSat && v >= o.minVal;
    }
  }

  return { COURT, SINGLES_COURT, DOUBLES_COURT, rectSignedDistance, courtZone,
           homographyFromPoints, invert3x3, applyH, Calibration,
           Tracker, detectBounces, detectHits, isTrueBounce, makeCall,
           Engine, Detector };
});
