/* Node tests for the JS engine port.
 *
 *   node webapp/test/test_core.mjs
 *
 * 1. Parity: Python-detector candidates in -> the JS engine must call every
 *    ground-truth bounce correctly, with no spurious calls.
 * 2. Geometry/homography unit checks.
 * 3. JS Detector end-to-end on synthetic RGBA frames rendered in JS.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const core = require(join(here, "..", "core.js"));

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) console.log(`  ok  ${name}`);
  else { console.error(`FAIL  ${name}  ${detail}`); failures++; }
}

// ---------------------------------------------------------------- geometry
{
  const { COURT, rectSignedDistance, SINGLES_COURT, Calibration } = core;
  check("court dims", COURT.DOUBLES_WIDTH === 10.97 && COURT.LENGTH === 23.77);
  check("center is inside", rectSignedDistance(SINGLES_COURT, 5.485, 11.885) < 0);
  check(
    "corner distance is euclidean",
    Math.abs(rectSignedDistance(SINGLES_COURT, COURT.ALLEY - 0.3, -0.4) - 0.5) < 1e-9
  );

  // homography round trip on a synthetic quad
  const src = [[236, 434], [723, 434], [652, 175], [307, 175]];
  const calib = Calibration(src);
  for (let i = 0; i < 4; i++) {
    const [cx, cy] = calib.imageToCourt(src[i][0], src[i][1]);
    const t = COURT.DOUBLES_CORNERS[i];
    check(`corner ${i} maps exactly`, Math.hypot(cx - t[0], cy - t[1]) < 1e-6);
  }
  const [ix, iy] = calib.courtToImage(5.485, 11.885);
  const [bx, by] = calib.imageToCourt(ix, iy);
  check("roundtrip", Math.hypot(bx - 5.485, by - 11.885) < 1e-9);
}

// ------------------------------------------------------------------ parity
{
  const fixture = JSON.parse(readFileSync(join(here, "fixture.json"), "utf8"));
  const scale = fixture.height / 540;
  const engine = new core.Engine(fixture.fps, { scale });
  engine.setCalibration(fixture.corner_pixels);

  for (const frameCands of fixture.candidates) {
    engine.step(frameCands.map(([x, y, area]) => ({ x, y, area })));
  }
  engine.finish();

  const calls = engine.stats.calls;
  const truth = fixture.truth_bounces;
  check(
    `no spurious calls (${calls.length} calls for ${truth.length} bounces)`,
    calls.length === truth.length,
    JSON.stringify(calls.map((c) => [c.frame, c.decision]))
  );
  // Position tolerances reflect the per-frame-JPEG condition this fixture
  // uses (the phone path): the Python pipeline itself lands 0.60 m on the
  // worst near-court bounce here.  Decisions must all be correct.
  let correct = 0;
  const errs = [];
  for (const tb of truth) {
    let best = null;
    for (const c of calls) if (!best || Math.abs(c.frame - tb.frame) < Math.abs(best.frame - tb.frame)) best = c;
    const near = best && Math.abs(best.frame - tb.frame) <= 6;
    const decisionOk = near && best.decision === tb.expected_call;
    const err = near
      ? Math.hypot(best.courtXY[0] - tb.court_xy[0], best.courtXY[1] - tb.court_xy[1])
      : Infinity;
    errs.push(err);
    check(
      `truth bounce @${tb.frame} ${tb.expected_call}`,
      decisionOk && err < 0.7,
      near ? `called ${best.decision} err ${err.toFixed(2)}m` : "no nearby call"
    );
    if (decisionOk) correct++;
  }
  check("all decisions correct", correct === truth.length, `${correct}/${truth.length}`);
  const meanErr = errs.reduce((a, b) => a + b, 0) / errs.length;
  check("mean position error < 0.30 m", meanErr < 0.3, meanErr.toFixed(3));
  check("rallies segmented", engine.stats.rallies.length === 4, `${engine.stats.rallies.length}`);
  const s = engine.summary();
  check("summary consistent", s.in + s.out === calls.length && s.rallies === 4);
}

// --------------------------------------------- JS detector e2e (synthetic)
{
  // Tiny synthetic scene rendered directly into RGBA buffers: static green
  // court-ish background, a swaying dark "player", and a yellow ball that
  // falls, bounces at a known pixel, and rises.  The full JS stack
  // (Detector -> Engine) must call that bounce.
  const W = 480, H = 270;
  const scale = H / 540;
  const detector = new core.Detector(W, H, { warmupFrames: 10 });
  const engine = new core.Engine(30, { scale });
  // identity-ish calibration: a plausible court quad in this small frame
  engine.setCalibration([[80, 240], [400, 240], [340, 80], [140, 80]]);

  const frame = new Uint8ClampedArray(W * H * 4);
  function paintBG() {
    for (let i = 0; i < W * H; i++) {
      frame[i * 4] = 60; frame[i * 4 + 1] = 120; frame[i * 4 + 2] = 70; frame[i * 4 + 3] = 255;
    }
  }
  function disc(cx, cy, r, rgb) {
    for (let y = Math.max(0, cy - r | 0); y <= Math.min(H - 1, cy + r | 0); y++) {
      for (let x = Math.max(0, cx - r | 0); x <= Math.min(W - 1, cx + r | 0); x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
          const i4 = (y * W + x) * 4;
          frame[i4] = rgb[0]; frame[i4 + 1] = rgb[1]; frame[i4 + 2] = rgb[2];
        }
      }
    }
  }
  function rect(x0, y0, x1, y1, rgb) {
    for (let y = Math.max(0, y0 | 0); y <= Math.min(H - 1, y1 | 0); y++) {
      for (let x = Math.max(0, x0 | 0); x <= Math.min(W - 1, x1 | 0); x++) {
        const i4 = (y * W + x) * 4;
        frame[i4] = rgb[0]; frame[i4 + 1] = rgb[1]; frame[i4 + 2] = rgb[2];
      }
    }
  }

  // ballistic in pixels: bounce at frame 60 at (240, 200)
  const bounceF = 60, bx = 240, by = 200;
  const totalFrames = 120;
  let calls = 0, decision = null, callPix = null;
  for (let f = 0; f < totalFrames; f++) {
    paintBG();
    const sway = 12 * Math.sin(f * 0.15);
    rect(60 + sway, 150, 90 + sway, 230, [40, 40, 90]); // "player"
    if (f >= 20 && f <= 110) {
      const t = f - bounceF;
      const ballX = bx + 1.6 * t;
      // real bounce kinematics: falling with speed AT contact, then rising
      const ballY = t <= 0 ? by + 3.0 * t - 0.03 * t * t : by - (2.2 * t - 0.03 * t * t);
      disc(ballX, ballY, 2.5, [235, 220, 60]);
    }
    const events = engine.step(detector.process(frame));
    for (const e of events) {
      if (e.type === "call") {
        calls++;
        decision = e.call.decision;
        callPix = e.call.imageXY;
      }
    }
  }
  engine.finish();
  check("JS detector+engine finds exactly one bounce", calls === 1, `calls=${calls}`);
  if (calls === 1) {
    check(
      "bounce localized near truth pixel",
      Math.hypot(callPix[0] - bx, callPix[1] - by) < 8 * scale + 4,
      `at ${callPix.map((v) => v.toFixed(1))} vs (${bx}, ${by})`
    );
  }
}

console.log(failures ? `\n${failures} FAILURES` : "\nall tests passed");
process.exit(failures ? 1 : 0);
