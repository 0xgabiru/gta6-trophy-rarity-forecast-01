import * as fs from "node:fs";
import * as path from "node:path";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const EPOCH_MS = Date.parse("2026-11-19T00:00:00Z");

function clamp(x: number): number {
  return Math.min(99.99, Math.max(0.01, x));
}

function logit(p: number): number {
  const q = clamp(p) / 100;
  return Math.log(q / (1 - q));
}

function invLogit(l: number): number {
  return 100 / (1 + Math.exp(-l));
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function hoursOf(iso: string): number {
  return (Date.parse(iso) - EPOCH_MS) / 3600000;
}

// Round, clamp to [0.01, 99.99], then enforce non-decreasing by running max.
function finishQs(qs: number[]): number[] {
  const out = qs.map((v) => clamp(round4(v)));
  for (let i = 1; i < out.length; i++) {
    if (out[i] < out[i - 1]) out[i] = out[i - 1];
  }
  return out;
}

function quantilesFrom(median: number, sigma: number): number[] {
  const lm = logit(median);
  return finishQs(Z.map((z) => invLogit(lm + z * sigma)));
}

interface Trophy {
  id: number;
  tier: string;
}

interface State {
  h: number;
  rate: number;
}

const DEPTH_BY_ID: Record<number, number> = {
  2: 0,
  3: 1,
  4: 2,
  5: 3,
  6: 4,
  7: 5,
  14: 10,
  1: 10,
};

const CEILING_BY_TIER: Record<string, number> = {
  platinum: 0.8,
  gold: 25,
  silver: 12,
  bronze: 45,
};

function dilutionQuantiles(t: Trophy, last: number): number[] {
  let depth: number;
  if (t.id in DEPTH_BY_ID) {
    depth = DEPTH_BY_ID[t.id];
  } else if (t.tier === "silver") {
    depth = 6;
  } else if (t.tier === "gold") {
    depth = 8;
  } else {
    depth = 5;
  }
  const f = 0.15 + 0.45 * Math.exp(-depth / 2);
  const m = Math.max(last * f, 0.011);
  return quantilesFrom(m, 1.2);
}

function growthQuantiles(
  t: Trophy,
  pts: State[],
  last: number,
  cut: number,
  tgt: number,
): number[] {
  const weight = (h: number): number => Math.pow(0.5, (cut - h) / 12);
  const xAt = (h: number): number => Math.log(1 + h);
  const xTgt = xAt(tgt);

  // (a) Recency-weighted logit-linear fit on the 12h window (fallback 24h,
  // then 48h when refreshes are sparse: a window with fewer than 3 distinct
  // states cannot identify a slope, and older points — already downweighted
  // by 0.5^((cut-h)/12) — beat no points at all).
  let win = pts.filter((p) => p.h >= cut - 12);
  if (win.length < 3) win = pts.filter((p) => p.h >= cut - 24);
  if (win.length < 3) win = pts.filter((p) => p.h >= cut - 48);
  if (win.length === 0) win = pts.slice(-3);

  let m1: number;
  let res1: number;
  if (win.length < 3) {
    m1 = last;
    res1 = 1.0;
  } else {
    let sw = 0;
    let swx = 0;
    let swy = 0;
    let swxx = 0;
    let swxy = 0;
    for (const p of win) {
      const wi = weight(p.h);
      const x = xAt(p.h);
      const y = logit(p.rate);
      sw += wi;
      swx += wi * x;
      swy += wi * y;
      swxx += wi * x * x;
      swxy += wi * x * y;
    }
    const den = sw * swxx - swx * swx;
    let b1 = 0;
    let a1 = swy / sw;
    if (den !== 0) {
      b1 = (sw * swxy - swx * swy) / den;
      a1 = (swy - b1 * swx) / sw;
    }
    let sse = 0;
    for (const p of win) {
      const wi = weight(p.h);
      const r = logit(p.rate) - (a1 + b1 * xAt(p.h));
      sse += wi * r * r;
    }
    res1 = Math.sqrt(sse / sw);
    m1 = invLogit(a1 + b1 * xTgt);
  }

  // (b) Saturating fit: rate approaches a tier ceiling in logit space.
  const c = CEILING_BY_TIER[t.tier] ?? 45;
  const fracLogit = (rate: number): number => {
    const f = Math.min(0.9999, Math.max(0.0001, rate / c));
    return Math.log(f / (1 - f));
  };
  let bestErr = Infinity;
  let bestA = 0;
  let bestB = 0;
  for (let k = 0; k <= 24; k++) {
    const b = k * 0.25;
    let sw = 0;
    let s = 0;
    for (const p of win) {
      const wi = weight(p.h);
      sw += wi;
      s += wi * (fracLogit(p.rate) - b * xAt(p.h));
    }
    const a = s / sw;
    let sse = 0;
    for (const p of win) {
      const wi = weight(p.h);
      const r = fracLogit(p.rate) - (a + b * xAt(p.h));
      sse += wi * r * r;
    }
    const err = sse / sw;
    if (err < bestErr) {
      bestErr = err;
      bestA = a;
      bestB = b;
    }
  }
  // Shrink the slope toward 2.0, then re-solve the intercept with the new slope.
  const n = pts.length;
  const b2 = (n / (n + 3)) * bestB + (3 / (n + 3)) * 2.0;
  let sw2 = 0;
  let s2 = 0;
  for (const p of win) {
    const wi = weight(p.h);
    sw2 += wi;
    s2 += wi * (fracLogit(p.rate) - b2 * xAt(p.h));
  }
  const a2 = s2 / sw2;
  let sse2 = 0;
  for (const p of win) {
    const wi = weight(p.h);
    const r = fracLogit(p.rate) - (a2 + b2 * xAt(p.h));
    sse2 += wi * r * r;
  }
  const err2 = sse2 / sw2;
  let m2 = c / (1 + Math.exp(-(a2 + b2 * xTgt)));
  // Damp toward the last rate (0.7 / 0.3 in log space).
  m2 = Math.exp(
    0.7 * Math.log(Math.max(m2, 0.011)) + 0.3 * Math.log(Math.max(last, 0.011)),
  );

  // (c) Ensemble by inverse-residual weights, in log space.
  const w1 = 1 / (res1 + 0.15);
  const w2 = 1 / (Math.sqrt(err2) + 0.15);
  let m = Math.exp(
    (w1 * Math.log(Math.max(m1, 0.011)) + w2 * Math.log(Math.max(m2, 0.011))) /
      (w1 + w2),
  );
  m = Math.min(Math.max(m, last / 3), Math.max(last * 40, 0.5));
  m = Math.min(Math.max(m, 0.011), 99.98);

  // (d) Spread from both fits' residuals.
  const sigma =
    0.35 + 0.5 * Math.min(res1, 1) + 0.5 * Math.min(Math.sqrt(err2), 1);
  return quantilesFrom(m, sigma);
}

interface CaseInput {
  id: string;
  at: string;
  target: string;
  trophies: Trophy[];
  history: {
    at: string;
    players_tracked: number;
    rates: (number | null)[];
  }[];
}

function forecastCase(c: CaseInput): { trophies: { id: number; q: number[] }[] } {
  const cut = hoursOf(c.at);
  const tgt = hoursOf(c.target);
  const rows: { id: number; q: number[] }[] = [];
  const byId = new Map<number, { id: number; q: number[] }>();

  // Page-freeze detection (global, from players_tracked, not per-trophy rates:
  // a single trophy's rate can sit still on a live page, but a frozen
  // players_tracked means Exophase has not refreshed at all). If the page has
  // not refreshed in the 24h before the cut, no recent trend is observable:
  // extrapolating a trend fitted to stale points invents growth that may have
  // ended days ago, so fall back to the last rate with the branch's own
  // no-information spread (sigma = 0.35 + 0.5 + 0.5 = 1.35).
  let lastPlayersChangeH = -Infinity;
  let prevPlayers: number | null = null;
  for (const read of c.history) {
    if (typeof read.players_tracked === "number") {
      if (prevPlayers !== null && read.players_tracked !== prevPlayers) {
        lastPlayersChangeH = hoursOf(read.at);
      }
      prevPlayers = read.players_tracked;
    }
  }
  const pageFrozen = cut - lastPlayersChangeH > 24;

  for (let i = 0; i < c.trophies.length; i++) {
    const t = c.trophies[i];
    // Distinct refresh states, oldest to newest; skip nulls; keep a read only
    // when its rate differs from the previously kept rate by > 1e-9.
    const pts: State[] = [];
    let prevKept: number | null = null;
    const playersSeen = new Set<number>();
    for (const read of c.history) {
      if (typeof read.players_tracked === "number") {
        playersSeen.add(read.players_tracked);
      }
      const r = read.rates[i];
      if (r === null || r === undefined) continue;
      if (prevKept === null || Math.abs(r - prevKept) > 1e-9) {
        pts.push({ h: hoursOf(read.at), rate: r });
        prevKept = r;
      }
    }
    const nStates = playersSeen.size;

    let q: number[];
    if (pts.length === 0) {
      q = quantilesFrom(0.011, 1.2);
    } else {
      const last = pts[pts.length - 1].rate;
      if (last >= 99.99) {
        // The 100% trophy: left-skewed band, never exactly 100.
        q = [99.0, 99.7, 99.95, 99.99, 99.99];
      } else if (nStates <= 2) {
        q = dilutionQuantiles(t, last);
      } else if (pageFrozen) {
        q = quantilesFrom(last, 1.35);
      } else {
        q = growthQuantiles(t, pts, last, cut, tgt);
      }
    }
    const row = { id: t.id, q };
    rows.push(row);
    byId.set(t.id, row);
  }

  // (e) Story-chain monotonicity: medians of ids [3,4,5,6,7] non-increasing.
  const CHAIN = [3, 4, 5, 6, 7];
  const present = CHAIN.filter((id) => byId.has(id));
  if (present.length >= 2) {
    const oldMed = present.map((id) => byId.get(id)!.q[2]);
    const med = oldMed.slice();
    for (let pass = 0; pass < 10; pass++) {
      let j = -1;
      for (let k = 0; k + 1 < med.length; k++) {
        if (med[k] < med[k + 1]) {
          j = k;
          break;
        }
      }
      if (j < 0) break;
      const mean = (med[j] + med[j + 1]) / 2;
      med[j] = mean;
      med[j + 1] = mean;
    }
    for (let k = 0; k < present.length; k++) {
      if (med[k] !== oldMed[k]) {
        const row = byId.get(present[k])!;
        const delta = logit(med[k]) - logit(oldMed[k]);
        row.q = finishQs(row.q.map((v) => invLogit(logit(v) + delta)));
      }
    }
  }

  return { trophies: rows };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    const c = JSON.parse(
      fs.readFileSync(path.join(inputDir, f), "utf8"),
    ) as CaseInput;
    const out = forecastCase(c);
    const id = f.slice(0, -".in".length);
    fs.writeFileSync(
      path.join(outputDir, id + ".out"),
      JSON.stringify(out) + "\n",
    );
  }
}

main();
