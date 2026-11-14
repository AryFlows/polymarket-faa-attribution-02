// FAA attribution price forecaster (Node 22, standard library only).
//
// Forecast for each market: a persistence median with a per-market,
// sub-diffusive spread and an empirically shaped band:
//
//   q_tau(h) = clamp(x + shape_tau(h) * sigma(h), 0, 1), rounded to 6 dp
//
// where x is the price at the checkpoint.
//
// - sigma(h) = sigma1 * h^alpha, fitted per market from the standard
//   deviation of h-minute price changes at anchor horizons over the recent
//   window. Price dispersion grows far slower than sqrt(h) (alpha ~ 0.25 on
//   the practice data), so the house baseline's fixed 0.0025*sqrt(h) band is
//   far too wide at long horizons. sigma(h) is floored at a fraction of the
//   baseline band, so a market that sat still recently still carries a
//   jump allowance.
// - shape_tau(h) is the empirical quantile of pooled standardised changes:
//   every market's h-minute changes, centred by their median and scaled by
//   their own standard deviation, pooled across the case's markets. Prices
//   move in steps, so the change distribution is sharply peaked at zero
//   with fat tails: the inner quantiles sit far closer to the median than a
//   normal band puts them, while the 5%/95% tails are wider than normal.
//   The shape is re-estimated from each case's own history, so it adapts to
//   the case's regime instead of assuming one.
//
// The centre is the last price: out-of-sample, neither the recent trend, nor
// the deviation from the recent mean, nor the last jump predicts the next
// 4 hours' move on the practice data (each adjustment tested made the
// backtest worse), so persistence is the honest centre.
//
// The program reads every <id>.in in the input directory and writes one
// <id>.out per case. It is deterministic: no clock, no randomness, and the
// same input always produces the same bytes.
//
//   MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Fallback shape: the standard normal's quantile at each level. */
const NORMAL_Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

/** Minutes of recent history used to estimate spread and shape. */
const WINDOW = 300;
/** Anchor horizons (minutes) at which spread and shape are measured. */
const ANCHOR_HORIZONS = [1, 5, 15, 30, 60, 120];
/** Minimum samples for an anchor to count. */
const MIN_SAMPLES = 20;
/** Bounds on the fitted diffusion exponent. */
const ALPHA_MIN = 0.15;
const ALPHA_MAX = 0.5;
/** The house baseline's per-square-root-minute spread (the floor's scale). */
const BASELINE_SIGMA = 0.0025;
/** The floor on sigma(h) is FLOOR_MULT times the baseline band at h. */
const FLOOR_MULT = 0.25;
/** Cap on sigma(h), in price units. */
const SIGMA_CAP = 0.45;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0, 1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Population standard deviation of the values in arr. */
function pstdev(arr: number[]): number {
  if (arr.length < 2) return 0;
  let m = 0;
  for (const v of arr) m += v;
  m /= arr.length;
  let ss = 0;
  for (const v of arr) ss += (v - m) * (v - m);
  return Math.sqrt(ss / arr.length);
}

/** Median of arr (a copy is sorted; arr is left alone). */
function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = (s.length - 1) / 2;
  const lo = Math.floor(mid);
  const hi = Math.ceil(mid);
  return (s[lo] as number) + ((s[hi] as number) - (s[lo] as number)) * (mid - lo);
}

/** Empirical quantile of the sorted array s at level p (linear interpolation). */
function quantileSorted(s: number[], p: number): number {
  if (s.length === 0) return 0;
  const i = (s.length - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.min(lo + 1, s.length - 1);
  return (s[lo] as number) + ((s[hi] as number) - (s[lo] as number)) * (i - lo);
}

/**
 * Fit log sigma = log sigma1 + alpha * log h through the (h, sigma) anchor
 * points by least squares. Returns [sigma1, alpha] with alpha clamped.
 */
function fitPowerLaw(points: Array<[number, number]>): [number, number] {
  const usable = points.filter(([, s]) => s > 0);
  if (usable.length === 0) return [BASELINE_SIGMA, 0.5];
  if (usable.length === 1) return [usable[0]![1] / Math.pow(usable[0]![0], 0.3), 0.3];
  let sx = 0;
  let sy = 0;
  for (const [h, s] of usable) {
    sx += Math.log(h);
    sy += Math.log(s);
  }
  const mx = sx / usable.length;
  const my = sy / usable.length;
  let num = 0;
  let den = 0;
  for (const [h, s] of usable) {
    const dx = Math.log(h) - mx;
    num += dx * (Math.log(s) - my);
    den += dx * dx;
  }
  let alpha = den > 0 ? num / den : 0.3;
  alpha = Math.min(ALPHA_MAX, Math.max(ALPHA_MIN, alpha));
  const sigma1 = Math.exp(my - alpha * mx);
  return [sigma1, alpha];
}

/** Per-market spread fit plus the pooled standardised changes per anchor. */
interface MarketStats {
  sigma1: number;
  alpha: number;
}

/**
 * Measure one market: the anchor standard deviations of its h-minute
 * changes, and its standardised changes (for the pooled shape). Changes are
 * taken over the recent WINDOW minutes only, so the estimates track the
 * current regime.
 */
function measureMarket(
  prices: number[],
  pooled: Map<number, number[]>,
): MarketStats {
  const n = prices.length;
  const points: Array<[number, number]> = [];
  const start = Math.max(0, n - WINDOW);
  for (const h of ANCHOR_HORIZONS) {
    const t0 = start;
    const t1 = n - h - 1;
    if (t1 - t0 + 1 < MIN_SAMPLES) continue;
    const changes: number[] = [];
    for (let t = t0; t <= t1; t++) changes.push((prices[t + h] as number) - (prices[t] as number));
    const s = pstdev(changes);
    if (s > 0) {
      points.push([h, s]);
      const med = median(changes);
      let bucket = pooled.get(h);
      if (bucket === undefined) {
        bucket = [];
        pooled.set(h, bucket);
      }
      for (const c of changes) bucket.push((c - med) / s);
    }
  }
  const [sigma1, alpha] = fitPowerLaw(points);
  return { sigma1, alpha };
}

/**
 * The pooled shape: for each anchor horizon, the empirical quantiles of the
 * standardised changes at the five levels. Anchors with too few samples
 * fall back to the normal shape.
 */
function pooledShape(pooled: Map<number, number[]>): Map<number, number[]> {
  const shape = new Map<number, number[]>();
  for (const h of ANCHOR_HORIZONS) {
    const bucket = pooled.get(h);
    if (bucket === undefined || bucket.length < 100) {
      shape.set(h, [...NORMAL_Z]);
      continue;
    }
    const sorted = [...bucket].sort((a, b) => a - b);
    shape.set(h, QUANTILES.map((t) => quantileSorted(sorted, t)));
  }
  return shape;
}

/** Shape at horizon h: log-linear interpolation between the anchor shapes. */
function shapeAt(shape: Map<number, number[]>, h: number): number[] {
  const anchors = ANCHOR_HORIZONS.filter((a) => shape.has(a));
  if (anchors.length === 0) return [...NORMAL_Z];
  const first = anchors[0] as number;
  const last = anchors[anchors.length - 1] as number;
  if (h <= first) return shape.get(first) as number[];
  if (h >= last) return shape.get(last) as number[];
  let lo = first;
  let hi = last;
  for (let i = 0; i < anchors.length - 1; i++) {
    if (h >= (anchors[i] as number) && h <= (anchors[i + 1] as number)) {
      lo = anchors[i] as number;
      hi = anchors[i + 1] as number;
      break;
    }
  }
  const w = (Math.log(h) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
  const zlo = shape.get(lo) as number[];
  const zhi = shape.get(hi) as number[];
  return zlo.map((z, j) => z + ((zhi[j] as number) - z) * w);
}

/** The forecast of one market: `horizon` rows of the five quantiles. */
function forecastMarket(
  prices: number[],
  stats: MarketStats,
  shape: Map<number, number[]>,
  horizon: number,
): number[][] {
  const n = prices.length;
  const x = n > 0 ? (prices[n - 1] as number) : 0.5;
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const floor = FLOOR_MULT * BASELINE_SIGMA * Math.sqrt(h);
    const sig = Math.min(SIGMA_CAP, Math.max(floor, stats.sigma1 * Math.pow(h, stats.alpha)));
    const z = shapeAt(shape, h);
    const row = z.map((zj) => price6(x + zj * sig));
    // Rounding and clamping preserve order, but enforce it outright: the
    // shape's inner quantiles can coincide, never cross.
    for (let j = 1; j < row.length; j++) {
      if ((row[j] as number) < (row[j - 1] as number)) row[j] = row[j - 1] as number;
    }
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }
  // First pass: per-market spread fits and the pooled shape.
  const pooled = new Map<number, number[]>();
  const stats = c.markets.map((m) => measureMarket(m.prices, pooled));
  const shape = pooledShape(pooled);
  const forecasts = c.markets.map((m, i) => ({
    id: m.id,
    q: forecastMarket(m.prices, stats[i] as MarketStats, shape, c.horizon_minutes),
  }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
