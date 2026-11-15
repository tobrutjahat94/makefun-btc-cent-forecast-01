// btc-cent-forecast-01 entry (Node 22, standard library only).
//
// Idea: Bitcoin's 4-hour move is not predictable from where it sits alone, but
// the shape of the last 72 hours carries weak, honest signal — short-horizon
// momentum, mean reversion toward slow moving averages, and how jumpy the tape
// has been lately. This program fits that shape inside the case itself, with
// nothing carried over from any other case:
//
//   - work in log prices, so a move is a return, not a dollar amount;
//   - at every minute t, describe the tape with 13 numbers: log returns over
//     5 / 15 / 60 / 240 / 720 / 1440 minutes, the log price's distance from
//     its 60 / 240 / 720 / 1440 / 4320-minute moving averages, and the
//     realised volatility of 1-minute log returns over 60 and 240 minutes;
//   - the target at t is the log return over the next 240 minutes;
//   - fit a ridge regression of target on features over all t where the
//     target is observable (about 2,640 rows), standardising both sides
//     first, so the penalty treats every feature on the same scale;
//   - forecast from the features at the last close, then clip the predicted
//     log return to a few standard deviations of the training targets, so a
//     wild fit can never throw the guess far from the checkpoint.
//
// Everything is deterministic and computed from the one case file. If anything
// about the model fails for a case, the guess falls back to the checkpoint's
// own close (persistence), so the output is always a valid price.
//
// Tunables (defaults are the shipped ones; the grader runs with none set):
//   MF_LAMBDA — ridge penalty on standardised features (default 100)
//   MF_CLIP   — prediction clip, in sigmas of the training targets (default 3)
//   MF_SHRINK — shrink factor on the predicted log return (default 1)

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Smallest positive price on the grid: one unit of its 5 decimals. */
const MIN_PRICE = 0.00001;
/** Largest grid price below the scorer's exclusive 10,000,000-dollar cap. */
const MAX_PRICE = 9_999_999.99999;

function numEnv(name: string, fallback: number): number {
  const v = Number(process.env[name]);
  return Number.isFinite(v) && v > 0 ? v : fallback;
}

const LAMBDA = numEnv('MF_LAMBDA', 100);
const CLIP_SIGMA = numEnv('MF_CLIP', 3);
const SHRINK = numEnv('MF_SHRINK', 1);

interface CaseMarket {
  id: string;
  prices: number[];
}

interface CaseFile {
  case?: string;
  markets: CaseMarket[];
}

/** x kept on the grid's 5 decimals and inside the scorer's (0, 10,000,000). */
const price5 = (x: number): number =>
  Math.round(Math.min(MAX_PRICE, Math.max(MIN_PRICE, x)) * 1e5) / 1e5;

/** Solve A x = b by Gaussian elimination with partial pivoting. A is n x n, b length n. */
function solveLinear(aIn: number[][], bIn: number[]): number[] {
  const n = bIn.length;
  const a = aIn.map((row) => row.slice());
  const b = bIn.slice();
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(a[r]![col]!) > Math.abs(a[pivot]![col]!)) pivot = r;
    }
    if (Math.abs(a[pivot]![col]!) < 1e-12) throw new Error('singular system');
    if (pivot !== col) {
      [a[col], a[pivot]] = [a[pivot]!, a[col]!];
      [b[col], b[pivot]] = [b[pivot]!, b[col]!];
    }
    const diag = a[col]![col]!;
    for (let r = col + 1; r < n; r++) {
      const f = a[r]![col]! / diag;
      if (f === 0) continue;
      for (let c = col; c < n; c++) a[r]![c]! -= f * a[col]![c]!;
      b[r]! -= f * b[col]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r]!;
    for (let c = r + 1; c < n; c++) s -= a[r]![c]! * x[c]!;
    x[r] = s / a[r]![r]!;
  }
  return x;
}

const RETURN_LAGS = [5, 15, 60, 240, 720, 1440];
const MA_WINDOWS = [60, 240, 720, 1440, 4320];
const VOL_WINDOWS = [60, 240];
const N_FEATURES = RETURN_LAGS.length + MA_WINDOWS.length + VOL_WINDOWS.length; // 13
const HORIZON = 240;
const MAX_LAG = 1440;

/** The 13 features of the tape at minute t (0-based index into log prices). */
function featuresAt(
  logs: number[],
  logPrefix: number[],
  retPrefix: number[],
  retSqPrefix: number[],
  t: number,
): number[] {
  const out = new Array<number>(N_FEATURES);
  let i = 0;
  for (const k of RETURN_LAGS) out[i++] = logs[t]! - logs[t - k]!;
  for (const k of MA_WINDOWS) {
    const w = Math.min(k, t + 1);
    const mean = (logPrefix[t + 1]! - logPrefix[t + 1 - w]!) / w;
    out[i++] = logs[t]! - mean;
  }
  for (const k of VOL_WINDOWS) {
    const w = Math.min(k, t); // w one-minute returns ending at t
    const s = retPrefix[t + 1]! - retPrefix[t + 1 - w]!;
    const s2 = retSqPrefix[t + 1]! - retSqPrefix[t + 1 - w]!;
    const mean = s / w;
    const variance = Math.max(0, s2 / w - mean * mean);
    out[i++] = Math.sqrt(variance);
  }
  return out;
}

/** Forecast the close 240 minutes after the last price in `prices`. */
function forecastPrice(prices: number[]): number {
  const last = prices[prices.length - 1]!;
  if (!Array.isArray(prices) || prices.length < MAX_LAG + HORIZON + 60) {
    return price5(last); // too little history to fit: stay at the checkpoint
  }
  for (const p of prices) {
    if (typeof p !== 'number' || !Number.isFinite(p) || p <= 0) {
      return price5(last);
    }
  }
  const n = prices.length;
  const logs = prices.map((p) => Math.log(p));

  // Prefix sums: log prices (for moving averages), 1-minute returns and their
  // squares (for realised volatility). All feature lookups below are O(1).
  const logPrefix = new Array<number>(n + 1).fill(0);
  const retPrefix = new Array<number>(n + 1).fill(0);
  const retSqPrefix = new Array<number>(n + 1).fill(0);
  for (let i = 0; i < n; i++) {
    logPrefix[i + 1] = logPrefix[i]! + logs[i]!;
    const r = i === 0 ? 0 : logs[i]! - logs[i - 1]!;
    retPrefix[i + 1] = retPrefix[i]! + r;
    retSqPrefix[i + 1] = retSqPrefix[i]! + r * r;
  }

  // Training rows: every t whose 240-minute target is inside the case.
  const rows: number[][] = [];
  const targets: number[] = [];
  for (let t = MAX_LAG; t + HORIZON <= n - 1; t++) {
    rows.push(featuresAt(logs, logPrefix, retPrefix, retSqPrefix, t));
    targets.push(logs[t + HORIZON]! - logs[t]!);
  }
  const m = rows.length;
  if (m < 100) return price5(last);

  // Standardise features and target (mean 0, std 1) so the ridge penalty is
  // scale-free; keep the target's raw std for the final clip.
  const fMean = new Array<number>(N_FEATURES).fill(0);
  const fStd = new Array<number>(N_FEATURES).fill(0);
  for (let j = 0; j < N_FEATURES; j++) {
    let s = 0;
    for (let i = 0; i < m; i++) s += rows[i]![j]!;
    fMean[j] = s / m;
    let v = 0;
    for (let i = 0; i < m; i++) {
      const d = rows[i]![j]! - fMean[j]!;
      v += d * d;
    }
    fStd[j] = Math.sqrt(v / m) || 1;
  }
  let yMean = 0;
  for (const y of targets) yMean += y;
  yMean /= m;
  let yVar = 0;
  for (const y of targets) yVar += (y - yMean) * (y - yMean);
  const yStd = Math.sqrt(yVar / m) || 1e-9;

  // Normal equations on standardised data: (X'X + lambda I) beta = X'y.
  const a: number[][] = Array.from({ length: N_FEATURES }, () =>
    new Array<number>(N_FEATURES).fill(0),
  );
  const b = new Array<number>(N_FEATURES).fill(0);
  for (let i = 0; i < m; i++) {
    const row = rows[i]!;
    const y = (targets[i]! - yMean) / yStd;
    for (let j = 0; j < N_FEATURES; j++) {
      const xj = (row[j]! - fMean[j]!) / fStd[j]!;
      b[j]! += xj * y;
      for (let k = j; k < N_FEATURES; k++) {
        const xk = (row[k]! - fMean[k]!) / fStd[k]!;
        a[j]![k]! += xj * xk;
      }
    }
  }
  for (let j = 0; j < N_FEATURES; j++) {
    for (let k = j + 1; k < N_FEATURES; k++) a[k]![j] = a[j]![k]!;
    a[j]![j]! += LAMBDA;
  }
  const beta = solveLinear(a, b);

  // Predict at the checkpoint: standardise, dot, unstandardise, clip, shrink.
  const now = featuresAt(logs, logPrefix, retPrefix, retSqPrefix, n - 1);
  let pred = 0;
  for (let j = 0; j < N_FEATURES; j++) {
    pred += beta[j]! * ((now[j]! - fMean[j]!) / fStd[j]!);
  }
  pred = pred * yStd + yMean;
  const cap = CLIP_SIGMA * yStd;
  pred = Math.max(-cap, Math.min(cap, pred)) * SHRINK;
  return price5(last * Math.exp(pred));
}

/** The persistence guess of one market: its last close. */
function persistence(prices: number[]): number {
  const x = prices[prices.length - 1]!;
  return price5(typeof x === 'number' && Number.isFinite(x) ? x : 1);
}

/** Best-effort recovery of a case id and last price from unparseable text. */
function looseExtract(text: string): { id: string; last: number } | null {
  const idMatch = /"case"\s*:\s*"([^"]+)"/.exec(text);
  const pricesMatch = /"prices"\s*:\s*\[([\s\S]*?)\]/.exec(text);
  if (!idMatch || !pricesMatch) return null;
  const nums = pricesMatch[1]!.match(/-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g);
  if (!nums || nums.length === 0) return null;
  const last = Number(nums[nums.length - 1]);
  if (!Number.isFinite(last) || last <= 0) return null;
  return { id: idMatch[1]!, last };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const raw = readFileSync(join(INPUT_DIR, file), 'utf8');
  try {
    const c = JSON.parse(raw) as CaseFile;
    if (!Array.isArray(c.markets) || c.markets.length !== 1) {
      throw new Error(`${file} does not hold exactly one market`);
    }
    const market = c.markets[0]!;
    let price: number;
    try {
      price = forecastPrice(market.prices);
    } catch {
      price = persistence(market.prices);
    }
    writeFileSync(
      join(OUTPUT_DIR, `${id}.out`),
      `${JSON.stringify({ forecasts: [{ case: id, price }] })}\n`,
    );
  } catch {
    // Unparseable case file: still try to emit a valid persistence output.
    const loose = looseExtract(raw);
    if (loose) {
      writeFileSync(
        join(OUTPUT_DIR, `${id}.out`),
        `${JSON.stringify({ forecasts: [{ case: loose.id, price: price5(loose.last) }] })}\n`,
      );
    }
  }
}
