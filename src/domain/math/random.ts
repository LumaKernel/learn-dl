import { Data } from "effect";
import { Vector } from "./vector";
import * as V from "./vector";
import { Matrix } from "./matrix";
import * as Mat from "./matrix";

/** Seedable PRNG state (xoshiro128**) */
export class Rng extends Data.TaggedClass("Rng")<{
  readonly state: readonly [number, number, number, number];
}> {}

const rotl = (x: number, k: number): number =>
  ((x << k) | (x >>> (32 - k))) >>> 0;

export const next = (rng: Rng): readonly [number, Rng] => {
  const [s0, s1, s2, s3] = rng.state;
  const result = (rotl((s1 * 5) >>> 0, 7) * 9) >>> 0;
  const t = (s1 << 9) >>> 0;
  const newS2 = (s2 ^ s0) >>> 0;
  const newS3 = (s3 ^ s1) >>> 0;
  const newS1 = (s1 ^ newS2) >>> 0;
  const newS0 = (s0 ^ newS3) >>> 0;
  const newS2f = (newS2 ^ t) >>> 0;
  const newS3f = rotl(newS3, 11);
  return [result, new Rng({ state: [newS0, newS1, newS2f, newS3f] })];
};

export const fromSeed = (seed: number): Rng => {
  const splitmix = (s: number): readonly [number, number] => {
    const z1 = (s + 0x9e3779b9) >>> 0;
    const z2 = ((z1 ^ (z1 >>> 16)) * 0x85ebca6b) >>> 0;
    const z3 = ((z2 ^ (z2 >>> 13)) * 0xc2b2ae35) >>> 0;
    return [(z3 ^ (z3 >>> 16)) >>> 0, z1];
  };
  const [a, s1] = splitmix(seed);
  const [b, s2] = splitmix(s1);
  const [c, s3] = splitmix(s2);
  const [d] = splitmix(s3);
  return new Rng({ state: [a, b, c, d] });
};

export const uniform01 = (rng: Rng): readonly [number, Rng] => {
  const [bits, nextRng] = next(rng);
  return [(bits >>> 0) / 0x100000000, nextRng];
};

export const normal = (rng: Rng, mean: number = 0, std: number = 1): readonly [number, Rng] => {
  const [u1, rng1] = uniform01(rng);
  const [u2, rng2] = uniform01(rng1);
  const z = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2);
  return [mean + std * z, rng2];
};

export const normalVector = (rng: Rng, n: number, mean: number = 0, std: number = 1): readonly [Vector, Rng] => {
  const result: number[] = [];
  let current = rng;
  for (let i = 0; i < n; i++) {
    const [v, nextRng] = normal(current, mean, std);
    result.push(v);
    current = nextRng;
  }
  return [V.from(result), current];
};

export const normalMatrix = (
  rng: Rng,
  rows: number,
  cols: number,
  mean: number = 0,
  std: number = 1,
): readonly [Matrix, Rng] => {
  const data: Vector[] = [];
  let current = rng;
  for (let i = 0; i < rows; i++) {
    const [row, nextRng] = normalVector(current, cols, mean, std);
    data.push(row);
    current = nextRng;
  }
  return [Mat.fromRows(data), current];
};

export const xavierStd = (fanIn: number, fanOut: number): number =>
  Math.sqrt(2 / (fanIn + fanOut));

export const heStd = (fanIn: number): number =>
  Math.sqrt(2 / fanIn);
