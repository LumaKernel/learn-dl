import { Data } from "effect";

export class Vector extends Data.TaggedClass("Vector")<{
  readonly data: ReadonlyArray<number>;
}> {
  get length(): number {
    return this.data.length;
  }

  at(i: number): number {
    return this.data[i] ?? 0;
  }
}

export const zeros = (n: number): Vector =>
  new Vector({ data: Array.from({ length: n }, () => 0) });

export const ones = (n: number): Vector =>
  new Vector({ data: Array.from({ length: n }, () => 1) });

export const from = (arr: ReadonlyArray<number>): Vector =>
  new Vector({ data: arr });

export const add = (a: Vector, b: Vector): Vector =>
  new Vector({ data: a.data.map((v, i) => v + (b.data[i] ?? 0)) });

export const sub = (a: Vector, b: Vector): Vector =>
  new Vector({ data: a.data.map((v, i) => v - (b.data[i] ?? 0)) });

export const scale = (v: Vector, s: number): Vector =>
  new Vector({ data: v.data.map((x) => x * s) });

export const dot = (a: Vector, b: Vector): number =>
  a.data.reduce((sum, v, i) => sum + v * (b.data[i] ?? 0), 0);

export const hadamard = (a: Vector, b: Vector): Vector =>
  new Vector({ data: a.data.map((v, i) => v * (b.data[i] ?? 0)) });

export const map = (v: Vector, f: (x: number, i: number) => number): Vector =>
  new Vector({ data: v.data.map(f) });

export const sum = (v: Vector): number =>
  v.data.reduce((acc, x) => acc + x, 0);

export const max = (v: Vector): number =>
  v.data.reduce((acc, x) => Math.max(acc, x), -Infinity);

export const argmax = (v: Vector): number =>
  v.data.reduce((best, x, i, arr) => (x > (arr[best] ?? -Infinity) ? i : best), 0);
