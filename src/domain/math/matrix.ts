import { Data } from "effect";
import { Vector } from "./vector";
import * as V from "./vector";

/** Row-major matrix: rows x cols */
export class Matrix extends Data.TaggedClass("Matrix")<{
  readonly rows: number;
  readonly cols: number;
  readonly data: ReadonlyArray<Vector>;
}> {
  at(i: number, j: number): number {
    return this.data[i]?.at(j) ?? 0;
  }

  row(i: number): Vector {
    return this.data[i] ?? V.zeros(this.cols);
  }

  col(j: number): Vector {
    return V.from(this.data.map((r) => r.at(j)));
  }
}

export const create = (rows: number, cols: number, fill: number = 0): Matrix =>
  new Matrix({
    rows,
    cols,
    data: Array.from({ length: rows }, () => V.from(Array.from({ length: cols }, () => fill))),
  });

export const zeros = (rows: number, cols: number): Matrix => create(rows, cols, 0);

export const fromRows = (data: ReadonlyArray<Vector>): Matrix =>
  new Matrix({
    rows: data.length,
    cols: data[0]?.length ?? 0,
    data,
  });

export const fromArrays = (data: ReadonlyArray<ReadonlyArray<number>>): Matrix =>
  fromRows(data.map(V.from));

export const transpose = (m: Matrix): Matrix =>
  fromRows(Array.from({ length: m.cols }, (_, j) => m.col(j)));

/** Matrix-vector multiply: (rows x cols) * (cols,) -> (rows,) */
export const mulVec = (m: Matrix, v: Vector): Vector =>
  V.from(m.data.map((r) => V.dot(r, v)));

/** Matrix-matrix multiply */
export const mul = (a: Matrix, b: Matrix): Matrix => {
  const bt = transpose(b);
  return fromRows(
    a.data.map((aRow) => V.from(bt.data.map((bCol) => V.dot(aRow, bCol)))),
  );
};

export const add = (a: Matrix, b: Matrix): Matrix =>
  fromRows(a.data.map((r, i) => V.add(r, b.data[i] ?? V.zeros(a.cols))));

export const sub = (a: Matrix, b: Matrix): Matrix =>
  fromRows(a.data.map((r, i) => V.sub(r, b.data[i] ?? V.zeros(a.cols))));

export const scale = (m: Matrix, s: number): Matrix =>
  fromRows(m.data.map((r) => V.scale(r, s)));

export const map = (m: Matrix, f: (x: number, i: number, j: number) => number): Matrix =>
  fromRows(m.data.map((r, i) => V.from(r.data.map((x, j) => f(x, i, j)))));

/** Outer product: (n,) x (m,) -> (n x m) */
export const outer = (a: Vector, b: Vector): Matrix =>
  fromRows(a.data.map((ai) => V.from(b.data.map((bj) => ai * bj))));
