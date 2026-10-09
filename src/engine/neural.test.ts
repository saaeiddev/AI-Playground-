import { describe, it, expect } from "vitest";
import {
  activate,
  derivative,
  neuron,
  createNetwork,
  forward,
  gradients,
  loss,
  Adam,
  predict,
  accuracy,
  setBias,
  setWeight,
  parameterCount,
  type Activation,
} from "./neural";
import { DATASETS, dataset } from "./datasets";
describe("single neuron", () => {
  it("multiplies each input by its weight and adds bias once", () => {
    expect(neuron([2, -3], [0.5, -2], 1, "linear")).toEqual({
      terms: [1, 6],
      sum: 7,
      z: 8,
      y: 8,
    });
  });
  it("evaluates the initial displayed example exactly", () => {
    const n = neuron([0.8, 0.4, 0.6], [0.7, -0.5, 0.9], -0.2, "sigmoid");
    expect(n.z).toBeCloseTo(0.7, 12);
    expect(n.y).toBeCloseTo(0.6681877721681662, 12);
  });
  it.each<[Activation, number, number]>([
    ["sigmoid", 0, 0.5],
    ["sigmoid", -1000, 0],
    ["sigmoid", 1000, 1],
    ["relu", -2, 0],
    ["relu", 3, 3],
    ["leakyRelu", -2, -0.02],
    ["leakyRelu", 2, 2],
    ["tanh", 0, 0],
    ["tanh", 1, Math.tanh(1)],
    ["linear", -3, -3],
    ["step", -0.1, 0],
    ["step", 0, 1],
    ["step", 0.1, 1],
  ])("%s(%s) = %s", (kind, x, y) =>
    expect(activate(x, kind)).toBeCloseTo(y, 12),
  );
  it("rejects invalid values and mismatched inputs", () => {
    expect(() => neuron([], [], 0, "relu")).toThrow();
    expect(() => neuron([1], [1, 2], 0, "relu")).toThrow();
    expect(() => activate(NaN, "sigmoid")).toThrow();
    expect(() => neuron([1], [Infinity], 0, "linear")).toThrow();
  });
  it.each<Activation>(["sigmoid", "relu", "leakyRelu", "tanh", "linear"])(
    "checks %s derivative against central differences",
    (kind) => {
      for (const z of [-1.7, 0.6, 2]) {
        const e = 1e-6;
        expect(derivative(z, kind)).toBeCloseTo(
          (activate(z + e, kind) - activate(z - e, kind)) / (2 * e),
          7,
        );
      }
    },
  );
  it("does not pretend step is differentiable", () =>
    expect(() => derivative(0, "step")).toThrow());
});
describe("feedforward network", () => {
  it("matches a manual multilayer computation", () => {
    const n = createNetwork([2, 2, 1], "linear", 42, "linear");
    n.weights = [
      [
        [1, 2],
        [-1, 0.5],
      ],
      [[3, -2]],
    ];
    n.biases = [[0.5, -0.5], [1]];
    const t = forward(n, [2, 4]);
    expect(t.z).toEqual([[10.5, -0.5], [33.5]]);
    expect(t.a).toEqual([[2, 4], [10.5, -0.5], [33.5]]);
  });
  it("supports multiple outputs and architecture changes", () => {
    for (const sizes of [
      [1, 1],
      [2, 4, 3],
      [3, 8, 5, 2],
      [2, 2, 2, 2, 2, 1],
    ]) {
      const n = createNetwork(sizes);
      const p = predict(n, Array(sizes[0]).fill(0.4));
      expect(p.length).toBe(sizes.at(-1));
      expect(p.every(Number.isFinite)).toBe(true);
      expect(parameterCount(n)).toBe(
        sizes.slice(1).reduce((s, v, i) => s + v * (sizes[i] + 1), 0),
      );
    }
  });
  it("validates sizes and input dimensions", () => {
    expect(() => createNetwork([2, 0, 1])).toThrow();
    expect(() => createNetwork([2, 1.5, 1])).toThrow();
    expect(() => createNetwork([9, 2])).toThrow();
    expect(() => predict(createNetwork(), [1])).toThrow();
  });
  it("parameter editing immediately changes the true prediction", () => {
    const n = createNetwork([2, 1]);
    setWeight(n, 0, 0, 0, 2);
    setWeight(n, 0, 0, 1, -1);
    setBias(n, 0, 0, 0.5);
    expect(predict(n, [1, 1])[0]).toBeCloseTo(activate(1.5, "sigmoid"), 12);
    expect(() => setBias(n, 0, 0, NaN)).toThrow();
    expect(() => setWeight(n, 5, 0, 0, 1)).toThrow();
  });
  it("has deterministic, independently allocated initialization", () => {
    const a = createNetwork(),
      b = createNetwork();
    expect(a).toEqual(b);
    a.weights[0][0][0]++;
    expect(a.weights).not.toEqual(b.weights);
    expect(createNetwork([2, 4, 4, 1], "tanh", 43).weights).not.toEqual(
      b.weights,
    );
  });
});
describe("backpropagation and cross entropy", () => {
  it.each<Activation>(["sigmoid", "tanh", "relu", "leakyRelu", "linear"])(
    "checks every %s weight and bias gradient against finite differences",
    (kind) => {
      const n = createNetwork([2, 3, 2], kind, 31),
        data = [
          { x: [0.35, 0.7], y: [1, 0] },
          { x: [0.8, -0.3], y: [0, 1] },
        ],
        g = gradients(n, data),
        e = 1e-5;
      n.weights.forEach((layer, l) =>
        layer.forEach((row, j) =>
          row.forEach((w, i) => {
            n.weights[l][j][i] = w + e;
            const a = loss(n, data);
            n.weights[l][j][i] = w - e;
            const b = loss(n, data);
            n.weights[l][j][i] = w;
            expect(g.weights[l][j][i]).toBeCloseTo((a - b) / (2 * e), 6);
          }),
        ),
      );
      n.biases.forEach((row, l) =>
        row.forEach((b, j) => {
          n.biases[l][j] = b + e;
          const a = loss(n, data);
          n.biases[l][j] = b - e;
          const c = loss(n, data);
          n.biases[l][j] = b;
          expect(g.biases[l][j]).toBeCloseTo((a - c) / (2 * e), 6);
        }),
      );
    },
  );
  it("keeps BCE finite for extreme logits and matches known BCE", () => {
    const n = createNetwork([1, 1]);
    n.weights = [[[0]]];
    n.biases = [[0]];
    expect(loss(n, [{ x: [1], y: [1] }])).toBeCloseTo(Math.log(2), 12);
    n.biases = [[1000]];
    expect(loss(n, [{ x: [1], y: [0] }])).toBe(1000);
    expect(loss(n, [{ x: [1], y: [1] }])).toBe(0);
  });
  it("rejects step training and invalid datasets", () => {
    expect(() =>
      gradients(createNetwork([2, 2, 1], "step"), dataset("XOR")),
    ).toThrow();
    expect(() => loss(createNetwork(), [])).toThrow();
    expect(() => loss(createNetwork(), [{ x: [1, 2], y: [2] }])).toThrow();
    expect(() =>
      loss(createNetwork([2, 1], "tanh", 42, "linear"), dataset("XOR")),
    ).toThrow();
  });
});
describe("real training", () => {
  it.each(["AND", "OR", "XOR"] as const)(
    "learns %s, reducing loss and predicting every truth-table row",
    (name) => {
      const n = createNetwork(),
        d = dataset(name),
        before = loss(n, d),
        adam = new Adam(n);
      for (let i = 0; i < 650; i++) adam.step(n, d, 0.03);
      expect(loss(n, d)).toBeLessThan(before * 0.05);
      expect(accuracy(n, d)).toBe(1);
    },
  );
  it.each(["Linear", "Circle", "Spiral"] as const)(
    "learns the %s decision boundary",
    (name) => {
      const n = createNetwork([2, 8, 8, 1]),
        d = dataset(name),
        before = loss(n, d),
        adam = new Adam(n);
      for (let i = 0; i < 1200; i++) adam.step(n, d, 0.03);
      expect(loss(n, d)).toBeLessThan(before * 0.3);
      expect(accuracy(n, d)).toBeGreaterThan(0.9);
    },
    15000,
  );
  it("pause without steps preserves weights; resuming matches uninterrupted training", () => {
    const a = createNetwork(),
      b = createNetwork(),
      oa = new Adam(a),
      ob = new Adam(b),
      d = dataset("XOR");
    for (let i = 0; i < 25; i++) {
      oa.step(a, d, 0.02);
      ob.step(b, d, 0.02);
    }
    const snapshot = structuredClone(a);
    for (let i = 0; i < 5; i++) predict(a, [0, 1]);
    expect(a).toEqual(snapshot);
    for (let i = 0; i < 25; i++) {
      oa.step(a, d, 0.02);
      ob.step(b, d, 0.02);
    }
    expect(a).toEqual(b);
    expect(oa.steps).toBe(50);
    const reset = createNetwork();
    expect(new Adam(reset).steps).toBe(0);
    expect(reset.weights).not.toEqual(a.weights);
  });
  it("validates the learning rate", () => {
    const n = createNetwork();
    expect(() => new Adam(n).step(n, dataset("XOR"), 0)).toThrow();
    expect(() => new Adam(n).step(n, dataset("XOR"), NaN)).toThrow();
  });
});
describe("datasets", () => {
  it.each(DATASETS)("%s has finite inputs and both binary classes", (name) => {
    const d = dataset(name);
    expect(d.length).toBeGreaterThan(0);
    expect(
      d.every(
        (s) =>
          s.x.length === 2 && s.x.every(Number.isFinite) && s.y.length === 1,
      ),
    ).toBe(true);
    expect(new Set(d.map((s) => s.y[0]))).toEqual(new Set([0, 1]));
    expect(d).toEqual(dataset(name));
  });
});
