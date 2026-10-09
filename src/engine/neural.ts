/** Small, auditable dense MLP. All arrays use [layer][destination][source]. */
export const ACTIVATIONS = [
  "sigmoid",
  "relu",
  "leakyRelu",
  "tanh",
  "linear",
  "step",
] as const;
export type Activation = (typeof ACTIVATIONS)[number];
export const activationNames: Record<Activation, string> = {
  sigmoid: "Sigmoid",
  relu: "ReLU",
  leakyRelu: "Leaky ReLU",
  tanh: "Tanh",
  linear: "Linear",
  step: "Step",
};
export const finite = (n: number) => {
  if (!Number.isFinite(n)) throw new Error("Use a finite number.");
  return n;
};
export function activate(z: number, kind: Activation): number {
  finite(z);
  switch (kind) {
    case "sigmoid":
      return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
    case "relu":
      return Math.max(0, z);
    case "leakyRelu":
      return z >= 0 ? z : 0.01 * z;
    case "tanh":
      return Math.tanh(z);
    case "linear":
      return z;
    case "step":
      return z >= 0 ? 1 : 0;
    default:
      throw new Error("Unknown activation.");
  }
}
export function derivative(z: number, kind: Activation): number {
  const a = activate(z, kind);
  switch (kind) {
    case "sigmoid":
      return a * (1 - a);
    case "relu":
      return z > 0 ? 1 : 0;
    case "leakyRelu":
      return z >= 0 ? 1 : 0.01;
    case "tanh":
      return 1 - a * a;
    case "linear":
      return 1;
    case "step":
      throw new Error(
        "Step is not differentiable at zero and has zero gradient elsewhere. Choose a differentiable activation for training.",
      );
  }
}
export function neuron(
  inputs: number[],
  weights: number[],
  bias: number,
  activation: Activation,
) {
  if (!inputs.length || inputs.length !== weights.length)
    throw new Error("Each input needs one weight.");
  const terms = inputs.map((x, i) => finite(x) * finite(weights[i]));
  const sum = terms.reduce((a, b) => a + b, 0),
    z = finite(sum + finite(bias));
  return { terms, sum, z, y: activate(z, activation) };
}
export function random(seed = 42) {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export interface Network {
  sizes: number[];
  weights: number[][][];
  biases: number[][];
  hiddenActivation: Activation;
  outputActivation: Activation;
  seed: number;
}
export interface Sample {
  x: number[];
  y: number[];
}
export interface Trace {
  a: number[][];
  z: number[][];
}
export interface Gradients {
  weights: number[][][];
  biases: number[][];
}
export function createNetwork(
  sizes = [2, 4, 4, 1],
  hiddenActivation: Activation = "tanh",
  seed = 42,
  outputActivation: Activation = "sigmoid",
): Network {
  if (
    sizes.length < 2 ||
    sizes.length > 6 ||
    sizes.some((n) => !Number.isInteger(n) || n < 1 || n > 8)
  )
    throw new Error("Use 2–6 layers with 1–8 neurons each.");
  const rng = random(seed);
  const weights = sizes
    .slice(1)
    .map((n, l) =>
      Array.from({ length: n }, () =>
        Array.from(
          { length: sizes[l] },
          () => (rng() * 2 - 1) * Math.sqrt(6 / (sizes[l] + n)),
        ),
      ),
    );
  return {
    sizes: [...sizes],
    weights,
    biases: sizes.slice(1).map((n) => Array(n).fill(0)),
    hiddenActivation,
    outputActivation,
    seed,
  };
}
export function forward(net: Network, input: number[]): Trace {
  if (input.length !== net.sizes[0])
    throw new Error("Input dimension does not match the network.");
  const a = [input.map(finite)],
    z: number[][] = [];
  for (let l = 0; l < net.weights.length; l++) {
    z.push(
      net.weights[l].map((w, j) =>
        finite(w.reduce((s, v, i) => s + v * a[l][i], net.biases[l][j])),
      ),
    );
    a.push(
      z[l].map((v) =>
        activate(
          v,
          l === net.weights.length - 1
            ? net.outputActivation
            : net.hiddenActivation,
        ),
      ),
    );
  }
  return { a, z };
}
export function predict(net: Network, input: number[]) {
  return forward(net, input).a.at(-1)!;
}
export function zeroGrad(net: Network): Gradients {
  return {
    weights: net.weights.map((layer) => layer.map((row) => row.map(() => 0))),
    biases: net.biases.map((row) => row.map(() => 0)),
  };
}
function validateData(net: Network, data: Sample[]) {
  if (!data.length) throw new Error("Dataset cannot be empty.");
  if (net.outputActivation !== "sigmoid")
    throw new Error("Binary cross-entropy training requires sigmoid outputs.");
  data.forEach((s) => {
    if (
      s.x.length !== net.sizes[0] ||
      s.y.length !== net.sizes.at(-1) ||
      s.y.some((t) => !Number.isFinite(t) || t < 0 || t > 1)
    )
      throw new Error("Invalid dataset dimensions or targets.");
  });
}
/** Stable binary cross-entropy from logits, averaged over samples and outputs. */
export function loss(net: Network, data: Sample[]): number {
  validateData(net, data);
  let total = 0;
  for (const s of data) {
    const z = forward(net, s.x).z.at(-1)!;
    z.forEach((v, i) => {
      total += Math.max(v, 0) - v * s.y[i] + Math.log1p(Math.exp(-Math.abs(v)));
    });
  }
  return total / (data.length * net.sizes.at(-1)!);
}
export function gradients(net: Network, data: Sample[]): Gradients {
  validateData(net, data);
  if (net.hiddenActivation === "step" && net.weights.length > 1)
    throw new Error("Step activation cannot be trained by backpropagation.");
  const g = zeroGrad(net),
    L = net.weights.length,
    den = data.length * net.sizes.at(-1)!;
  for (const s of data) {
    const t = forward(net, s.x);
    let delta = t.a[L].map((v, j) => (v - s.y[j]) / den);
    for (let l = L - 1; l >= 0; l--) {
      delta.forEach((d, j) => {
        g.biases[l][j] += d;
        t.a[l].forEach((v, i) => {
          g.weights[l][j][i] += d * v;
        });
      });
      if (l > 0)
        delta = t.a[l].map(
          (_, i) =>
            net.weights[l].reduce((sum, w, j) => sum + w[i] * delta[j], 0) *
            derivative(t.z[l - 1][i], net.hiddenActivation),
        );
    }
  }
  return g;
}
/** Adam: full-batch gradient optimization with bias-corrected moments. */
export class Adam {
  m: Gradients;
  v: Gradients;
  steps = 0;
  constructor(net: Network) {
    this.m = zeroGrad(net);
    this.v = zeroGrad(net);
  }
  step(net: Network, data: Sample[], rate: number) {
    if (!Number.isFinite(rate) || rate <= 0 || rate > 1)
      throw new Error("Learning rate must be in (0, 1].");
    const g = gradients(net, data);
    this.steps++;
    const update = (
      value: number,
      grad: number,
      m: number[],
      v: number[],
      i: number,
    ) => {
      m[i] = 0.9 * m[i] + 0.1 * grad;
      v[i] = 0.999 * v[i] + 0.001 * grad * grad;
      return finite(
        value -
          (rate * (m[i] / (1 - 0.9 ** this.steps))) /
            (Math.sqrt(v[i] / (1 - 0.999 ** this.steps)) + 1e-8),
      );
    };
    net.weights.forEach((layer, l) =>
      layer.forEach((row, j) =>
        row.forEach((w, i) => {
          net.weights[l][j][i] = update(
            w,
            g.weights[l][j][i],
            this.m.weights[l][j],
            this.v.weights[l][j],
            i,
          );
        }),
      ),
    );
    net.biases.forEach((row, l) =>
      row.forEach((b, j) => {
        net.biases[l][j] = update(
          b,
          g.biases[l][j],
          this.m.biases[l],
          this.v.biases[l],
          j,
        );
      }),
    );
  }
}
export function accuracy(net: Network, data: Sample[]) {
  return (
    data.filter((s) =>
      predict(net, s.x).every((p, i) => (p >= 0.5 ? 1 : 0) === s.y[i]),
    ).length / data.length
  );
}
export function parameterCount(net: Network) {
  return net.weights.reduce(
    (s, l) => s + l.reduce((n, w) => n + w.length + 1, 0),
    0,
  );
}
export function setWeight(
  net: Network,
  l: number,
  j: number,
  i: number,
  value: number,
) {
  finite(value);
  if (!net.weights[l]?.[j] || i < 0 || i >= net.weights[l][j].length)
    throw new Error("Unknown connection.");
  net.weights[l][j][i] = value;
}
export function setBias(net: Network, l: number, j: number, value: number) {
  finite(value);
  if (net.biases[l]?.[j] === undefined) throw new Error("Unknown neuron.");
  net.biases[l][j] = value;
}
