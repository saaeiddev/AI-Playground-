// AI Playground · small, dependency-free fully-connected neural network.
// Binary classification with real forward propagation, BCE backpropagation and Adam/SGD.
export function random(seed = 2026) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let n = t;
    n = Math.imul(n ^ (n >>> 15), n | 1);
    n ^= n + Math.imul(n ^ (n >>> 7), n | 61);
    return ((n ^ (n >>> 14)) >>> 0) / 4294967296;
  };
}
const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const act = (x, name) => name === "relu" ? Math.max(0, x) : name === "sigmoid" ? sigmoid(x) : Math.tanh(x);
const deriv = (z, a, name) => name === "relu" ? (z > 0 ? 1 : 0) : name === "sigmoid" ? a * (1 - a) : 1 - a * a;

export function makeDataset(kind = "xor", seed = 1024) {
  const r = random(seed), data = [];
  const gauss = () => Math.sqrt(-2 * Math.log(Math.max(1e-8, r()))) * Math.cos(2 * Math.PI * r());
  if (kind === "xor") {
    for (let i = 0; i < 200; i++) {
      const sx = i % 2 ? 1 : -1, sy = Math.floor(i / 2) % 2 ? 1 : -1;
      data.push({ x: [clamp(sx * .61 + gauss() * .18, -1, 1), clamp(sy * .61 + gauss() * .18, -1, 1)], y: sx !== sy ? 1 : 0 });
    }
  } else if (kind === "circle") {
    for (let i = 0; i < 240; i++) {
      const x = r() * 2 - 1, y = r() * 2 - 1;
      data.push({ x: [x, y], y: x * x + y * y < .33 ? 1 : 0 });
    }
  } else if (kind === "spiral") {
    for (let label = 0; label < 2; label++) {
      for (let i = 0; i < 120; i++) {
        const t = (i / 119) * 3.3 * Math.PI + label * Math.PI + gauss() * .14;
        const radius = .08 + .86 * (i / 119);
        data.push({ x: [radius * Math.cos(t), radius * Math.sin(t)], y: label });
      }
    }
  }
  // Reproducible Fisher–Yates shuffle.
  for (let i = data.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [data[i], data[j]] = [data[j], data[i]]; }
  return data;
}

export class NeuralNet {
  constructor({ hiddenLayers = 2, width = 6, activation = "tanh", optimizer = "adam", learningRate = .025, seed = 42 } = {}) {
    this.config = { hiddenLayers, width, activation, optimizer, learningRate, seed };
    this.sizes = [2, ...Array(hiddenLayers).fill(width), 1];
    this.weights = []; this.biases = [];
    this.mW = []; this.vW = []; this.mB = []; this.vB = []; this.t = 0;
    const r = random(seed);
    for (let l = 1; l < this.sizes.length; l++) {
      const nIn = this.sizes[l - 1], nOut = this.sizes[l];
      const scale = Math.sqrt((activation === "relu" ? 2 : 1.5) / nIn);
      this.weights.push(Array.from({ length: nOut }, () => Array.from({ length: nIn }, () => (r() * 2 - 1) * scale)));
      this.biases.push(Array(nOut).fill(0));
      this.mW.push(Array.from({ length: nOut }, () => Array(nIn).fill(0)));
      this.vW.push(Array.from({ length: nOut }, () => Array(nIn).fill(0)));
      this.mB.push(Array(nOut).fill(0));
      this.vB.push(Array(nOut).fill(0));
    }
  }
  forward(x) {
    const A = [x.slice()], Z = [];
    for (let l = 0; l < this.weights.length; l++) {
      const z = this.weights[l].map((row, j) => row.reduce((s, w, k) => s + w * A[l][k], this.biases[l][j]));
      const a = z.map(v => l === this.weights.length - 1 ? sigmoid(v) : act(v, this.config.activation));
      Z.push(z); A.push(a);
    }
    return { A, Z, probability: A.at(-1)[0] };
  }
  predict(x) { return this.forward(x).probability; }
  evaluate(data) {
    let loss = 0, correct = 0;
    for (const sample of data) {
      const p = clamp(this.predict(sample.x), 1e-7, 1 - 1e-7);
      loss -= sample.y * Math.log(p) + (1 - sample.y) * Math.log(1 - p);
      correct += (p >= .5 ? 1 : 0) === sample.y ? 1 : 0;
    }
    return { loss: loss / data.length, accuracy: correct / data.length };
  }
  trainBatch(data) {
    if (!data.length) return;
    const L = this.weights.length;
    const gW = this.weights.map(layer => layer.map(row => row.map(() => 0)));
    const gB = this.biases.map(layer => layer.map(() => 0));
    for (const sample of data) {
      const { A, Z } = this.forward(sample.x);
      const D = Array(L);
      D[L - 1] = [A.at(-1)[0] - sample.y]; // BCE + sigmoid: dL / dz = p - y
      for (let l = L - 2; l >= 0; l--) {
        D[l] = A[l + 1].map((a, j) => {
          let sum = 0;
          for (let k = 0; k < D[l + 1].length; k++) sum += this.weights[l + 1][k][j] * D[l + 1][k];
          return sum * deriv(Z[l][j], a, this.config.activation);
        });
      }
      for (let l = 0; l < L; l++)
        for (let j = 0; j < D[l].length; j++) {
          gB[l][j] += D[l][j];
          for (let k = 0; k < A[l].length; k++) gW[l][j][k] += D[l][j] * A[l][k];
        }
    }
    this.t++;
    const lr = this.config.learningRate, n = data.length, adam = this.config.optimizer === "adam";
    const b1 = .9, b2 = .999, eps = 1e-8;
    const update = (value, grad, m, v) => {
      const g = clamp(grad / n, -3, 3);
      if (!adam) return [value - lr * g, m, v];
      m = b1 * m + (1 - b1) * g;
      v = b2 * v + (1 - b2) * g * g;
      value -= lr * (m / (1 - Math.pow(b1, this.t))) / (Math.sqrt(v / (1 - Math.pow(b2, this.t))) + eps);
      return [value, m, v];
    };
    for (let l = 0; l < L; l++)
      for (let j = 0; j < this.weights[l].length; j++) {
        [this.biases[l][j], this.mB[l][j], this.vB[l][j]] = update(this.biases[l][j], gB[l][j], this.mB[l][j], this.vB[l][j]);
        for (let k = 0; k < this.weights[l][j].length; k++)
          [this.weights[l][j][k], this.mW[l][j][k], this.vW[l][j][k]] = update(this.weights[l][j][k], gW[l][j][k], this.mW[l][j][k], this.vW[l][j][k]);
      }
  }
}
