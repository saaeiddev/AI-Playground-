import { NeuralNet, makeDataset } from '../neural.mjs';
function check(condition, message) { if (!condition) throw Error(message); }
for (const kind of ['xor', 'circle', 'spiral']) {
  const data = makeDataset(kind, 123);
  check(data.length >= 200, 'Missing data for ' + kind);
  check(data.every(s => s.x.length === 2 && (s.y === 0 || s.y === 1)), 'Bad samples');
}
const net = new NeuralNet();
const data = makeDataset('xor');
const initial = net.evaluate(data);
for (let i = 0; i < 700; i++) {
  const batch = Array.from({ length: 32 }, (_, j) => data[(i * 32 + j) % data.length]);
  net.trainBatch(batch);
}
const result = net.evaluate(data);
check(result.accuracy > 0.97, 'XOR accuracy lower than expected: ' + result.accuracy);
check(result.loss < initial.loss / 10, 'XOR loss did not decrease enough');
for (const activation of ['sigmoid', 'tanh', 'relu']) {
  for (const optimizer of ['adam', 'sgd']) {
    const n = new NeuralNet({ activation, optimizer });
    n.trainBatch(data.slice(0, 24));
    check(Number.isFinite(n.predict([.5, -.5])), 'Non-finite output');
  }
}
console.log('PASS: dataset, XOR convergence, Adam/SGD and activation smoke tests');
