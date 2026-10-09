/// <reference lib="webworker" />
import { Adam, loss, accuracy, type Network, type Sample } from "./neural";
let net: Network,
  data: Sample[],
  optimizer: Adam,
  epoch = 0,
  rate = 0.03,
  revision = 0,
  running = false,
  timer: ReturnType<typeof setTimeout> | undefined;
function emit(error?: string) {
  postMessage({
    revision,
    net,
    epoch,
    loss: loss(net, data),
    accuracy: accuracy(net, data),
    running,
    error,
  });
}
function tick() {
  if (!running) return;
  try {
    const until = performance.now() + 22;
    let count = 0;
    do {
      optimizer.step(net, data, rate);
      epoch++;
      count++;
    } while (performance.now() < until && count < 12 && epoch < 20000);
    if (epoch >= 20000) running = false;
    emit();
    if (running) timer = setTimeout(tick, 25);
  } catch (e) {
    running = false;
    emit(String(e));
  }
}
onmessage = (e) => {
  const m = e.data;
  if (m.type === "init") {
    revision = m.revision;
    clearTimeout(timer);
    net = m.net;
    data = m.data;
    rate = m.rate;
    optimizer = new Adam(net);
    epoch = 0;
    running = false;
    emit();
  }
  if (m.type === "start" && !running) {
    revision = m.revision;
    rate = m.rate;
    running = true;
    tick();
  }
  if (m.type === "pause") {
    revision = m.revision;
    running = false;
    clearTimeout(timer);
    emit();
  }
  if (m.type === "rate") {
    rate = m.rate;
  }
};
