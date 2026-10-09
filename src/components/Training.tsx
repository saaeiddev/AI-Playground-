import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause, RotateCcw, Activity, CheckCircle2 } from "lucide-react";
import {
  createNetwork,
  loss,
  predict,
  accuracy,
  parameterCount,
  type Network,
  type Activation,
} from "../engine/neural";
import {
  DATASETS,
  dataset,
  descriptions,
  type Dataset,
} from "../engine/datasets";
import Scene from "./Scene";
import Architecture from "./Architecture";
import Inspector from "./Inspector";
import { ActivationSelect, NumberControl, Legend, fmt } from "./Controls";
import { LossChart, DecisionBoundary, type LossPoint } from "./Charts";
import type { Selection } from "../graphics/NeuralScene";
export default function Training({ reduced }: { reduced: boolean }) {
  const [name, setName] = useState<Dataset>("XOR"),
    [sizes, setSizes] = useState([2, 4, 4, 1]),
    [activation, setActivation] = useState<Activation>("tanh"),
    [rate, setRate] = useState(0.03),
    [seed, setSeed] = useState(42),
    [net, setNet] = useState(() => createNetwork()),
    [running, setRunning] = useState(false),
    [pausing, setPausing] = useState(false),
    [epoch, setEpoch] = useState(0),
    [metric, setMetric] = useState({ loss: 0, accuracy: 0 }),
    [history, setHistory] = useState<LossPoint[]>([]),
    [probe, setProbe] = useState([0, 1]),
    [selected, setSelected] = useState<Selection>({ layer: 1, node: 0 }),
    [error, setError] = useState("");
  const worker = useRef<Worker | null>(null),
    revision = useRef(0),
    data = useMemo(() => dataset(name), [name]);
  const init = (model: Network) => {
    setNet(model);
    setRunning(false);
    setPausing(false);
    setEpoch(0);
    setMetric({ loss: loss(model, data), accuracy: accuracy(model, data) });
    setHistory([{ epoch: 0, loss: loss(model, data) }]);
    setError("");
    worker.current?.postMessage({
      type: "init",
      revision: ++revision.current,
      net: model,
      data,
      rate,
    });
  };
  useEffect(() => {
    const w = new Worker(
      new URL("../engine/training.worker.ts", import.meta.url),
      { type: "module" },
    );
    worker.current = w;
    w.onmessage = (e) => {
      const m = e.data;
      if (m.revision !== revision.current || worker.current !== w) return;
      if (!m.running) setPausing(false);
      setNet(m.net);
      setEpoch(m.epoch);
      setMetric({ loss: m.loss, accuracy: m.accuracy });
      setRunning(m.running);
      if (m.error) setError(m.error);
      setHistory((h) => {
        const next =
          h.at(-1)?.epoch === m.epoch
            ? [...h.slice(0, -1), { epoch: m.epoch, loss: m.loss }]
            : [...h, { epoch: m.epoch, loss: m.loss }];
        return next.length > 400
          ? [next[0], ...next.slice(1).filter((_, i) => i % 2 === 0)]
          : next;
      });
    };
    w.onerror = () => {
      setError("Training could not continue. Reset the model and try again.");
      setRunning(false);
    };
    init(createNetwork(sizes, activation, seed));
    setSelected({ layer: Math.min(1, sizes.length - 1), node: 0 });
    return () => {
      w.terminate();
      worker.current = null;
    };
  }, [sizes, activation, seed, data]);
  function toggle() {
    if (running) {
      setPausing(true);
      worker.current?.postMessage({
        type: "pause",
        revision: ++revision.current,
      });
    } else {
      worker.current?.postMessage({
        type: "start",
        revision: ++revision.current,
        rate,
      });
      setRunning(true);
    }
  }
  const choose = (n: Dataset) => {
    setName(n);
    setSizes(
      n === "Spiral"
        ? [2, 8, 8, 1]
        : n === "Circle"
          ? [2, 6, 6, 1]
          : [2, 4, 4, 1],
    );
    setProbe(n === "XOR" || n === "OR" || n === "AND" ? [0, 1] : [0.2, 0.3]);
  };
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">TRAINING STUDIO</span>
          <h1>Watch a network learn.</h1>
          <p>Real gradients. Real predictions. Every epoch, in view.</p>
        </div>
        <span className={`status-pill ${running ? "is-running" : ""}`}>
          <Activity size={14} />
          {pausing
            ? "Finishing current batch…"
            : running
              ? "Training in your browser"
              : epoch >= 20000
                ? "Epoch limit reached"
                : epoch
                  ? "Training paused"
                  : "Ready to train"}
        </span>
      </div>
      <div className="training-layout">
        <aside className="panel training-controls">
          <div className="panel-heading">
            <h3>Experiment settings</h3>
            <span className="tiny-badge">ADAM</span>
          </div>
          <label className="field-label">
            Dataset
            <select
              value={name}
              onChange={(e) => choose(e.target.value as Dataset)}
              disabled={running}
            >
              {DATASETS.map((d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
          </label>
          <p className="hint dataset-description">
            {descriptions[name]} {data.length} training samples.
          </p>
          <Architecture
            sizes={sizes}
            onChange={setSizes}
            training
            disabled={running}
          />
          <ActivationSelect
            value={activation}
            onChange={setActivation}
            training
            disabled={running}
          />
          <p className="hint">Output: sigmoid · Loss: binary cross-entropy</p>
          <NumberControl
            label="Learning rate"
            value={rate}
            min={0.001}
            max={0.2}
            step={0.001}
            onChange={(n) => {
              setRate(n);
              worker.current?.postMessage({ type: "rate", rate: n });
            }}
          />
          <div className="button-row training-actions">
            <button
              className="primary"
              onClick={toggle}
              disabled={epoch >= 20000 || pausing}
            >
              {running ? <Pause size={16} /> : <Play size={16} />}{" "}
              {pausing
                ? "Pausing…"
                : running
                  ? "Pause"
                  : epoch
                    ? "Resume training"
                    : "Start training"}
            </button>
            <button
              className="icon-button"
              title="Reset model parameters"
              aria-label="Reset model parameters"
              onClick={() => {
                worker.current?.postMessage({
                  type: "pause",
                  revision: ++revision.current,
                });
                setSeed((s) => s + 1);
              }}
            >
              <RotateCcw size={17} />
            </button>
          </div>
          <p className="hint">
            Reset samples new weights; pause preserves weights and optimizer
            state. Limit: 20,000 epochs.
          </p>
          {error && (
            <p role="alert" className="error">
              {error}
            </p>
          )}
        </aside>
        <div className="training-main">
          <div className="metrics">
            <div>
              <span>Epoch</span>
              <strong data-testid="epoch">{epoch.toLocaleString()}</strong>
            </div>
            <div>
              <span>Training loss</span>
              <strong data-testid="loss">
                {metric.loss > 0 && metric.loss < 0.0001
                  ? metric.loss.toExponential(2)
                  : metric.loss.toFixed(5)}
              </strong>
            </div>
            <div>
              <span>Training accuracy</span>
              <strong>
                {Math.round(metric.accuracy * 100)}
                <small>%</small>
              </strong>
            </div>
            <div>
              <span>Parameters</span>
              <strong>{parameterCount(net)}</strong>
            </div>
          </div>
          <section className="panel network-canvas">
            <div className="panel-heading">
              <h3>Learning network</h3>
              <span className="subtle">{sizes.join(" → ")}</span>
            </div>
            <Scene
              mode="network"
              net={net}
              inputs={probe}
              weights={[]}
              bias={0}
              activation={activation}
              reduced={reduced}
              running={running}
              selected={selected}
              onSelect={setSelected}
              onConnection={(l, j) => setSelected({ layer: l + 1, node: j })}
            />
            <Legend />
            <p className="hint visualization-note">
              Signals show the selected probe’s forward pass; parameters refresh
              after each batch of actual optimizer steps.
            </p>
          </section>
          <div className="training-charts">
            <section className="panel">
              <div className="panel-heading">
                <h3>Loss over time</h3>
                <span className="subtle">BCE · lower is better</span>
              </div>
              <LossChart points={history} />
              <div className="prediction-summary">
                <CheckCircle2 size={16} />
                <span>
                  Probe ({fmt(probe[0], 2)}, {fmt(probe[1], 2)})
                </span>
                <strong>P(class 1) = {fmt(predict(net, probe)[0])}</strong>
              </div>
              <div className="probe-controls">
                <NumberControl
                  label="Probe x₁"
                  value={probe[0]}
                  onChange={(v) => setProbe([v, probe[1]])}
                  min={-1.2}
                  max={1.3}
                />
                <NumberControl
                  label="Probe x₂"
                  value={probe[1]}
                  onChange={(v) => setProbe([probe[0], v])}
                  min={-1.2}
                  max={1.3}
                />
              </div>
              <p className="hint">
                Accuracy is measured on the training set, not an unseen test
                set.
              </p>
            </section>
            <section className="panel boundary-panel">
              <div className="panel-heading">
                <h3>Decision boundary</h3>
                <span className="subtle">2D input space</span>
              </div>
              <DecisionBoundary
                net={net}
                data={data}
                name={name}
                onProbe={setProbe}
              />
            </section>
          </div>
        </div>
        <aside className="panel training-inspector">
          <Inspector
            net={net}
            inputs={probe}
            selected={selected}
            onSelect={setSelected}
            disabled={running}
            onChange={init}
          />
          <h4>Sample predictions</h4>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Inputs</th>
                  <th>Target</th>
                  <th>P(1)</th>
                </tr>
              </thead>
              <tbody>
                {data.slice(0, 8).map((s, i) => (
                  <tr key={i} onClick={() => setProbe(s.x)}>
                    <td>{s.x.map((x) => fmt(x, 2)).join(", ")}</td>
                    <td>{s.y[0]}</td>
                    <td>{fmt(predict(net, s.x)[0], 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">
            Editing a parameter starts a new training history and optimizer.
            Network animation is illustrative, not a timing trace.
          </p>
        </aside>
      </div>
    </>
  );
}
