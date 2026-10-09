import { useEffect, useState, lazy, Suspense } from "react";
import {
  Atom,
  FlaskConical,
  Network as NetworkIcon,
  Activity,
  BookOpen,
  Play,
  Pause,
  RotateCcw,
  Plus,
  Minus,
  Info,
  SlidersHorizontal,
  Check,
  MousePointer2,
} from "lucide-react";
import Scene from "./components/Scene";
import {
  ActivationSelect,
  NumberControl,
  Legend,
  fmt,
} from "./components/Controls";
import Architecture from "./components/Architecture";
import Inspector from "./components/Inspector";
import Learning from "./components/Learning";
import { anatomy } from "./components/Biology";
import {
  neuron,
  createNetwork,
  activationNames,
  forward,
  parameterCount,
  type Activation,
  type Network,
} from "./engine/neural";
import type { Selection } from "./graphics/NeuralScene";
const Training = lazy(() => import("./components/Training"));
type Page = "explorer" | "lab" | "builder" | "training" | "learning";
const pages: [Page, string, typeof Atom][] = [
  ["explorer", "Neuron Explorer", Atom],
  ["lab", "Neuron Lab", FlaskConical],
  ["builder", "Network Builder", NetworkIcon],
  ["training", "Training Studio", Activity],
  ["learning", "Learning Center", BookOpen],
];
const operations = [
  "Receive inputs",
  "Multiply by weights",
  "Sum contributions",
  "Add bias",
  "Apply activation",
  "Read output",
];
function currentPage(): Page {
  const h = location.hash.slice(1);
  return pages.some((p) => p[0] === h) ? (h as Page) : "explorer";
}
export default function App() {
  const [page, setPage] = useState<Page>(currentPage),
    [reduced, setReduced] = useState(
      () => matchMedia("(prefers-reduced-motion: reduce)").matches,
    ),
    [inputs, setInputs] = useState([0.8, 0.4, 0.6]),
    [weights, setWeights] = useState([0.7, -0.5, 0.9]),
    [bias, setBias] = useState(-0.2),
    [activation, setActivation] = useState<Activation>("sigmoid"),
    [playing, setPlaying] = useState(true),
    [labels, setLabels] = useState(true),
    [bioPotential, setBioPotential] = useState(-50),
    [part, setPart] = useState("Dendrites"),
    [step, setStep] = useState(-1);
  const [net, setNet] = useState<Network>(() => createNetwork()),
    [networkInputs, setNetworkInputs] = useState([0.3, 0.7]),
    [selected, setSelected] = useState<Selection>({ layer: 1, node: 0 });
  const result = neuron(inputs, weights, bias, activation);
  useEffect(() => {
    const fn = () => setPage(currentPage());
    window.addEventListener("hashchange", fn);
    return () => window.removeEventListener("hashchange", fn);
  }, []);
  function navigate(p: Page) {
    location.hash = p;
    setPage(p);
    window.scrollTo({ top: 0, behavior: "instant" });
  }
  function reset() {
    setInputs([0.8, 0.4, 0.6]);
    setWeights([0.7, -0.5, 0.9]);
    setBias(-0.2);
    setActivation("sigmoid");
    setStep(-1);
    setPlaying(true);
  }
  const updateInput = (i: number, v: number) =>
    setInputs((a) => a.map((n, j) => (i === j ? v : n)));
  const updateWeight = (i: number, v: number) =>
    setWeights((a) => a.map((n, j) => (i === j ? v : n)));
  const singleProps = {
    mode: "single" as const,
    inputs,
    weights,
    bias,
    activation,
    reduced,
    running: playing,
    labels,
  };
  const changeArchitecture = (sizes: number[]) => {
    setNet(createNetwork(sizes, net.hiddenActivation, net.seed));
    setNetworkInputs(
      Array.from({ length: sizes[0] }, (_, i) => networkInputs[i] ?? 0.5),
    );
    setSelected({ layer: 1, node: 0 });
  };
  const trace = forward(net, networkInputs);
  return (
    <div className="app">
      <header className="site-header">
        <a
          className="brand"
          href="#explorer"
          onClick={() => navigate("explorer")}
        >
          <span className="brand-mark">
            Σ<span>ƒ</span>
          </span>
          <span>
            AI Playground<small>NEURAL NETWORK LABORATORY</small>
          </span>
        </a>
        <div className="header-right">
          <span className="header-caption">
            A space to understand intelligence.
          </span>
          <label className="motion-toggle">
            <input
              type="checkbox"
              checked={reduced}
              onChange={(e) => setReduced(e.target.checked)}
            />
            <span className="toggle-track" />
            Reduced motion
          </label>
        </div>
      </header>
      <nav className="main-nav" aria-label="Main navigation">
        {pages.map(([id, label, Icon], i) => (
          <button
            key={id}
            aria-current={page === id ? "page" : undefined}
            className={page === id ? "active" : ""}
            onClick={() => navigate(id)}
          >
            <Icon size={17} />
            {label}
            <span className="nav-index">0{i + 1}</span>
          </button>
        ))}
      </nav>
      <main>
        {page === "explorer" && (
          <>
            <div className="page-heading explorer-heading">
              <div>
                <span className="eyebrow">
                  <span className="eyebrow-line" />
                  THE ORIGIN OF INTELLIGENCE
                </span>
                <h1>
                  Different by nature.
                  <br />
                  <em>Connected by an idea.</em>
                </h1>
                <p>
                  Explore the living inspiration. Experiment with the
                  mathematical abstraction.
                </p>
              </div>
              <div className="explorer-actions">
                <button className="primary" onClick={() => navigate("lab")}>
                  <FlaskConical size={17} />
                  Experiment with a neuron
                </button>
                <span>
                  <MousePointer2 size={13} />
                  Real 3D. Yours to explore.
                </span>
              </div>
            </div>
            <div className="comparison">
              <section className="neuron-card biological-card">
                <div className="card-title">
                  <div>
                    <span className="eyebrow amber">01 / BIOLOGICAL</span>
                    <h2>Nature’s signal processor</h2>
                  </div>
                  <span className="specimen-badge amber">
                    <Atom size={15} />
                    Living system
                  </span>
                </div>
                <Scene
                  mode="bio"
                  inputs={[]}
                  weights={[]}
                  bias={0}
                  activation="linear"
                  reduced={reduced}
                  running={playing}
                  labels={labels}
                  bioFires={bioPotential >= -55}
                  onPart={setPart}
                />
                <div className="bio-part">
                  <span className="part-dot" />
                  <strong>{part}</strong>
                  <p>{anatomy[part]}</p>
                </div>
                <div className="card-bottom">
                  <span className="formula bio-formula">
                    Receive · Integrate · Transmit
                  </span>
                  <button
                    className="text-button"
                    onClick={() => navigate("learning")}
                  >
                    Explore the biology
                  </button>
                </div>
              </section>
              <section className="neuron-card artificial-card">
                <div className="card-title">
                  <div>
                    <span className="eyebrow">02 / ARTIFICIAL</span>
                    <h2>A little math. A new possibility.</h2>
                  </div>
                  <span className="specimen-badge">
                    <FlaskConical size={15} />
                    Mathematical model
                  </span>
                </div>
                <Scene {...singleProps} />
                <div className="art-equation">
                  <span className="formula">z = Σ wᵢxᵢ + b</span>
                  <span className="equation-divider" />
                  <span className="formula">y = f(z)</span>
                  <span className="live-output">
                    y <strong>{fmt(result.y)}</strong>
                  </span>
                </div>
                <div className="card-bottom">
                  <span className="small-caption">
                    Inputs become a weighted, activated output.
                  </span>
                  <button
                    className="text-button"
                    onClick={() => navigate("lab")}
                  >
                    Open neuron lab
                  </button>
                </div>
              </section>
            </div>
            <div className="explorer-control-bar">
              <div className="button-row">
                <button
                  className="small-button"
                  onClick={() => setPlaying((v) => !v)}
                >
                  {playing ? <Pause size={14} /> : <Play size={14} />}{" "}
                  {playing ? "Pause signals" : "Play signals"}
                </button>
                <button
                  className={`small-button ${labels ? "selected" : ""}`}
                  onClick={() => setLabels((v) => !v)}
                >
                  <Check size={14} />
                  Labels {labels ? "on" : "off"}
                </button>
              </div>
              <NumberControl
                label="Illustrative membrane potential (mV)"
                value={bioPotential}
                min={-80}
                max={-40}
                step={1}
                onChange={setBioPotential}
              />
              <span
                className={`threshold ${bioPotential >= -55 ? "above" : ""}`}
              >
                {bioPotential >= -55
                  ? "Threshold reached · axon signal"
                  : "Below threshold · local inputs only"}
              </span>
            </div>
            <div className="science-note">
              <Info size={17} />
              <p>
                Inspired, not identical. Biological neurons use electrochemical
                signals; artificial neurons operate on numbers. Shapes, signal
                speed, and the −55 mV threshold are educational simplifications.
              </p>
            </div>
            <div className="explorer-next">
              <span>YOUR NEXT EXPERIMENT</span>
              <button onClick={() => navigate("builder")}>
                <NetworkIcon size={19} />
                <strong>Connect neurons into a network</strong>
                <span>Build layer by layer</span>
              </button>
              <button onClick={() => navigate("training")}>
                <Activity size={19} />
                <strong>Teach a network to solve XOR</strong>
                <span>Watch real learning happen</span>
              </button>
            </div>
          </>
        )}
        {page === "lab" && (
          <>
            <div className="page-heading">
              <div>
                <span className="eyebrow">ARTIFICIAL NEURON LAB</span>
                <h1>Make the math visible.</h1>
                <p>Change a number. Follow its contribution. See the result.</p>
              </div>
              <button onClick={reset}>
                <RotateCcw size={16} />
                Reset neuron
              </button>
            </div>
            <div className="lab-layout">
              <div className="lab-main">
                <section className="panel single-lab">
                  <div className="panel-heading">
                    <h3>The glass neuron</h3>
                    <span className="status-pill">
                      {inputs.length} inputs · {activationNames[activation]}
                    </span>
                  </div>
                  <Scene {...singleProps} step={step} />
                  <Legend />
                  <div className="equation-results">
                    <div>
                      <span>Weighted inputs</span>
                      <strong>Σ wᵢxᵢ = {fmt(result.sum)}</strong>
                    </div>
                    <div>
                      <span>Bias included</span>
                      <strong>z = {fmt(result.z)}</strong>
                    </div>
                    <div className="output-result">
                      <span>Activated output</span>
                      <strong data-testid="neuron-output">
                        y = {fmt(result.y)}
                      </strong>
                    </div>
                  </div>
                </section>
                <section className="panel computation">
                  <div className="panel-heading">
                    <h3>Inside the calculation</h3>
                    <button
                      className="text-button"
                      onClick={() => {
                        setStep((s) => (s + 1) % 6);
                        setPlaying(true);
                      }}
                    >
                      <Play size={14} />
                      Next step
                    </button>
                  </div>
                  <div className="computation-steps">
                    {operations.map((op, i) => (
                      <button
                        key={op}
                        className={step === i ? "active" : ""}
                        onClick={() => {
                          setStep(i);
                          setPlaying(true);
                        }}
                      >
                        <span>{i + 1}</span>
                        {op}
                      </button>
                    ))}
                  </div>
                  <div className="calculation-detail">
                    <span className="formula">
                      z ={" "}
                      {result.terms.map((v) => `(${fmt(v, 3)})`).join(" + ")} +
                      ({fmt(bias)}) = <b>{fmt(result.z)}</b>
                    </span>
                    <span className="formula">
                      y = {activationNames[activation]}({fmt(result.z)}) ={" "}
                      <b>{fmt(result.y)}</b>
                    </span>
                  </div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Input</th>
                          <th>xᵢ</th>
                          <th>wᵢ</th>
                          <th>wᵢ × xᵢ</th>
                        </tr>
                      </thead>
                      <tbody>
                        {inputs.map((x, i) => (
                          <tr key={i}>
                            <td>x{i + 1}</td>
                            <td>{fmt(x)}</td>
                            <td>{fmt(weights[i])}</td>
                            <td>{fmt(result.terms[i])}</td>
                          </tr>
                        ))}
                        <tr>
                          <td>Bias x₀</td>
                          <td>1</td>
                          <td>{fmt(bias)}</td>
                          <td>{fmt(bias)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <p className="hint">
                    {step >= 0 ? operations[step] + ": " : ""}
                    {step === 0
                      ? "The input values are numbers, not electrical impulses."
                      : step === 1
                        ? "Each input is multiplied by its own weight."
                        : step === 2
                          ? "The weighted products are added together."
                          : step === 3
                            ? "Bias shifts the sum. The x₀ = 1 path adds it exactly once."
                            : step === 4
                              ? "The selected activation maps z to the output value."
                              : step === 5
                                ? "The output can become an input to another layer."
                                : "All values recalculate immediately. Select a step to follow the process."}
                  </p>
                </section>
              </div>
              <aside className="panel lab-controls">
                <div className="panel-heading">
                  <h3>
                    <SlidersHorizontal size={16} />
                    Neuron parameters
                  </h3>
                  <span className="tiny-badge">LIVE</span>
                </div>
                <div className="input-list">
                  {inputs.map((x, i) => (
                    <div className="input-editor" key={i}>
                      <div className="input-title">
                        <span className="input-tag">x{i + 1}</span>
                        <strong>Input {i + 1}</strong>
                        <button
                          aria-label={`Remove input ${i + 1}`}
                          disabled={inputs.length === 1}
                          onClick={() => {
                            setInputs((a) => a.filter((_, j) => j !== i));
                            setWeights((a) => a.filter((_, j) => j !== i));
                          }}
                        >
                          <Minus size={14} />
                        </button>
                      </div>
                      <NumberControl
                        label={`Input ${i + 1} value`}
                        value={x}
                        onChange={(v) => updateInput(i, v)}
                      />
                      <NumberControl
                        label={`Weight ${i + 1}`}
                        value={weights[i]}
                        onChange={(v) => updateWeight(i, v)}
                      />
                      <span className="contribution">
                        Contribution <b>{fmt(result.terms[i])}</b>
                      </span>
                    </div>
                  ))}
                </div>
                <button
                  className="add-input"
                  disabled={inputs.length >= 6}
                  onClick={() => {
                    setInputs((a) => [...a, 0.5]);
                    setWeights((a) => [...a, 1]);
                  }}
                >
                  <Plus size={15} />
                  Add input {inputs.length}/6
                </button>
                <div className="bias-control">
                  <NumberControl
                    label="Bias b"
                    value={bias}
                    onChange={setBias}
                  />
                </div>
                <ActivationSelect value={activation} onChange={setActivation} />
                <button
                  className="primary full-width"
                  onClick={() => {
                    setStep(-1);
                    setPlaying((p) => !p);
                  }}
                >
                  {playing ? <Pause size={16} /> : <Play size={16} />}{" "}
                  {playing ? "Pause signal flow" : "Run signal flow"}
                </button>
                <p className="hint">
                  Cyan = positive weight. Violet = negative weight. Particle
                  size and speed reflect |wᵢxᵢ|. Zero contributions have no
                  moving particle.
                </p>
              </aside>
            </div>
          </>
        )}
        {page === "builder" && (
          <>
            <div className="page-heading">
              <div>
                <span className="eyebrow">NETWORK BUILDER</span>
                <h1>One neuron is only the beginning.</h1>
                <p>
                  Build layers, inspect connections, and follow a real forward
                  pass.
                </p>
              </div>
              <button
                onClick={() => {
                  setNet(
                    createNetwork(
                      net.sizes,
                      net.hiddenActivation,
                      net.seed + 1,
                    ),
                  );
                }}
              >
                <RotateCcw size={16} />
                Reset parameters
              </button>
            </div>
            <div className="builder-layout">
              <aside className="panel">
                <Architecture sizes={net.sizes} onChange={changeArchitecture} />
                <ActivationSelect
                  value={net.hiddenActivation}
                  onChange={(a) => setNet({ ...net, hiddenActivation: a })}
                />
                <p className="hint">
                  Hidden layers use this activation. Output neurons use sigmoid.
                </p>
                <h4>Network inputs</h4>
                {networkInputs.map((x, i) => (
                  <NumberControl
                    key={i}
                    label={`Network input ${i + 1}`}
                    value={x}
                    onChange={(v) =>
                      setNetworkInputs((a) =>
                        a.map((n, j) => (i === j ? v : n)),
                      )
                    }
                  />
                ))}
                <div className="builder-stats">
                  <span>{net.sizes.reduce((a, b) => a + b, 0)} neurons</span>
                  <span>{parameterCount(net)} parameters</span>
                </div>
              </aside>
              <section className="panel builder-scene">
                <div className="panel-heading">
                  <h3>Your network</h3>
                  <span className="subtle">{net.sizes.join(" → ")}</span>
                </div>
                <Scene
                  mode="network"
                  net={net}
                  inputs={networkInputs}
                  weights={[]}
                  bias={0}
                  activation={net.hiddenActivation}
                  reduced={reduced}
                  running={playing}
                  selected={selected}
                  onSelect={setSelected}
                  onConnection={(l, j) =>
                    setSelected({ layer: l + 1, node: j })
                  }
                />
                <Legend />
                <div className="network-output">
                  {trace.a.at(-1)!.map((y, i) => (
                    <div key={i}>
                      <span>Output {i + 1}</span>
                      <strong>{fmt(y)}</strong>
                    </div>
                  ))}
                </div>
                <div className="builder-bottom">
                  <button
                    className="small-button"
                    onClick={() => setPlaying((p) => !p)}
                  >
                    {playing ? <Pause size={14} /> : <Play size={14} />}Signal
                    flow
                  </button>
                  <span>
                    Drag to orbit · right-drag to pan · scroll to zoom
                  </span>
                </div>
              </section>
              <aside className="panel">
                <Inspector
                  net={net}
                  inputs={networkInputs}
                  selected={selected}
                  onSelect={setSelected}
                  onChange={setNet}
                />
              </aside>
            </div>
          </>
        )}
        {page === "training" && (
          <Suspense
            fallback={
              <div className="panel loading-panel" role="status">
                Preparing training studio…
              </div>
            }
          >
            <Training reduced={reduced} />
          </Suspense>
        )}
        {page === "learning" && <Learning reduced={reduced} />}
      </main>
      <footer>
        <a className="footer-brand" href="#explorer">
          AI Playground<span>Curiosity, made interactive.</span>
        </a>
        <span>Created By Amir Saeid Dehghan</span>
        <a
          href="https://github.com/saaeiddev/AI-Playground-"
          target="_blank"
          rel="noreferrer"
        >
          Source & documentation
        </a>
      </footer>
    </div>
  );
}
