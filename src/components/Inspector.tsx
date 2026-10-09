import { Crosshair } from "lucide-react";
import {
  forward,
  setWeight,
  setBias,
  activationNames,
  type Network,
} from "../engine/neural";
import type { Selection } from "../graphics/NeuralScene";
import { NumberControl, fmt } from "./Controls";
export default function Inspector({
  net,
  inputs,
  selected,
  onSelect,
  onChange,
  disabled = false,
}: {
  net: Network;
  inputs: number[];
  selected: Selection;
  onSelect: (s: Selection) => void;
  onChange: (n: Network) => void;
  disabled?: boolean;
}) {
  const trace = forward(net, inputs),
    { layer: l, node: j } = selected;
  const edit = (fn: (copy: Network) => void) => {
    const copy = structuredClone(net);
    fn(copy);
    onChange(copy);
  };
  return (
    <section className="inspector">
      <div className="panel-heading">
        <h3>
          <Crosshair size={17} />
          Neuron inspector
        </h3>
        <span className="tiny-badge">LIVE</span>
      </div>
      <label className="field-label">
        Selected neuron
        <select
          aria-label="Selected neuron"
          value={`${l}:${j}`}
          onChange={(e) => {
            const [layer, node] = e.target.value.split(":").map(Number);
            onSelect({ layer, node });
          }}
        >
          {net.sizes.flatMap((n, layer) =>
            Array.from({ length: n }, (_, node) => (
              <option key={`${layer}:${node}`} value={`${layer}:${node}`}>
                L{layer} · N{node + 1}
                {layer === 0 ? " · Input" : ""}
              </option>
            )),
          )}
        </select>
      </label>
      <div className="inspect-values">
        <div>
          <span>Neuron ID</span>
          <strong>
            L{l} · N{j + 1}
          </strong>
        </div>
        <div>
          <span>Activation</span>
          <strong>
            {l === 0
              ? "Input"
              : activationNames[
                  l === net.sizes.length - 1
                    ? net.outputActivation
                    : net.hiddenActivation
                ]}
          </strong>
        </div>
        <div>
          <span>Weighted sum z</span>
          <strong>{l ? fmt(trace.z[l - 1][j]) : "—"}</strong>
        </div>
        <div>
          <span>Output a</span>
          <strong className="cyan">{fmt(trace.a[l][j])}</strong>
        </div>
      </div>
      {l > 0 && (
        <>
          <NumberControl
            label="Neuron bias"
            value={net.biases[l - 1][j]}
            min={-20}
            max={20}
            onChange={(v) => edit((n) => setBias(n, l - 1, j, v))}
            disabled={disabled}
          />
          <h4>Incoming connections</h4>
          <div className="incoming">
            {net.weights[l - 1][j].map((w, i) => (
              <div key={i} className="incoming-row">
                <span>
                  x{i + 1} = {fmt(trace.a[l - 1][i], 3)}
                </span>
                <NumberControl
                  label={`Weight from L${l - 1} N${i + 1}`}
                  value={w}
                  min={-20}
                  max={20}
                  slider={false}
                  onChange={(v) => edit((n) => setWeight(n, l - 1, j, i, v))}
                  disabled={disabled}
                />
              </div>
            ))}
          </div>
        </>
      )}
      <p className="hint">
        {l === net.sizes.length - 1
          ? "Output neuron · no outgoing connections."
          : `${net.sizes[l + 1]} outgoing connections to layer ${l + 1}.`}{" "}
        {disabled
          ? "Pause training to edit parameters."
          : "Select a glass neuron or connection to inspect it."}
      </p>
      {l < net.sizes.length - 1 && (
        <div className="outgoing">
          {net.weights[l].map((row, i) => (
            <span key={i}>
              N{i + 1}: {fmt(row[j], 3)}
            </span>
          ))}
        </div>
      )}
    </section>
  );
}
