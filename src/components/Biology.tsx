import { useState } from "react";
import { Atom, Check, Pause, Play } from "lucide-react";
import Scene from "./Scene";
import { NumberControl } from "./Controls";

export const anatomy: Record<string, string> = {
  Dendrites: "Branched extensions receive graded synaptic inputs.",
  Soma: "The cell body supports metabolism and integrates incoming signals.",
  Nucleus:
    "Contains DNA and regulates cell function; electrical signals do not pass through it.",
  "Axon hillock":
    "The axon emerges here. Action potentials usually initiate in the nearby axon initial segment.",
  Axon: "Carries action potentials toward the terminals.",
  "Myelin sheath": "Insulation that speeds conduction along the axon.",
  "Nodes of Ranvier": "Gaps in myelin where action potentials are regenerated.",
  "Axon terminals":
    "Calcium entry triggers neurotransmitter release at chemical synapses.",
  Synapses: "Junctions where a neuron communicates with another cell.",
};

export default function Biology({ reduced }: { reduced: boolean }) {
  const [part, setPart] = useState("Dendrites"),
    [playing, setPlaying] = useState(true),
    [labels, setLabels] = useState(true),
    [potential, setPotential] = useState(-50);

  return (
    <section className="biology-model" aria-label="Interactive biology model">
      <div className="panel-heading">
        <h3>Biological neuron · 3D</h3>
        <span className="specimen-badge amber">
          <Atom size={15} /> Living system
        </span>
      </div>
      <Scene
        className="biology-model-scene"
        mode="bio"
        inputs={[]}
        weights={[]}
        bias={0}
        activation="linear"
        reduced={reduced}
        running={playing}
        labels={labels}
        bioFires={potential >= -55}
        onPart={setPart}
      />
      <div className="biology-model-controls">
        <div className="button-row">
          <button
            className="small-button"
            onClick={() => setPlaying((v) => !v)}
          >
            {playing ? <Pause size={14} /> : <Play size={14} />}
            {playing ? "Pause signals" : "Play signals"}
          </button>
          <button
            className="small-button"
            aria-pressed={labels}
            onClick={() => setLabels((v) => !v)}
          >
            <Check size={14} /> Labels {labels ? "on" : "off"}
          </button>
        </div>
        <p className="hint">
          Rotate, zoom, or click a structure to inspect it.
        </p>
        <div className="biology-inspector">
          <label className="field-label">
            Selected structure
            <select
              aria-label="Biological structure"
              value={part}
              onChange={(e) => setPart(e.target.value)}
            >
              {Object.keys(anatomy).map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
          </label>
          <p className="biology-description" aria-live="polite">
            {anatomy[part]}
          </p>
        </div>
        <NumberControl
          label="Illustrative membrane potential (mV)"
          value={potential}
          min={-80}
          max={-40}
          step={1}
          onChange={setPotential}
        />
        <p className="hint">
          {potential >= -55
            ? "Threshold reached · axon signal"
            : "Below threshold · incoming signals only"}
          {" · "}−55 mV is an educational example, not a universal threshold.
        </p>
      </div>
    </section>
  );
}
