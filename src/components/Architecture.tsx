import { Plus, Minus } from "lucide-react";
export default function Architecture({
  sizes,
  onChange,
  training = false,
  disabled = false,
}: {
  sizes: number[];
  onChange: (s: number[]) => void;
  training?: boolean;
  disabled?: boolean;
}) {
  return (
    <div className="architecture">
      <div className="panel-heading">
        <h3>Architecture</h3>
        <span className="tiny-badge">{sizes.join(" · ")}</span>
      </div>
      <div className="layer-controls">
        {sizes.map((n, l) => (
          <div className="layer-control" key={l}>
            <span>
              {l === 0
                ? "Inputs"
                : l === sizes.length - 1
                  ? "Outputs"
                  : `Hidden ${l}`}
            </span>
            <div>
              <button
                aria-label={`Remove neuron from layer ${l}`}
                disabled={
                  disabled ||
                  n <= 1 ||
                  (training && (l === 0 || l === sizes.length - 1))
                }
                onClick={() =>
                  onChange(sizes.map((v, i) => (i === l ? v - 1 : v)))
                }
              >
                <Minus size={13} />
              </button>
              <strong>{n}</strong>
              <button
                aria-label={`Add neuron to layer ${l}`}
                disabled={
                  disabled ||
                  n >= 8 ||
                  (training && (l === 0 || l === sizes.length - 1))
                }
                onClick={() =>
                  onChange(sizes.map((v, i) => (i === l ? v + 1 : v)))
                }
              >
                <Plus size={13} />
              </button>
            </div>
          </div>
        ))}
      </div>
      <div className="button-row">
        <button
          className="small-button"
          aria-label="Add hidden layer"
          disabled={disabled || sizes.length >= 6}
          onClick={() => onChange([...sizes.slice(0, -1), 4, sizes.at(-1)!])}
        >
          <Plus size={14} />
          Hidden layer
        </button>
        <button
          className="small-button"
          aria-label="Remove hidden layer"
          disabled={disabled || sizes.length <= 2}
          onClick={() => onChange([...sizes.slice(0, -2), sizes.at(-1)!])}
        >
          <Minus size={14} />
          Hidden layer
        </button>
      </div>
      <p className="hint">
        Dense connections · up to 4 hidden layers, 8 neurons each. Architecture
        changes reset parameters.
      </p>
    </div>
  );
}
