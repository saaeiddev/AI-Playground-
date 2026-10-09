import { useEffect, useState } from "react";
import {
  ACTIVATIONS,
  activationNames,
  type Activation,
} from "../engine/neural";
export const fmt = (n: number, d = 4) =>
  Number.isFinite(n) ? Number(n.toFixed(d)).toString() : "—";
export function NumberControl({
  label,
  value,
  onChange,
  min = -5,
  max = 5,
  step = 0.05,
  slider = true,
  disabled = false,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
  slider?: boolean;
  disabled?: boolean;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(Number(value.toFixed(5)))), [value]);
  return (
    <label className="number-control">
      <span>{label}</span>
      <div className="number-row">
        {slider && (
          <input
            aria-label={`${label} slider`}
            type="range"
            min={min}
            max={max}
            step={step}
            value={Math.max(min, Math.min(max, value))}
            onChange={(e) => onChange(+e.target.value)}
            disabled={disabled}
          />
        )}
        <input
          aria-label={label}
          type="number"
          min={min}
          max={max}
          step={step}
          value={draft}
          disabled={disabled}
          onChange={(e) => {
            setDraft(e.target.value);
            if (
              e.target.value !== "" &&
              Number.isFinite(+e.target.value) &&
              +e.target.value >= min &&
              +e.target.value <= max
            )
              onChange(+e.target.value);
          }}
          onBlur={() => setDraft(String(Number(value.toFixed(5))))}
        />
      </div>
    </label>
  );
}
export function ActivationSelect({
  value,
  onChange,
  training = false,
  disabled = false,
}: {
  value: Activation;
  onChange: (v: Activation) => void;
  training?: boolean;
  disabled?: boolean;
}) {
  return (
    <label className="field-label">
      {training ? "Hidden activation" : "Activation function"}
      <select
        aria-label={training ? "Hidden activation" : "Activation function"}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value as Activation)}
      >
        {ACTIVATIONS.filter((a) => !training || a !== "step").map((a) => (
          <option key={a} value={a}>
            {activationNames[a]}
          </option>
        ))}
      </select>
    </label>
  );
}
export function Legend() {
  return (
    <div className="legend">
      <span>
        <i className="positive" />
        Positive weight
      </span>
      <span>
        <i className="negative" />
        Negative weight
      </span>
      <span>Brightness ∝ |weight|</span>
    </div>
  );
}
