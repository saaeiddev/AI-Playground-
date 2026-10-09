import { useEffect, useRef } from "react";
import { predict, type Network, type Sample } from "../engine/neural";
import { domain, type Dataset } from "../engine/datasets";
export interface LossPoint {
  epoch: number;
  loss: number;
}
export function LossChart({ points }: { points: LossPoint[] }) {
  const max = Math.max(0.1, ...points.map((p) => p.loss)),
    last = points.at(-1)?.epoch ?? 0;
  const path = points
    .map(
      (p, i) =>
        `${i ? "L" : "M"}${42 + (p.epoch / Math.max(1, last)) * 340},${133 - (p.loss / max) * 102}`,
    )
    .join(" ");
  return (
    <svg
      className="loss-chart"
      viewBox="0 0 400 164"
      role="img"
      aria-label={`Training loss over ${last} epochs. Latest loss ${points.at(-1)?.loss.toPrecision(5)}`}
    >
      <defs>
        <linearGradient id="lossfill" x1="0" y1="0" x2="0" y2="1">
          <stop stopColor="#47c4e3" stopOpacity=".25" />
          <stop offset="1" stopColor="#47c4e3" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 0.5, 1].map((n) => (
        <g key={n}>
          <line
            x1="42"
            x2="384"
            y1={133 - n * 102}
            y2={133 - n * 102}
            stroke="#e5ecee"
            strokeDasharray="3 4"
          />
          <text x="3" y={137 - n * 102}>
            {(max * n).toFixed(2)}
          </text>
        </g>
      ))}
      {points.length > 1 && (
        <>
          <path d={`${path} L382,133 L42,133 Z`} fill="url(#lossfill)" />
          <path d={path} fill="none" stroke="#087fa7" strokeWidth="2.5" />
        </>
      )}
      <text x="42" y="156">
        0
      </text>
      <text x="375" y="156" textAnchor="end">
        {last} epochs
      </text>
      {points.length < 2 && (
        <text x="210" y="85" textAnchor="middle">
          Start training to plot real loss
        </text>
      )}
    </svg>
  );
}
export function DecisionBoundary({
  net,
  data,
  name,
  onProbe,
}: {
  net: Network;
  data: Sample[];
  name: Dataset;
  onProbe: (x: number[]) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!,
      ctx = canvas.getContext("2d")!,
      [lo, hi] = domain(name),
      size = 280,
      grid = 56;
    canvas.width = size;
    canvas.height = size;
    for (let j = 0; j < grid; j++)
      for (let i = 0; i < grid; i++) {
        const p = predict(net, [
          lo + ((i + 0.5) / grid) * (hi - lo),
          hi - ((j + 0.5) / grid) * (hi - lo),
        ])[0];
        const a = [230, 238, 249],
          b = [228, 244, 231];
        ctx.fillStyle = `rgb(${a.map((v, k) => Math.round(v * (1 - p) + b[k] * p)).join(",")})`;
        ctx.fillRect(
          (i * size) / grid,
          (j * size) / grid,
          size / grid + 1,
          size / grid + 1,
        );
      }
    const zero = ((0 - lo) / (hi - lo)) * size;
    ctx.strokeStyle = "#becdd455";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(zero, 0);
    ctx.lineTo(zero, size);
    ctx.moveTo(0, size - zero);
    ctx.lineTo(size, size - zero);
    ctx.stroke();
    data.forEach((s) => {
      ctx.beginPath();
      ctx.arc(
        ((s.x[0] - lo) / (hi - lo)) * size,
        ((hi - s.x[1]) / (hi - lo)) * size,
        data.length < 10 ? 6 : 3.2,
        0,
        Math.PI * 2,
      );
      ctx.fillStyle = s.y[0] ? "#4a9871" : "#5177b7";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }, [net, data, name]);
  return (
    <div className="boundary-wrap">
      <span className="axis-y">x₂</span>
      <canvas
        ref={ref}
        className="boundary"
        role="img"
        aria-label="Decision boundary. Blue represents class 0, green class 1. Click to inspect a prediction."
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect(),
            [lo, hi] = domain(name);
          onProbe([
            lo + ((e.clientX - r.left) / r.width) * (hi - lo),
            hi - ((e.clientY - r.top) / r.height) * (hi - lo),
          ]);
        }}
      />
      <span className="axis-x">x₁</span>
      <div className="legend">
        <span>
          <i style={{ background: "#5177b7" }} />
          Class 0
        </span>
        <span>
          <i style={{ background: "#4a9871" }} />
          Class 1
        </span>
        <span>Click to probe</span>
      </div>
    </div>
  );
}
