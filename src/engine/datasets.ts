import { random, type Sample } from "./neural";
export const DATASETS = [
  "XOR",
  "AND",
  "OR",
  "Linear",
  "Circle",
  "Spiral",
] as const;
export type Dataset = (typeof DATASETS)[number];
export const descriptions: Record<Dataset, string> = {
  XOR: "One input or the other, but not both. A classic nonlinear problem.",
  AND: "Class 1 only when both inputs are 1.",
  OR: "Class 1 when at least one input is 1.",
  Linear: "Separate two groups with a straight boundary.",
  Circle: "Learn a circular boundary around the center.",
  Spiral: "Untangle two intertwined arms with a nonlinear boundary.",
};
export function dataset(name: Dataset, seed = 23): Sample[] {
  if (["AND", "OR", "XOR"].includes(name))
    return [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ].map((x) => ({
      x,
      y: [
        name === "AND"
          ? +(x[0] && x[1])
          : name === "OR"
            ? +(x[0] || x[1])
            : +(x[0] !== x[1]),
      ],
    }));
  const rng = random(seed);
  if (name === "Spiral")
    return Array.from({ length: 160 }, (_, i) => {
      const c = i % 2,
        r = 0.08 + (0.9 * Math.floor(i / 2)) / 79,
        t = r * Math.PI * 2.2 + c * Math.PI + (rng() - 0.5) * 0.16;
      return { x: [r * Math.cos(t), r * Math.sin(t)], y: [c] };
    });
  return Array.from({ length: 160 }, () => {
    const x = [rng() * 2 - 1, rng() * 2 - 1];
    return {
      x,
      y: [
        name === "Circle"
          ? +(x[0] * x[0] + x[1] * x[1] < 0.45)
          : +(x[1] > 0.6 * x[0] + 0.1),
      ],
    };
  });
}
export function domain(name: Dataset): [number, number] {
  return ["XOR", "AND", "OR"].includes(name) ? [-0.25, 1.25] : [-1.1, 1.1];
}
