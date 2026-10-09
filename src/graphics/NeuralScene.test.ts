import { beforeAll, describe, it, expect, vi } from "vitest";
import { Vector3, Texture, Mesh } from "three";
import { createNetwork, forward } from "../engine/neural";
vi.mock("three", async (original) => {
  const actual = await original<typeof import("three")>();
  return {
    ...actual,
    WebGLRenderer: class {
      domElement = {
        setAttribute() {},
        style: {},
        addEventListener() {},
        removeEventListener() {},
        remove() {},
      };
      shadowMap = { enabled: false, type: 0 };
      setPixelRatio() {}
      setClearColor() {}
      setSize() {}
      render() {}
      dispose() {}
      forceContextLoss() {}
    },
    PMREMGenerator: class {
      fromScene() {
        return { texture: new actual.Texture(), dispose() {} };
      }
      dispose() {}
    },
  };
});
vi.mock("three/examples/jsm/controls/OrbitControls.js", () => ({
  OrbitControls: class {
    target = new Vector3();
    constructor() {}
    update() {}
    dispose() {}
  },
}));
import { NeuralScene, type SceneState } from "./NeuralScene";
beforeAll(() => {
  vi.stubGlobal("window", { devicePixelRatio: 1, innerWidth: 1400 });
  vi.stubGlobal("document", {
    hidden: false,
    createElement: () => ({
      width: 0,
      height: 0,
      getContext: () => ({ clearRect() {}, fillText() {} }),
    }),
  });
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal("requestAnimationFrame", () => 1);
  vi.stubGlobal("cancelAnimationFrame", () => {});
});
const host = () =>
  ({
    clientWidth: 600,
    clientHeight: 360,
    appendChild() {},
  }) as unknown as HTMLDivElement;
const state: SceneState = {
  mode: "single",
  inputs: [0.8, 0.4, 0.6],
  weights: [0.7, -0.5, 0.9],
  bias: -0.2,
  activation: "sigmoid",
  reduced: true,
};
describe("3D graph and computational synchronization (GPU-independent)", () => {
  it("creates volumetric geometry with finite vertices for all three scenes", () => {
    for (const mode of ["single", "bio", "network"] as const) {
      const net = createNetwork();
      const scene = new NeuralScene(
        host(),
        {
          ...state,
          mode,
          net,
          inputs: mode === "network" ? [0, 1] : state.inputs,
        },
        () => {},
      );
      let count = 0;
      scene.root.traverse((o) => {
        if (o instanceof Mesh && o.geometry) {
          const a = o.geometry.attributes.position;
          for (let i = 0; i < a.array.length; i++)
            expect(Number.isFinite(a.array[i])).toBe(true);
          count++;
        }
      });
      expect(count).toBeGreaterThan(10);
      expect(scene.camera.position.length()).toBeGreaterThan(1);
      scene.dispose();
      expect(scene.root.children).toHaveLength(0);
    }
  });
  it("renders each weighted contribution, the bias once, and the real output", () => {
    const scene = new NeuralScene(host(), state, () => {});
    expect(scene.links).toHaveLength(5);
    expect(scene.links.map((l) => l.value)).toEqual([
      -0.2,
      0.8 * 0.7,
      0.4 * -0.5,
      0.6 * 0.9,
      1 / (1 + Math.exp(-0.7)),
    ]);
    scene.update({ ...state, weights: [0, -2, 1] });
    expect(scene.links[1].value).toBe(0);
    expect(scene.links[2].value).toBe(-0.8);
    scene.dispose();
  });
  it("network connections use source activations from the actual forward pass", () => {
    const net = createNetwork([2, 3, 1]),
      inputs = [0.4, -0.7],
      s = { ...state, mode: "network" as const, net, inputs };
    const scene = new NeuralScene(host(), s, () => {});
    const t = forward(net, inputs);
    expect(scene.nodes).toHaveLength(6);
    expect(scene.links).toHaveLength(9);
    scene.links.forEach((l) =>
      expect(l.value).toBeCloseTo(
        net.weights[l.layer][l.destination][l.source] * t.a[l.layer][l.source],
        12,
      ),
    );
    net.weights[0][0][0] = 3;
    scene.update({ ...s, selected: { layer: 1, node: 0 } });
    expect(scene.links[0].value).toBeCloseTo(1.2);
    expect(
      scene.nodes.find((n) => n.layer === 1 && n.node === 0)?.rim.scale.x,
    ).toBe(1.15);
    scene.dispose();
  });
  it("builds the required biological structures and handles input count changes", () => {
    const s = new NeuralScene(host(), { ...state, mode: "bio" }, () => {});
    for (const id of [
      "Dendrites",
      "Soma",
      "Nucleus",
      "Axon hillock",
      "Axon",
      "Myelin sheath",
      "Nodes of Ranvier",
      "Axon terminals",
      "Synapses",
    ])
      expect(s.parts.has(id)).toBe(true);
    s.dispose();
    const n = new NeuralScene(host(), state, () => {});
    n.update({ ...state, inputs: [1], weights: [2] });
    expect(n.nodes).toHaveLength(3);
    expect(n.links).toHaveLength(3);
    n.dispose();
  });
});
void Texture;
