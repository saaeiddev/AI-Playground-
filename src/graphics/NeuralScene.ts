import * as T from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import {
  forward,
  type Network,
  type Activation,
  neuron,
} from "../engine/neural";
export type Selection = { layer: number; node: number };
export interface SceneState {
  mode: "single" | "bio" | "network";
  inputs: number[];
  weights: number[];
  bias: number;
  activation: Activation;
  net?: Network;
  selected?: Selection | null;
  running?: boolean;
  reduced: boolean;
  step?: number;
  bioFires?: boolean;
  labels?: boolean;
  onSelect?: (s: Selection) => void;
  onPart?: (id: string) => void;
  onConnection?: (l: number, j: number, i: number) => void;
}
interface TextLabel {
  sprite: T.Sprite;
  set: (text: string) => void;
}
interface Link {
  curve: T.CatmullRomCurve3;
  tube: T.Mesh;
  particle: T.Mesh;
  value: number;
  layer: number;
  destination: number;
  source: number;
  label?: TextLabel;
  offset: number;
}
interface Node3D {
  group: T.Group;
  core: T.Mesh;
  rim: T.Mesh;
  label?: TextLabel;
  layer: number;
  node: number;
}
const v = (x: number, y: number, z = 0) => new T.Vector3(x, y, z);
const positive = "#139bcc",
  negative = "#8a62b8";
export class NeuralScene {
  renderer: T.WebGLRenderer;
  scene = new T.Scene();
  camera = new T.PerspectiveCamera(34, 1, 0.1, 100);
  controls: OrbitControls;
  root = new T.Group();
  state: SceneState;
  nodes: Node3D[] = [];
  links: Link[] = [];
  labels: TextLabel[] = [];
  frame = 0;
  last = 0;
  time = 0;
  key = "";
  observer: ResizeObserver;
  visible = true;
  intersector: IntersectionObserver;
  ray = new T.Raycaster();
  mouse = new T.Vector2();
  pickables: T.Object3D[] = [];
  down = { x: 0, y: 0 };
  env: T.WebGLRenderTarget;
  parts = new Map<string, T.Object3D[]>();
  bioParticles: { mesh: T.Mesh; curve: T.CatmullRomCurve3; phase: number }[] =
    [];
  hovered: T.Object3D | null = null;
  disposed = false;
  onError: () => void;
  cameraDistance = 11;
  constructor(
    public host: HTMLDivElement,
    state: SceneState,
    onError: () => void,
  ) {
    this.state = state;
    this.onError = onError;
    this.renderer = new T.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(
      Math.min(window.devicePixelRatio, window.innerWidth < 700 ? 1.3 : 1.65),
    );
    this.renderer.setClearColor(0xf7fbfd, 0);
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.18;
    this.renderer.shadowMap.enabled = window.innerWidth > 700;
    this.renderer.shadowMap.type = T.PCFSoftShadowMap;
    this.renderer.domElement.setAttribute(
      "aria-label",
      state.mode === "bio"
        ? "Interactive 3D biological neuron"
        : state.mode === "single"
          ? "Interactive 3D glass artificial neuron"
          : "Interactive 3D glass neural network",
    );
    this.renderer.domElement.setAttribute("role", "img");
    this.renderer.domElement.style.touchAction = "none";
    host.appendChild(this.renderer.domElement);
    const pmrem = new T.PMREMGenerator(this.renderer),
      room = new RoomEnvironment();
    this.env = pmrem.fromScene(room, 0.04);
    room.dispose();
    pmrem.dispose();
    this.scene.environment = this.env.texture;
    this.scene.add(new T.HemisphereLight(0xf6ffff, 0xa3b4bc, 2));
    const light = new T.DirectionalLight(0xffffff, 4);
    light.position.set(-3, 7, 6);
    light.castShadow = true;
    light.shadow.mapSize.set(1024, 1024);
    light.shadow.camera.left = -8;
    light.shadow.camera.right = 8;
    light.shadow.camera.top = 6;
    light.shadow.camera.bottom = -6;
    light.shadow.bias = -0.002;
    this.scene.add(light);
    const rimLight = new T.DirectionalLight(0x64d6ff, 2);
    rimLight.position.set(4, 2, -3);
    this.scene.add(rimLight);
    this.scene.add(this.root);
    this.camera.position.set(0, 1.4, 11);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 25;
    this.controls.maxPolarAngle = Math.PI * 0.85;
    this.controls.target.set(0, 0, 0);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.intersector = new IntersectionObserver((e) => {
      this.visible = e[0].isIntersecting;
    });
    this.intersector.observe(host);
    this.renderer.domElement.addEventListener("pointerdown", this.pointerDown);
    this.renderer.domElement.addEventListener("pointerup", this.pointerUp);
    this.renderer.domElement.addEventListener("pointermove", this.pointerMove);
    this.renderer.domElement.addEventListener(
      "webglcontextlost",
      this.contextLost,
    );
    this.update(state);
    this.resize();
    this.animate(0);
  }
  contextLost = (e: Event) => {
    e.preventDefault();
    this.onError();
  };
  pointerDown = (e: PointerEvent) => {
    this.down = { x: e.clientX, y: e.clientY };
  };
  hit(e: PointerEvent) {
    const rect = this.host.getBoundingClientRect();
    this.mouse.set(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      (-(e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.ray.setFromCamera(this.mouse, this.camera);
    return this.ray.intersectObjects(this.pickables, false)[0]?.object;
  }
  pointerMove = (e: PointerEvent) => {
    const hit = this.hit(e) || null;
    this.renderer.domElement.style.cursor = hit ? "pointer" : "grab";
    if (hit !== this.hovered) {
      if (this.hovered) {
        const old = this.nodes.find((n) => n.group === this.hovered?.parent);
        if (old) {
          old.rim.scale.setScalar(1);
          (old.rim.material as T.MeshBasicMaterial).opacity = 0.44;
        }
      }
      this.hovered = hit;
      this.update(this.state);
      if (hit) {
        const node = this.nodes.find((n) => n.group === hit.parent);
        if (node) {
          node.rim.scale.setScalar(1.12);
          (node.rim.material as T.MeshBasicMaterial).opacity = 1;
        }
        const link = this.links.find((l) => l.tube === hit);
        if (link) (link.tube.material as T.MeshStandardMaterial).opacity = 1;
      }
    }
  };
  pointerUp = (e: PointerEvent) => {
    if (Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) > 6)
      return;
    const o = this.hit(e);
    if (o?.userData.selection) this.state.onSelect?.(o.userData.selection);
    if (o?.userData.part) this.state.onPart?.(o.userData.part);
    if (o?.userData.connection) {
      const c = o.userData.connection;
      this.state.onConnection?.(c.l, c.j, c.i);
    }
  };
  resize() {
    const { clientWidth: w, clientHeight: h } = this.host;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (!this.key) return;
    this.fit(false);
  }
  fit(reset = true) {
    const mode = this.state.mode;
    const width =
      mode === "network"
        ? Math.max(5.8, (this.state.net!.sizes.length - 1) * 2.6 + 1.5)
        : mode === "single"
          ? 7.4
          : 8.4;
    const height =
      mode === "network"
        ? Math.max(...this.state.net!.sizes) * 1.0 + 1.2
        : mode === "single"
          ? Math.max(4.4, this.state.inputs.length * 0.75 + 1.4)
          : 5.4;
    this.cameraDistance =
      (Math.max(height, width / this.camera.aspect) /
        (2 * Math.tan(T.MathUtils.degToRad(17)))) *
      1.08;
    this.cameraDistance = Math.min(42, this.cameraDistance);
    this.controls.maxDistance = this.cameraDistance * 2.1;
    if (reset) {
      this.camera.position.set(0, 0.8, this.cameraDistance);
      this.controls.target.set(0, 0, 0);
      this.controls.update();
    } else {
      const direction = this.camera.position
        .clone()
        .sub(this.controls.target)
        .normalize();
      this.camera.position.copy(
        this.controls.target
          .clone()
          .add(direction.multiplyScalar(this.cameraDistance)),
      );
    }
  }
  zoom(factor: number) {
    this.camera.position
      .sub(this.controls.target)
      .multiplyScalar(factor)
      .add(this.controls.target);
    this.controls.update();
  }
  text(text: string, pos: T.Vector3, size = 0.4, color = "#234357"): TextLabel {
    const c = document.createElement("canvas");
    c.width = 640;
    c.height = 180;
    const ctx = c.getContext("2d")!,
      texture = new T.CanvasTexture(c);
    texture.colorSpace = T.SRGBColorSpace;
    const mat = new T.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    const sprite = new T.Sprite(mat);
    sprite.position.copy(pos);
    sprite.scale.set(size * 3.55, size, 1);
    sprite.renderOrder = 12;
    let previous = "";
    const set = (s: string) => {
      if (s === previous) return;
      previous = s;
      ctx.clearRect(0, 0, 640, 180);
      ctx.font = "500 104px Georgia, serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = color;
      ctx.shadowColor = color === "#ffffff" ? "#5edbff" : "transparent";
      ctx.shadowBlur = color === "#ffffff" ? 14 : 0;
      ctx.fillText(s, 320, 91);
      texture.needsUpdate = true;
    };
    set(text);
    return { sprite, set };
  }
  glass(pos: T.Vector3, r: number, split = true): Node3D {
    const g = new T.Group();
    g.position.copy(pos);
    const geometry = new T.SphereGeometry(r, 48, 32);
    const shell = new T.Mesh(
      geometry,
      new T.MeshPhysicalMaterial({
        color: 0x77d8ff,
        metalness: 0.05,
        roughness: 0.07,
        transmission: 0.93,
        thickness: r * 0.6,
        ior: 1.46,
        clearcoat: 1,
        clearcoatRoughness: 0.06,
        envMapIntensity: 1.3,
        attenuationColor: new T.Color(0x38b7fa),
        attenuationDistance: 3,
      }),
    );
    g.add(shell);
    shell.castShadow = true;
    const core = new T.Mesh(
      new T.SphereGeometry(r * 0.9, 40, 28),
      new T.MeshPhysicalMaterial({
        color: split ? 0x0278c9 : 0x65caff,
        metalness: 0.16,
        roughness: 0.19,
        transparent: true,
        opacity: split ? 0.63 : 0.28,
        transmission: 0.25,
        thickness: 0.35,
        clearcoat: 1,
        emissive: 0x006dc9,
        emissiveIntensity: 0.12,
      }),
    );
    g.add(core);
    if (split) {
      const half = new T.Mesh(
        new T.SphereGeometry(r * 0.905, 32, 24, Math.PI / 2, Math.PI),
        new T.MeshPhysicalMaterial({
          color: 0x43d6f7,
          roughness: 0.13,
          transparent: true,
          opacity: 0.56,
          transmission: 0.3,
          thickness: 0.2,
          clearcoat: 1,
        }),
      );
      g.add(half);
      const curve = new T.CatmullRomCurve3([
        v(0, -r * 0.98, 0),
        v(0.03, -r * 0.6, r * 0.75),
        v(0.06, 0, r * 0.98),
        v(0.03, r * 0.6, r * 0.75),
        v(0, r * 0.98, 0),
      ]);
      const divider = new T.Mesh(
        new T.TubeGeometry(curve, 32, r * 0.013, 8, false),
        new T.MeshStandardMaterial({
          color: 0xccf9ff,
          emissive: 0x45d6ff,
          emissiveIntensity: 1.1,
          metalness: 0.3,
          roughness: 0.1,
        }),
      );
      g.add(divider);
      const sigma = this.text(
          "Σ",
          v(-r * 0.39, 0, r * 0.91),
          r * 0.7,
          "#ffffff",
        ),
        f = this.text("ƒ", v(r * 0.41, 0, r * 0.91), r * 0.7, "#ffffff");
      g.add(sigma.sprite, f.sprite);
    }
    const ring = new T.Mesh(
      new T.TorusGeometry(r * 1.012, r * 0.018, 8, 80),
      new T.MeshBasicMaterial({
        color: 0x68dcff,
        transparent: true,
        opacity: 0.44,
      }),
    );
    g.add(ring);
    this.root.add(g);
    const n = { group: g, core, rim: ring, layer: 0, node: 0 };
    this.nodes.push(n);
    this.pickables.push(shell);
    return n;
  }
  line(
    start: T.Vector3,
    end: T.Vector3,
    value: number,
    layer = 0,
    destination = 0,
    source = 0,
    label?: string,
  ) {
    const mid = start.clone().lerp(end, 0.5);
    mid.z -= Math.min(0.5, Math.abs(start.y - end.y) * 0.13);
    const curve = new T.CatmullRomCurve3([start, mid, end]);
    const tube = new T.Mesh(
      new T.TubeGeometry(curve, 24, 0.016, 6, false),
      new T.MeshStandardMaterial({
        color: value < 0 ? negative : positive,
        emissive: value < 0 ? negative : positive,
        emissiveIntensity: 0.35,
        transparent: true,
        opacity: 0.5,
      }),
    );
    this.root.add(tube);
    const particle = new T.Mesh(
      new T.SphereGeometry(0.043, 10, 8),
      new T.MeshBasicMaterial({ color: value < 0 ? 0xbf96f0 : 0x71e5ff }),
    );
    this.root.add(particle);
    const link: Link = {
      curve,
      tube,
      particle,
      value,
      layer,
      destination,
      source,
      offset: source * 0.11,
    };
    if (label) {
      link.label = this.text(
        label,
        curve.getPoint(0.45).add(v(0, 0.22, 0)),
        0.22,
      );
      this.root.add(link.label.sprite);
    }
    this.links.push(link);
    tube.userData.connection = { l: layer, j: destination, i: source };
    if (this.state.mode === "network") this.pickables.push(tube);
    return link;
  }
  single() {
    const r = 1.24;
    const main = this.glass(v(0.8, 0, 0), r);
    main.core.userData.main = true;
    const rows = [1, ...this.state.inputs],
      totalHeight = Math.min(5.6, rows.length * 0.84);
    rows.forEach((_, i) => {
      const y = totalHeight / 2 - i * (totalHeight / (rows.length - 1));
      const node = this.glass(v(-2.5, y, 0), 0.28, false);
      const label = this.text(
        i === 0 ? "x₀ = 1" : `x${sub(i)}`,
        v(0, 0, 0.29),
        0.28,
      );
      node.group.add(label.sprite);
      node.layer = -1;
      node.node = i;
      this.line(
        v(-2.2, y, 0),
        v(
          0.8 - Math.sqrt(Math.max(0.16, r * r - Math.min(y * y, 0.9))),
          Math.max(-0.88, Math.min(0.88, y * 0.4)),
          0,
        ),
        i === 0 ? this.state.bias : this.state.weights[i - 1],
        0,
        0,
        i,
        i === 0 ? "b" : `w${sub(i)}`,
      );
    });
    this.line(v(2.08, 0, 0), v(3.3, 0, 0), 1, 1, 0, 0);
    const cone = new T.Mesh(
      new T.ConeGeometry(0.13, 0.33, 24),
      new T.MeshPhysicalMaterial({
        color: 0x08a8ec,
        metalness: 0.2,
        roughness: 0.15,
        clearcoat: 1,
      }),
    );
    cone.rotation.z = -Math.PI / 2;
    cone.position.set(3.33, 0, 0);
    this.root.add(cone);
    const label = this.text("y", v(2.75, 0.4, 0), 0.4);
    this.root.add(label.sprite);
    this.labels.push(label);
    const shadow = new T.Mesh(
      new T.PlaneGeometry(12, 10),
      new T.ShadowMaterial({ opacity: 0.12 }),
    );
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = -totalHeight / 2 - 0.4;
    shadow.receiveShadow = true;
    this.root.add(shadow);
  }
  network() {
    const net = this.state.net!,
      L = net.sizes.length;
    const pos = (l: number, j: number) =>
      v((l - (L - 1) / 2) * 2.6, (net.sizes[l] - 1) / 2 - j, 0);
    for (let l = 1; l < L; l++)
      for (let j = 0; j < net.sizes[l]; j++)
        for (let i = 0; i < net.sizes[l - 1]; i++)
          this.line(
            pos(l - 1, i).add(v(0.34, 0, 0)),
            pos(l, j).add(v(-0.34, 0, 0)),
            net.weights[l - 1][j][i],
            l - 1,
            j,
            i,
          );
    net.sizes.forEach((num, l) => {
      const caption = this.text(
        l === 0 ? "INPUT" : l === L - 1 ? "OUTPUT" : `HIDDEN ${l}`,
        v((l - (L - 1) / 2) * 2.6, Math.max(...net.sizes) / 2 + 0.25, 0),
        0.24,
      );
      this.root.add(caption.sprite);
      for (let j = 0; j < num; j++) {
        const n = this.glass(pos(l, j), l === 0 ? 0.26 : 0.36, l > 0);
        n.layer = l;
        n.node = j;
        n.group.children[0].userData.selection = { layer: l, node: j };
        const label = this.text("0.00", v(0, -0.54, 0), 0.21);
        n.group.add(label.sprite);
        n.label = label;
      }
    });
  }
  addPart(mesh: T.Object3D, id: string) {
    mesh.userData.part = id;
    this.pickables.push(mesh);
    this.parts.set(id, [...(this.parts.get(id) || []), mesh]);
    this.root.add(mesh);
  }
  bioTube(points: T.Vector3[], radius: number, id: string, color = 0xc99256) {
    const curve = new T.CatmullRomCurve3(points);
    const mesh = new T.Mesh(
      new T.TubeGeometry(curve, 20, radius, 8, false),
      new T.MeshPhysicalMaterial({
        color,
        roughness: 0.38,
        metalness: 0.08,
        clearcoat: 0.4,
      }),
    );
    mesh.castShadow = true;
    this.addPart(mesh, id);
    return curve;
  }
  bio() {
    const center = v(-1.5, 0.15, 0),
      geo = new T.IcosahedronGeometry(0.65, 4);
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const a = v(p.getX(i), p.getY(i), p.getZ(i));
      a.multiplyScalar(1 + 0.065 * Math.sin(a.x * 14) * Math.cos(a.y * 11));
      p.setXYZ(i, a.x, a.y, a.z);
    }
    geo.computeVertexNormals();
    const soma = new T.Mesh(
      geo,
      new T.MeshPhysicalMaterial({
        color: 0xe5b17d,
        roughness: 0.34,
        transparent: true,
        opacity: 0.88,
        transmission: 0.1,
        thickness: 0.6,
        clearcoat: 0.4,
      }),
    );
    soma.position.copy(center);
    this.addPart(soma, "Soma");
    const nucleus = new T.Mesh(
      new T.SphereGeometry(0.235, 32, 24),
      new T.MeshPhysicalMaterial({
        color: 0xbc795a,
        roughness: 0.27,
        clearcoat: 1,
      }),
    );
    nucleus.position.copy(center.clone().add(v(-0.03, 0.03, 0.46)));
    this.addPart(nucleus, "Nucleus");
    const dendriteEnds: T.Vector3[] = [];
    for (let i = 0; i < 9; i++) {
      const angle = 0.75 + i * 0.59,
        dir = v(Math.cos(angle), Math.sin(angle), Math.sin(i * 3.4) * 0.28);
      const end = center
        .clone()
        .add(dir.clone().multiplyScalar(1.5 + (i % 3) * 0.3));
      const mid = center
        .clone()
        .lerp(end, 0.53)
        .add(v(0.02, Math.sin(i) * 0.17, 0.1));
      const curve = this.bioTube([center, mid, end], 0.065, "Dendrites");
      dendriteEnds.push(end);
      for (let j = 0; j < 3; j++) {
        const root = curve.getPoint(0.48 + j * 0.23),
          a = angle + (j % 2 ? -0.65 : 0.65);
        const tip = root
          .clone()
          .add(
            v(Math.cos(a) * 0.73, Math.sin(a) * 0.73, Math.cos(i + j) * 0.27),
          );
        this.bioTube(
          [
            root,
            root
              .clone()
              .lerp(tip, 0.5)
              .add(v(-0.1, 0.04, 0.1)),
            tip,
          ],
          0.025,
          "Dendrites",
        );
        for (let k = 0; k < 2; k++) {
          const twig = tip
            .clone()
            .add(
              v(
                Math.cos(a + (k ? -0.5 : 0.5)) * 0.38,
                Math.sin(a + (k ? -0.5 : 0.5)) * 0.38,
                0.09 * (k ? 1 : -1),
              ),
            );
          this.bioTube([tip, twig], 0.009, "Dendrites");
          const syn = new T.Mesh(
            new T.SphereGeometry(0.045, 10, 8),
            new T.MeshStandardMaterial({
              color: 0xe5b15c,
              emissive: 0xf9a72e,
              emissiveIntensity: 0.2,
            }),
          );
          syn.position.copy(twig);
          this.addPart(syn, "Synapses");
        }
      }
      this.addBioParticle(new T.CatmullRomCurve3([end, mid, center]), 0);
    }
    const hill = this.bioTube(
      [center.clone().add(v(0.4, 0, 0)), v(-0.56, 0.03, 0), v(-0.32, -0.07, 0)],
      0.13,
      "Axon hillock",
      0xc98348,
    );
    this.addBioParticle(hill, 1);
    const axon = this.bioTube(
      [
        v(-0.5, 0, 0),
        v(0.6, -0.15, 0.05),
        v(1.8, -0.35, 0.02),
        v(2.6, -0.2, 0),
      ],
      0.055,
      "Axon",
      0xdc903c,
    );
    this.addBioParticle(axon, 2);
    for (let i = 0; i < 5; i++) {
      const t = 0.1 + i * 0.175,
        pt = axon.getPoint(t);
      const next = axon.getPoint(t + 0.12);
      const sheath = new T.Mesh(
        new T.CapsuleGeometry(0.16, pt.distanceTo(next) - 0.13, 6, 16),
        new T.MeshPhysicalMaterial({
          color: 0xf0cea0,
          roughness: 0.33,
          clearcoat: 0.45,
        }),
      );
      sheath.position.copy(pt.clone().lerp(next, 0.5));
      sheath.quaternion.setFromUnitVectors(
        v(0, 1, 0),
        next.clone().sub(pt).normalize(),
      );
      this.addPart(sheath, "Myelin sheath");
      const node = new T.Mesh(
        new T.SphereGeometry(0.073, 12, 8),
        new T.MeshStandardMaterial({
          color: 0xcb7a31,
          emissive: 0xffab36,
          emissiveIntensity: 0.25,
        }),
      );
      node.position.copy(axon.getPoint(Math.min(0.98, t + 0.155)));
      this.addPart(node, "Nodes of Ranvier");
    }
    for (let i = 0; i < 5; i++) {
      const base = axon.getPoint(1),
        tip = v(3.35 + (i % 2) * 0.2, -1.05 + i * 0.44, Math.sin(i) * 0.23);
      const c = this.bioTube(
        [
          base,
          base
            .clone()
            .lerp(tip, 0.5)
            .add(v(0.05, (i - 2) * 0.08, 0.05)),
          tip,
        ],
        0.034,
        "Axon terminals",
      );
      this.addBioParticle(c, 3);
      const terminal = new T.Mesh(
        new T.SphereGeometry(0.095, 16, 12),
        new T.MeshStandardMaterial({ color: 0xe1a65c, roughness: 0.3 }),
      );
      terminal.position.copy(tip);
      this.addPart(terminal, "Axon terminals");
      const receptor = this.bioTube(
        [
          tip.clone().add(v(0.29, -0.16, 0)),
          tip.clone().add(v(0.21, 0, 0)),
          tip.clone().add(v(0.29, 0.16, 0)),
        ],
        0.028,
        "Synapses",
        0xb6ac83,
      );
      void receptor;
      this.addBioParticle(
        new T.CatmullRomCurve3([tip, tip.clone().add(v(0.21, 0, 0))]),
        4,
      );
    }
    const labelData: [string, T.Vector3][] = [
      ["Dendrites", v(-3.05, 2.2, 0)],
      ["Soma", v(-1.6, -0.85, 0)],
      ["Myelin sheath", v(0.8, 0.65, 0)],
      ["Axon terminals", v(2.95, 1.3, 0)],
    ];
    labelData.forEach(([text, p]) => {
      const l = this.text(text, p, 0.25, "#886143");
      this.root.add(l.sprite);
      this.labels.push(l);
    });
  }
  addBioParticle(curve: T.CatmullRomCurve3, phase: number) {
    const mesh = new T.Mesh(
      new T.SphereGeometry(phase === 4 ? 0.023 : 0.055, 10, 8),
      new T.MeshBasicMaterial({ color: 0xffb333 }),
    );
    this.root.add(mesh);
    this.bioParticles.push({ mesh, curve, phase });
  }
  clear() {
    const geos = new Set<T.BufferGeometry>(),
      mats = new Set<T.Material>(),
      textures = new Set<T.Texture>();
    this.root.traverse((o) => {
      const mesh = o as T.Mesh;
      if (mesh.geometry) geos.add(mesh.geometry);
      if (mesh.material)
        (Array.isArray(mesh.material)
          ? mesh.material
          : [mesh.material]
        ).forEach((m) => {
          mats.add(m);
          const map = (m as T.MeshBasicMaterial).map;
          if (map) textures.add(map);
        });
    });
    geos.forEach((g) => g.dispose());
    mats.forEach((m) => m.dispose());
    textures.forEach((t) => t.dispose());
    this.root.clear();
    this.nodes = [];
    this.links = [];
    this.pickables = [];
    this.labels = [];
    this.parts.clear();
    this.bioParticles = [];
  }
  update(state: SceneState) {
    this.state = state;
    const key =
      state.mode === "network"
        ? state.net!.sizes.join(",")
        : state.mode === "single"
          ? String(state.inputs.length)
          : "bio";
    if (key !== this.key) {
      this.clear();
      this.key = key;
      if (state.mode === "single") this.single();
      else if (state.mode === "network") this.network();
      else this.bio();
      this.fit();
    }
    this.labels.forEach((l) => {
      l.sprite.visible = state.labels !== false;
    });
    if (state.mode === "single") {
      const n = neuron(
        state.inputs,
        state.weights,
        state.bias,
        state.activation,
      );
      this.links.forEach((link, i) => {
        link.value =
          i === 0
            ? state.bias
            : i <= state.inputs.length
              ? n.terms[i - 1]
              : n.y;
        this.styleLink(
          link,
          i === 0
            ? state.bias
            : i <= state.inputs.length
              ? state.weights[i - 1]
              : n.y,
          false,
        );
      });
    }
    if (state.mode === "network") {
      const net = state.net!,
        t = forward(net, state.inputs);
      this.nodes.forEach((n) => {
        n.label?.set(t.a[n.layer][n.node].toFixed(2));
        const selected =
          state.selected?.layer === n.layer && state.selected.node === n.node;
        (n.rim.material as T.MeshBasicMaterial).opacity = selected ? 1 : 0.4;
        n.rim.scale.setScalar(selected ? 1.15 : 1);
        (n.core.material as T.MeshPhysicalMaterial).emissiveIntensity =
          0.04 + Math.min(1, Math.abs(t.a[n.layer][n.node])) * 0.25;
      });
      this.links.forEach((l) => {
        const w = net.weights[l.layer][l.destination][l.source];
        l.value = w * t.a[l.layer][l.source];
        const s = state.selected;
        const highlighted =
          !!s &&
          ((s.layer === l.layer && s.node === l.source) ||
            (s.layer === l.layer + 1 && s.node === l.destination));
        this.styleLink(l, w, highlighted);
      });
    }
  }
  styleLink(link: Link, w: number, highlighted: boolean) {
    const m = link.tube.material as T.MeshStandardMaterial;
    m.color.set(w < 0 ? negative : positive);
    m.emissive.copy(m.color);
    m.opacity = highlighted ? 0.9 : 0.12 + Math.min(1, Math.abs(w) / 2) * 0.45;
    m.emissiveIntensity = highlighted ? 0.8 : 0.2;
    (link.particle.material as T.MeshBasicMaterial).color.set(
      w < 0 ? 0xc3a0ef : 0x38d7ff,
    );
    link.particle.scale.setScalar(
      0.65 + Math.min(2, Math.abs(link.value)) * 0.4,
    );
  }
  animate = (now: number) => {
    if (this.disposed) return;
    this.frame = requestAnimationFrame(this.animate);
    if (!this.visible || document.hidden) return;
    const dt = Math.min(0.05, (now - this.last) / 1000 || 0);
    this.last = now;
    this.time += dt;
    this.controls.update();
    const active = this.state.running !== false && !this.state.reduced;
    this.links.forEach((link) => {
      link.particle.visible = active && Math.abs(link.value) > 0.0001;
      const phase = this.state.step;
      const p =
        (((this.time * (0.3 + Math.min(2, Math.abs(link.value)) * 0.13) +
          link.offset -
          link.layer * 0.2) %
          1) +
          1) %
        1;
      if (this.state.mode === "single") {
        const stage =
          phase !== undefined && phase >= 0
            ? phase
            : Math.floor(this.time * 0.65) % 6;
        const biasPath = link.layer === 0 && link.source === 0;
        link.particle.visible =
          active &&
          Math.abs(link.value) > 0.0001 &&
          ((stage < 2 && link.layer === 0 && !biasPath) ||
            (stage === 3 && biasPath) ||
            (stage === 5 && link.layer === 1));
        const core = this.nodes[0].core.material as T.MeshPhysicalMaterial;
        core.emissiveIntensity = stage === 2 ? 0.42 : stage === 4 ? 0.6 : 0.12;
      }
      link.particle.position.copy(link.curve.getPoint(p));
    });
    const phase = (this.time * 0.52) % 5;
    this.bioParticles.forEach((b, i) => {
      b.mesh.visible =
        active &&
        (this.state.bioFires !== false || b.phase === 0) &&
        Math.floor(phase) === b.phase;
      b.mesh.position.copy(b.curve.getPoint(((phase % 1) + i * 0.014) % 1));
    });
    this.renderer.render(this.scene, this.camera);
  };
  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    this.observer.disconnect();
    this.intersector.disconnect();
    this.controls.dispose();
    this.renderer.domElement.removeEventListener(
      "pointerdown",
      this.pointerDown,
    );
    this.renderer.domElement.removeEventListener("pointerup", this.pointerUp);
    this.renderer.domElement.removeEventListener(
      "pointermove",
      this.pointerMove,
    );
    this.renderer.domElement.removeEventListener(
      "webglcontextlost",
      this.contextLost,
    );
    this.clear();
    this.env.dispose();
    this.scene.traverse((o) => {
      if (o instanceof T.DirectionalLight) o.shadow.dispose();
    });
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.renderer.domElement.remove();
  }
}
function sub(n: number) {
  return String(n)
    .split("")
    .map((d) => "₀₁₂₃₄₅₆₇₈₉"[Number(d)])
    .join("");
}
