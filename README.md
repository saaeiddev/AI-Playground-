# AI Playground — Interactive 3D Neural Network Laboratory

An interactive, scientifically grounded teaching tool for exploring artificial neurons and training a real neural network directly in your browser.

**Creator:** Amir Saeid Dehghan  
**Repository:** [saaeiddev/AI-Playground-](https://github.com/saaeiddev/AI-Playground-)  
**GitHub Pages URL (repository name includes trailing hyphen):** https://saaeiddev.github.io/AI-Playground-/

## Features

- A luminous blue glass **3D artificial neuron** inspired by the biological-vs-artificial-neuron reference, including a summation/activation core, floating input cells, weighted connections, animated signal pulses, light and refraction. Drag to rotate; use mouse wheel to zoom.
- A built-in **2D fallback** when WebGL or the Three.js CDN cannot load. This prevents a blank site when WebGL is disabled.
- Real, from-scratch **fully connected feed-forward neural networks** with 1–3 hidden layers, 2–10 neurons/layer, tanh/ReLU/sigmoid activations and binary sigmoid output.
- Real **binary cross-entropy loss**, analytic **backpropagation**, mini-batch optimization via **Adam or SGD**.
- **XOR, circle and spiral** binary classification datasets; reproducible seeded initialization and data.
- Live decision-boundary plot and labeled data points, **loss curve**, training accuracy, epochs and topology/connection weights.
- Play/pause, single-step, reset, learning-rate and training-speed controls; live neuron input inspector showing actual learned weights, bias, weighted sum and activation.
- Responsive, accessible dark glass UI. No login, APIs, tracking or server computation.

## Run locally

This site is plain modern HTML, CSS and JavaScript. Use a local HTTP server (ES modules should not be loaded using `file://`):

```bash
python3 -m http.server 8080
# then open http://localhost:8080/
```

An internet connection is used only for the pinned Three.js 3D library and optional Google Fonts. The full neural network and learning charts work independently of those external resources. If external scripts are blocked, the neuron falls back to an illustrated glass sphere.

## Test

```bash
node tests/smoke.mjs
```

Tests cover dataset generation, XOR training convergence, optimizer combinations and finite predictions.

## How training works

The model calculates a forward pass, uses sigmoid at the binary output layer, measures binary cross-entropy, calculates gradients with the backpropagation chain rule, and updates actual weights in every mini-batch. Plots and the inspector use those **same learned parameters**, not invented demonstration metrics. The epoch counter counts training examples processed divided by the dataset size.

## Deployment

On every push to `main`, GitHub Actions runs the smoke tests and publishes the static site to GitHub Pages through `.github/workflows/pages.yml`. The workflow requests automatic Pages enablement.

**Important:** The existing repository is actually named `AI-Playground-`, with a final hyphen; therefore GitHub Pages uses `/AI-Playground-/`, not `/AI-Playground/`. If your repository Pages configuration was previously restricted, go to **Settings → Pages → Build and deployment** and set **Source: GitHub Actions**, then rerun the workflow.

## Source structure

- `index.html` — interface, semantic controls and diagrams
- `styles.css` — responsive premium visual system
- `neural.mjs` — actual MLP, datasets and learning algorithms
- `app.mjs` — training interactions, live graphs, WebGL neuron
- `tests/smoke.mjs` — tests runnable with Node.js 22
- `.github/workflows/pages.yml` — test/deploy pipeline

© 2026 Amir Saeid Dehghan. Educational project.
