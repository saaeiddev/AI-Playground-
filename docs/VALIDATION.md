# Validation record

Validated on 9 October 2026. This record distinguishes automated checks from browser observations and from hardware testing that was not available.

## Automated validation

`npm test` passes **52 tests** in two files:

- `src/engine/neural.test.ts`: 48 tests for all six activation functions, summation, bias, input and shape validation, seeded initialization, multilayer/multi-output forward propagation, parameter edits, stable binary cross-entropy, analytic gradients, Adam state and genuine training.
- `src/graphics/NeuralScene.test.ts`: four tests use actual Three.js geometry and mathematical classes with a mocked renderer. They check finite vertices, all three scene modes, biological structures, graph architecture changes, mathematical/visual synchronization, selection and resource disposal. An animation-path negative-modulo defect found by these tests was corrected.

Backpropagation gradients are checked against central finite differences for every weight and bias with sigmoid, ReLU, Leaky ReLU, tanh and linear hidden activations. Step is correctly excluded from gradient training.

Real optimizer tests reduce loss on AND, OR and XOR to less than 5% of initial loss and reach 100% training accuracy in the tested seeded configuration. Separate 8-by-8 hidden-layer runs on Linear, Circle and Spiral reduce loss below 30% of the initial value and exceed 90% training accuracy. These are assertions on newly computed results, not prerecorded trajectories or promises for every architecture/learning rate.

`npm run build` performs strict TypeScript checking and produces the static Vite bundle, including the training worker and local fonts. The GitHub Actions deployment repeats installation, tests and production build before publishing.

## Browser interaction checks

The actual application was exercised in cloud Chrome:

| Check | Observed result |
| --- | --- |
| Single-neuron defaults | Weighted sum 0.9, bias −0.2, z = 0.7, sigmoid output approximately 0.66818777. |
| Input, negative weight and activation edits | With x₁ = 2, w₁ = −1 and linear activation, output became −1.86. Adding/removing an input recalculated it; reset restored the defaults. |
| Network editing | Bias edits, adding an output neuron, and selecting the second output in the inspector updated the working network. |
| Genuine XOR training | Initial loss 0.72936 fell to 2.55e−5 by epoch 2,520 in the observed run. A prior longer run reached 100% training accuracy. |
| Pause acknowledgement | After the Resume button appeared, epoch stayed at 2,520 across subsequent observations. Resume advanced it to 2,700 before another acknowledged pause. |
| Reset | Reset returned epoch to 0 and sampled new parameters, with loss 0.77884 in that run. |
| Mobile-width layout | All five workspaces were checked inside a 390 × 844 viewport. Available body width and scroll width were both 375 CSS pixels; no horizontal document overflow remained. |
| Learning and accessibility | Lesson selection and Next navigated from Backpropagation to Gradient descent & Adam. Reduced motion could be checked through its accessible control. |
| WebGL failure behavior | Graphics initialization failure displayed the explicit retry panel; mathematical controls, network editing, worker training, charts and lessons continued working. |

Pause uses a worker acknowledgement and revision counter to prevent an obsolete queued training snapshot from restarting the visible running state. Reset and architecture/dataset changes initialize a fresh worker/model rather than applying stale results.

## Testing limits

The cloud browser reports its GL renderer as **Disabled** and cannot create a WebGL 2 context. Consequently, no hardware-rendered screenshot, glass/refraction comparison, raycast interaction verification or FPS measurement is claimed from this environment. Geometry tests validate the scene construction but cannot validate GPU shader appearance. The application retains its actual interactive Three.js scenes; the failure panel is not a replacement illustration.

The phone-width check validates responsive CSS, not actual iPhone Safari or Android device behavior. Physical touch gestures and GPU performance on Windows, macOS, iOS and Android still require testing on those devices. The biological illustration intentionally simplifies anatomy and electrochemical dynamics; it is not a physiological simulator.

## Publication

The production workflow targets the existing `saaeiddev/AI-Playground-` repository and its GitHub Pages site at <https://saaeiddev.github.io/AI-Playground-/>. A successful build alone is not proof of deployment; verify the Pages workflow result and the public site's loaded assets when publishing.
