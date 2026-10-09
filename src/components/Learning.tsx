import { useState } from "react";
import { ChevronRight, BookOpen, Check } from "lucide-react";
import Biology from "./Biology";
const lessons = [
  {
    group: "Biology",
    title: "Dendrites & synapses",
    body: "Dendrites receive inputs at synapses. Chemical neurotransmitters change ion-channel activity in the receiving cell, producing graded postsynaptic potentials. Some inputs are excitatory and others inhibitory.",
    equation: "Synaptic input → graded membrane potential",
    note: "A synapse is a junction between cells. A signal does not travel through the nucleus.",
  },
  {
    group: "Biology",
    title: "Soma & signal integration",
    body: "The cell body supports metabolism and integrates signals arriving across the dendrites and soma. The nucleus contains genetic material; it is not a relay station for electrical signals.",
    equation: "Excitatory + inhibitory inputs → integration",
    note: "The 3D model is a simplified multipolar neuron. Real cells vary greatly in shape and connectivity.",
  },
  {
    group: "Biology",
    title: "Axon & action potentials",
    body: "In many neurons, sufficient depolarization initiates an action potential in the axon initial segment near the axon hillock. Voltage-gated channels regenerate this electrical signal along the axon.",
    equation: "Threshold reached → action potential",
    note: "An action potential is approximately all-or-none. Stronger input often changes firing rate, rather than spike amplitude. The −55 mV threshold in the explorer is illustrative.",
  },
  {
    group: "Biology",
    title: "Myelin & transmission",
    body: "Myelin insulates sections of the axon. Action potentials are regenerated at the nodes of Ranvier, enabling saltatory conduction. At a chemical synapse, calcium entry triggers neurotransmitter release from terminal vesicles.",
    equation: "Axon → terminal → neurotransmitter → next cell",
    note: "The moving lights are a teaching device. They are not literal particles or a physically timed simulation of ion flow.",
  },
  {
    group: "Mathematics",
    title: "Inputs, weights & bias",
    body: "An artificial neuron receives numbers. Each input xᵢ is multiplied by a weight wᵢ. A positive weight increases the weighted sum for a positive input; a negative weight decreases it. Bias shifts the sum.",
    equation: "z = w₁x₁ + w₂x₂ + … + wₙxₙ + b",
    note: "x₀ = 1 with weight b is another way to write the bias. Our calculation adds b exactly once.",
  },
  {
    group: "Mathematics",
    title: "Activation functions",
    body: "The activation converts the weighted sum into the neuron’s output. Sigmoid maps to (0, 1), tanh to (−1, 1), ReLU keeps positive values, and leaky ReLU uses a slope of 0.01 for negative values. Linear returns z unchanged.",
    equation: "y = f(z)",
    note: "Step returns 1 when z ≥ 0 and 0 otherwise. Its derivative is zero away from zero and undefined at zero, so ordinary backpropagation cannot use it effectively.",
  },
  {
    group: "Networks",
    title: "Layers & forward propagation",
    body: "A dense layer connects every neuron in the previous layer to every neuron in the next. Its outputs become the next layer’s inputs. Forward propagation applies these calculations in order.",
    equation: "a⁽ˡ⁾ = f(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)",
    note: "Information flows from input to output. This laboratory uses feedforward networks, without recurrent connections.",
  },
  {
    group: "Training",
    title: "Predictions & loss",
    body: "For binary classification, a sigmoid output gives a score between 0 and 1. Binary cross-entropy compares that score with the target label. A score of at least 0.5 is classified as class 1.",
    equation: "L = −mean[t log(p) + (1−t) log(1−p)]",
    note: "The implementation uses an equivalent stable formula from logits. A low training loss does not establish performance on unseen data.",
  },
  {
    group: "Training",
    title: "Backpropagation",
    body: "Backpropagation applies the chain rule from the output layer backward. It calculates how each weight and bias affects the loss. It computes derivatives; the optimizer then changes parameters.",
    equation: "∂L/∂wᵢⱼ = δⱼ · aᵢ",
    note: "For sigmoid with binary cross-entropy, the output-layer derivative with respect to the logit is p − t, adjusted for averaging.",
  },
  {
    group: "Training",
    title: "Gradient descent & Adam",
    body: "Gradient descent moves parameters against the loss gradient. Adam adapts each update using moving averages of gradients and squared gradients, with bias correction. This laboratory uses full-batch Adam.",
    equation: "θ ← θ − η · m̂ / (√v̂ + ε)",
    note: "The learning rate η controls update size. Training can stall or become unstable; more layers and larger rates do not guarantee better results.",
  },
];
export default function Learning({ reduced }: { reduced: boolean }) {
  const [index, setIndex] = useState(0),
    lesson = lessons[index];
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">LEARNING CENTER</span>
          <h1>A little intuition. A lot of possibility.</h1>
          <p>From a living neuron to the mathematics of learning.</p>
        </div>
        <span className="status-pill">
          <BookOpen size={15} />
          10 short lessons
        </span>
      </div>
      <div className="learning-layout">
        <aside className="panel lesson-nav">
          {lessons.map((l, i) => (
            <button
              className={i === index ? "active" : ""}
              key={l.title}
              onClick={() => setIndex(i)}
            >
              <span className="lesson-number">
                {i < index ? (
                  <Check size={14} />
                ) : (
                  String(i + 1).padStart(2, "0")
                )}
              </span>
              <span>
                <small>{l.group}</small>
                {l.title}
              </span>
              <ChevronRight size={15} />
            </button>
          ))}
        </aside>
        <article className="panel lesson">
          <span className="eyebrow">
            {lesson.group} · LESSON {index + 1} / 10
          </span>
          <h2>{lesson.title}</h2>
          {lesson.group === "Biology" && <Biology reduced={reduced} />}
          <p>{lesson.body}</p>
          <div className="lesson-equation">{lesson.equation}</div>
          <div className="science-note">
            <BookOpen size={18} />
            <p>{lesson.note}</p>
          </div>
          <div className="lesson-actions">
            <button
              disabled={index === 0}
              onClick={() => setIndex((i) => i - 1)}
            >
              Previous lesson
            </button>
            <button
              className="primary"
              disabled={index === lessons.length - 1}
              onClick={() => setIndex((i) => i + 1)}
            >
              Next lesson
            </button>
          </div>
          <div className="analogy">
            <h3>Inspired by biology. Built from mathematics.</h3>
            <p>
              Artificial neurons are useful mathematical abstractions. They have
              no cell nucleus, axon, neurotransmitter, or biological action
              potential. The glass model visualizes a computation, not a
              microscopic object.
            </p>
          </div>
          <div className="sources">
            <h4>Read further</h4>
            <a
              href="https://www.ncbi.nlm.nih.gov/books/NBK26910/"
              target="_blank"
              rel="noreferrer"
            >
              Molecular Biology of the Cell · Electrical properties of membranes
            </a>
            <a
              href="https://www.deeplearningbook.org/contents/mlp.html"
              target="_blank"
              rel="noreferrer"
            >
              Goodfellow, Bengio & Courville · Deep feedforward networks
            </a>
            <a
              href="https://arxiv.org/abs/1412.6980"
              target="_blank"
              rel="noreferrer"
            >
              Kingma & Ba · Adam: A Method for Stochastic Optimization
            </a>
          </div>
        </article>
      </div>
    </>
  );
}
