---
theme: seriph
background: https://cover.sli.dev
title: 'Lecture 1 — Introduction to Neural Networks and Deep Learning'
info: |
  ## Advanced Topics in Artificial Intelligence
  Lecture 1: Introduction to Neural Networks and Deep Learning

  Free adaptation, in English, of the *15.773 Hands-on Deep Learning* material
  (MIT OpenCourseWare, Farias & Ramakrishnan, 2024) for the course
  **Advanced Topics in AI** — Prof. Ronaldo Prati (UFABC).
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
fonts:
  sans: 'Inter'
  serif: 'Source Serif Pro'
  mono: 'JetBrains Mono'
---

# Lecture 1

## Introduction to Neural Networks and *Deep Learning*

<div class="pt-12">
  <span class="px-2 py-1 rounded cursor-pointer" hover:bg="white op-10">
    Advanced Topics in Artificial Intelligence · UFABC
  </span>
</div>

<div class="abs-br m-6 text-sm opacity-60">
  Adapted from MIT 15.773 (Farias, Ramakrishnan) — OCW
</div>

---
layout: two-cols
---

# Prerequisites

<v-clicks>

- Intermediate-level familiarity with **Python**
- Knowledge of **fundamental Machine Learning concepts**:
  - train / validation / test
  - *overfitting* and *underfitting*
  - regularization
- Good intuition with **linear algebra and calculus**
  (vectors, matrices, partial derivatives)
- Willingness to **write code** and experiment

</v-clicks>

::right::

<div class="mt-12 ml-8">

```mermaid {scale: 0.8}
flowchart TD
  P[Python] --> ML[Basic ML]
  ML --> DL[Deep Learning]
  M[Mathematics] --> ML
  M --> DL
  C[Code + experiments] --> DL
  style DL fill:#4f46e5,stroke:#312e81,color:#fff
```

</div>

---
layout: section
---

# Course philosophy

Concepts before mathematics · Hands-on before theory

---

# How we will approach the content

<v-clicks>

- **Focus on the key ideas** that underpin *Deep Learning*
- Mathematics appears **when it helps**, not as an obstacle
- Learning DL is like **learning to swim**: you can't do it just by watching
  - We will write, train, and debug real models
- The goal is not to train ML engineers, but to give you the autonomy
  to build a **V1.0 model** without depending on others

</v-clicks>

<div class="mt-8 text-sm opacity-70" v-click>

> If you are looking for a strongly theoretical/mathematical approach, there are other more suitable courses.

</div>

---
layout: center
class: text-center
---

# AI, ML, DL, and Generative AI

Before diving into neural networks, let's understand the relationships between these ideas.

<div class="mt-8">
  <AIHierarchy />
</div>

---

# Artificial intelligence was born in **1956**

<div class="grid grid-cols-2 gap-8 mt-6">

<div>

<v-clicks>

- The term **"Artificial Intelligence"** was coined at a historic workshop held at **Dartmouth College** (summer of 1956)
- It brought together names such as **John McCarthy, Marvin Minsky, Claude Shannon, Allen Newell, Herbert Simon**
- It defined an optimistic agenda: machines that could learn, reason, and use language
- Since then, AI has gone through several **winters** and **revivals**

</v-clicks>

</div>

<div>
  <Timeline />
</div>

</div>

---

# The classical AI approach

<div class="grid grid-cols-2 gap-8 mt-4">

<div class="text-lg">

<v-clicks>

- **Goal**: give the computer the ability to perform tasks that only humans did well
- **Classical strategy**:
  ask *human experts* how they do it,
  transcribe into **IF…THEN rules**, program them explicitly
- Worked in **some well-delimited domains**
  (expert systems, rule-based chess)

</v-clicks>

</div>

<div>

```mermaid {scale: 0.85}
flowchart LR
  E[Human expert] -->|interview| R[IF-THEN Rules]
  R --> P[Program]
  D[Data] --> P
  P --> S[Output]
  style R fill:#fbbf24,stroke:#92400e
```

</div>
</div>

---

# Why is this so hard?

<div class="mt-8 text-xl">

<v-click>

> *"We know more than we can tell."*
> — **Polanyi's Paradox**

</v-click>

</div>

<div class="mt-6">

<v-clicks>

- We recognize a face, ride a bicycle, identify sarcasm —
  but it is **very hard to write the rules** that describe how we do it
- Explicit rules don't cover **edge cases** nor generalize to new situations
- Result: symbolic AI ran into the **complexity of the real world**

</v-clicks>

</div>

---
layout: center
---

# Paradigm shift

<div class="text-2xl mt-6 max-w-3xl mx-auto text-center">

Instead of **telling** the computer what to do…

<div class="mt-8" v-click>

…<strong class="text-indigo-400">show</strong> it <em>many examples</em> of input and output,
and let <strong class="text-indigo-400">statistical techniques</strong> learn the relationship.

</div>

</div>

<div class="mt-12 text-center" v-click>
  <span class="px-4 py-2 bg-indigo-500/20 rounded-lg text-indigo-200">
    This is Machine Learning
  </span>
</div>

---

# Machine Learning, in one figure

<div class="mt-4">
  <MLDiagram />
</div>

<div class="mt-4 text-sm opacity-80">

Classical algorithms: linear regression, logistic regression, trees, *random forests*,
*gradient boosting*, SVMs, shallow neural networks…

</div>

---

# ML shines with **structured data**

<div class="grid grid-cols-2 gap-6 mt-4">

<div>

<v-clicks>

- Structured data = data that fits **naturally into a spreadsheet**
- Each column is a meaningful numerical/categorical *feature*
- Classical ML works **very well** in these cases:
  - credit *score*
  - demand forecasting
  - fraud detection
  - diagnosis based on laboratory tests

</v-clicks>

</div>

<div>

| age | income | owns_property | defaulted |
|----:|-------:|:-------------:|:---------:|
| 32  | 5.4k   | yes           | 0         |
| 47  | 3.1k   | no            | 1         |
| 25  | 2.8k   | no            | 0         |
| 51  | 9.2k   | yes           | 0         |

<div class="mt-2 text-xs opacity-60 text-center">Illustrative example (synthetic)</div>

</div>

</div>

---

# But what about **unstructured data**?

<div class="grid grid-cols-3 gap-6 mt-6 text-center">

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="text-4xl">🖼️</div>
  <div class="mt-2 font-bold">Images</div>
  <div class="text-sm opacity-70 mt-1">RGB pixels without isolated meaning</div>
</div>

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="text-4xl">📝</div>
  <div class="mt-2 font-bold">Text</div>
  <div class="text-sm opacity-70 mt-1">sequences of characters/tokens</div>
</div>

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="text-4xl">🔊</div>
  <div class="mt-2 font-bold">Audio</div>
  <div class="text-sm opacity-70 mt-1">raw temporal samples</div>
</div>

</div>

<div class="mt-8" v-click>

The **raw form** of this data has no intrinsic meaning for classical algorithms.
A pixel `(R=128, G=64, B=200)` says nothing on its own about whether there is
a cat in the image.

</div>

---

# The difficulty of unstructured data

<div class="mt-4">
  <PixelGrid />
</div>

<div class="mt-2 text-sm opacity-70 text-center">
A classifier <em>does not see</em> the cat — it sees a matrix of numbers.
</div>

---

# The pre-DL solution: **feature engineering**

<div class="grid grid-cols-2 gap-6">

<div>

<v-clicks>

- Experts designed **handcrafted representations** of the data:
  - **SIFT**, **HOG**, **SURF** for images
  - **MFCC** for audio
  - **TF-IDF**, *bag-of-words* for text
- The extracted representation was then fed
  into a **classical** model (usually logistic regression!)

</v-clicks>

</div>

<div>

```mermaid {scale: 0.85}
flowchart LR
  R[Raw data<br/>image/text/audio] --> F[Feature<br/>engineering<br/>SIFT, HOG, MFCC...]
  F --> M[Classical model<br/>logistic, SVM...]
  M --> S[Prediction]
  style F fill:#f59e0b,stroke:#92400e,color:#000
```

</div>

</div>

<div class="mt-6" v-click>

This required **a great deal of human effort** — a *bottleneck* that drastically limited
the reach of ML on rich data such as images, speech, and natural language.

</div>

---
layout: center
class: text-center
---

# **Deep Learning** emerges

<div class="mt-6 text-xl max-w-3xl mx-auto">

Models that **learn the representation** directly from raw data —
eliminating the bottleneck of manual feature engineering.

</div>

<div class="mt-12">
  <AIHierarchy showDL />
</div>

---

# What does **DL do** that classical ML didn't?

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

<v-clicks>

- **Automatically extracts** useful representations from unstructured data
- These representations can feed even trivial models (a logistic regression
  on top already delivers impressive results)
- Solves the **human bottleneck** that limited ML on images, text, and audio

</v-clicks>

</div>

<div>

```mermaid {scale: 0.85}
flowchart LR
  R[Raw data] --> DL[Deep Learning]
  DL -->|learned<br/>representation| C[Classifier]
  C --> Y[Prediction]
  style DL fill:#4f46e5,stroke:#312e81,color:#fff
  style C fill:#10b981,stroke:#065f46,color:#fff
```

<div class="mt-6 text-sm opacity-80">
Feature engineering stops being <em>human</em>
and becomes <em>learned</em>.
</div>

</div>

</div>

---

# Why did it happen **now**?

<div class="grid grid-cols-3 gap-6 mt-6 text-center">

<div v-click class="p-6 rounded-lg bg-gradient-to-b from-blue-500/10 to-blue-700/20 border border-blue-500/30">
  <div class="text-5xl">💡</div>
  <div class="mt-3 font-bold text-lg">Algorithms</div>
  <div class="text-sm opacity-80 mt-2">
    ReLU, dropout, batch norm,
    convolutional and
    transformer architectures, optimizers
    (Adam, etc.)
  </div>
</div>

<div v-click class="p-6 rounded-lg bg-gradient-to-b from-emerald-500/10 to-emerald-700/20 border border-emerald-500/30">
  <div class="text-5xl">📊</div>
  <div class="mt-3 font-bold text-lg">Data</div>
  <div class="text-sm opacity-80 mt-2">
    Digitization of everything:
    photos, videos, social networks,
    sensors, logs.
    <br/>ImageNet, Common Crawl…
  </div>
</div>

<div v-click class="p-6 rounded-lg bg-gradient-to-b from-amber-500/10 to-amber-700/20 border border-amber-500/30">
  <div class="text-5xl">⚡</div>
  <div class="mt-3 font-bold text-lg">Compute</div>
  <div class="text-sm opacity-80 mt-2">
    GPUs, then TPUs.
    Massive parallel training
    in reduced precision.
  </div>
</div>

</div>

<div class="mt-8 text-center text-lg" v-click>

…applied to an <strong>old</strong> idea: <strong>artificial neural networks</strong>.

</div>

---

# Immediate application: **perception**

<div class="mt-4">

Each **sensor** can gain the ability to detect, recognize, and
classify what it is perceiving. *Coupling* DL to cameras, microphones, and
sensors creates qualitatively different products.

</div>

<div class="grid grid-cols-2 md:grid-cols-4 gap-3 mt-6 text-center">

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">📷</div>
  <div class="text-sm mt-1">Object detection</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🩺</div>
  <div class="text-sm mt-1">Medical imaging diagnosis</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🚗</div>
  <div class="text-sm mt-1">Autonomous driving</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🏭</div>
  <div class="text-sm mt-1">Industrial inspection</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🎤</div>
  <div class="text-sm mt-1">Speech recognition</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🔬</div>
  <div class="text-sm mt-1">Microscopy analysis</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🛰️</div>
  <div class="text-sm mt-1">Remote sensing</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🦅</div>
  <div class="text-sm mt-1">Bio-acoustics</div>
</div>

</div>

---
layout: center
---

# And what about **output**?

<div class="text-xl mt-4 max-w-3xl mx-auto text-center">

Classical DL easily predicted **structured outputs**
(a number, a label, a probability vector).

</div>

<div class="mt-8" v-click>

But **generating** text, images, audio, code? For a long time, that was
**difficult territory**.

</div>

---

# Outputs that ML/DL predicted well

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

<v-clicks>

- **A number**
  - probability of default
  - next week's demand
- **A few numbers**
  - distribution over 1,000 ImageNet classes
  - coordinates of a *bounding box*
- **A discrete label**
  - sentiment (positive/negative)
  - product category

</v-clicks>

</div>

<div>

```mermaid {scale: 0.85}
flowchart LR
  X[Input<br/>struct. or not] --> M[Deep Learning]
  M --> Y1[Number]
  M --> Y2[Class vector]
  M --> Y3[Bounding box]
  style M fill:#4f46e5,stroke:#312e81,color:#fff
```

</div>

</div>

---
layout: center
class: text-center
---

# Then came **Generative AI**

<div class="mt-6 max-w-3xl mx-auto text-xl">

Models capable of **producing** unstructured content:
images, text, code, audio, video.

</div>

<div class="mt-10">
  <AIHierarchy showGenAI />
</div>

---

# The $X \to Y$ "matrix" of Generative AI

<div class="mt-4 text-center text-lg">

Both **$X$** and **$Y$** can be text, image, audio, video, code…
and even **multimodal**.

</div>

<div class="mt-6">
  <XYMatrix />
</div>

---

# Summary: $X \to Y$

<div class="mt-8 text-center">

```mermaid {scale: 1.0}
flowchart LR
  X((X)) --> DL[Deep Learning] --> Y((Y))
  style DL fill:#4f46e5,stroke:#312e81,color:#fff
  style X fill:#f59e0b,stroke:#92400e,color:#000
  style Y fill:#10b981,stroke:#065f46,color:#fff
```

</div>

<div class="mt-8 text-center text-xl">

> All the current excitement about AI is, in essence,
> the success of **Deep Learning** (and the Generative AI that came on top of it).

</div>

---
layout: section
---

# What is a neural network?

Let's start from the beginning.

---

# Back to **logistic regression**

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

Logistic regression maps an input vector to a **probability**:

$$
p(y=1\mid \mathbf{x}) \;=\; \sigma\!\left(b + \sum_{i=1}^{n} w_i\, x_i\right)
$$

where $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ is the **sigmoid** function.

<v-click>

Let's look at this expression as a **network of mathematical operations**.

</v-click>

</div>

<div>
  <SigmoidPlot />
</div>

</div>

---

# Example: predicting who gets called for an interview

<div class="grid grid-cols-2 gap-6 mt-4">

<div class="text-base">

<v-clicks>

- **Inputs**:
  - GPA (grade point average)
  - Years of experience
- **Output**: probability of being called for an interview
  ($1$ = called, $0$ = not called)
- **Model**: logistic regression trained on historical data

</v-clicks>

</div>

<div>

| # | GPA | Exp. | Called |
|--:|----:|-----:|:------:|
| 1 | 3.27 | 1.93 | 0 |
| 2 | 3.37 | 0.07 | 0 |
| 3 | 3.57 | 1.91 | 0 |
| 4 | 3.91 | 4.35 | 0 |
| 6 | 3.90 | 2.41 | 1 |
| 7 | 3.94 | 3.00 | 1 |
| 15 | 3.77 | 2.06 | 1 |
| ... | ... | ... | ... |

</div>

</div>

---

# Suppose we trained and obtained:

<div class="mt-4 text-center text-2xl">

$$
p(\text{called}) = \sigma\!\big(0{,}4 + 0{,}2\cdot\text{GPA} + 0{,}5\cdot\text{Exp}\big)
$$

</div>

<div class="mt-8">

Rewriting as a **network** with flow from left to right:

</div>

<div class="mt-4">
  <LogRegNetwork />
</div>

---

# Predicting with the "network"

<div class="mt-2">
Consider a candidate with <strong>GPA = 3.8</strong> and <strong>1.2 years</strong> of experience.
</div>

<div class="mt-4">
  <LogRegNetwork :gpa="3.8" :exp="1.2" showValues />
</div>

<div class="mt-4 text-sm">

$$
\sigma(0{,}4 + 0{,}2\cdot 3{,}8 + 0{,}5\cdot 1{,}2) \;=\; \sigma(1{,}76) \;\approx\; 0{,}85
$$

The estimated probability of being called for an interview is about **85%**.

</div>

---

# Basic vocabulary

<div class="mt-6">
  <NeuralNetwork
    :layers="[3, 1]"
    :labels="['Inputs', 'Output']"
    showWeights
  />
</div>

<div class="mt-6 grid grid-cols-2 gap-x-12 gap-y-2 text-base">

<div><strong class="text-indigo-300">Weights</strong> — multipliers on the connections between neurons</div>
<div><strong class="text-indigo-300">Bias</strong> — additive term (intercept) at each neuron</div>
<div><strong class="text-indigo-300">Neuron</strong> — unit that computes $\sum + \text{bias}$ and applies activation</div>
<div><strong class="text-indigo-300">Layer</strong> — vertical stack of neurons</div>

</div>

---

# Why is the "network lens" useful?

<div class="mt-4 text-lg max-w-3xl mx-auto">

<v-clicks>

- It allows us to **transform** the input data before the final decision
- Instead of feeding raw data directly to logistic regression,
  we can first **learn a better representation** of it
- This is the heart of *Deep Learning*: stacking learned transformations

</v-clicks>

</div>

---

# Stacking linear functions

<div class="mt-4 max-w-4xl mx-auto">

Before the final decision, **insert** a layer of linear functions
that combines the inputs:

</div>

<div class="mt-4">
  <NeuralNetwork :layers="[3, 3, 1]" :labels="['Input', 'Hidden', 'Output']" />
</div>

<div class="mt-4 text-sm opacity-80 text-center">

Each node in the intermediate layer is $z_j = b_j + \sum_i w_{ji} x_i$
— a **linear transformation** of the inputs.

</div>

---

# Stacking even more

<div class="mt-2">
  <NeuralNetwork :layers="[3, 3, 3, 1]" :labels="['Input', 'Hidden 1', 'Hidden 2', 'Output']" />
</div>

<div class="mt-4 max-w-3xl mx-auto">

We can stack **as many layers** as we want. At the end, we feed
logistic regression (sigmoid) with the transformed vector.

</div>

<div class="mt-4 text-amber-300 text-center" v-click>

⚠ But with only <em>linear</em> layers the composition remains linear —
we need something more.

</div>

---

# **Non-linear** activation functions

<div class="mt-4 max-w-3xl mx-auto">

For stacking layers to make a difference, each neuron applies a
**non-linear activation function** after the linear combination:

$$
a_j = \phi\!\Big(b_j + \sum_i w_{ji}\, x_i\Big)
$$

</div>

<div class="mt-6">
  <ActivationFunctions />
</div>

---

# Common activation functions

<div class="grid grid-cols-3 gap-4 mt-4">

<div class="p-4 rounded-lg bg-slate-800/40">

**Sigmoid**

$\sigma(z) = \dfrac{1}{1+e^{-z}}$

<div class="text-xs opacity-70 mt-2">
Output in (0,1). Used in the <strong>output layer</strong> for binary classification. Suffers from <em>vanishing gradients</em> in deep layers.
</div>

</div>

<div class="p-4 rounded-lg bg-slate-800/40">

**Tanh**

$\tanh(z) = \dfrac{e^z - e^{-z}}{e^z + e^{-z}}$

<div class="text-xs opacity-70 mt-2">
Output in (−1,1), centered at zero. Better behaved than sigmoid, but still saturates.
</div>

</div>

<div class="p-4 rounded-lg bg-slate-800/40">

**ReLU**

$\mathrm{ReLU}(z) = \max(0,\, z)$

<div class="text-xs opacity-70 mt-2">
Modern standard in hidden layers. Cheap, avoids saturation for z &gt; 0, enables much deeper networks.
</div>

</div>

</div>

<div class="mt-6 text-center text-sm opacity-80">
Others: Leaky ReLU, ELU, GELU, Swish, Softmax (in multi-class output)…
</div>

---

# Visual notation

<div class="mt-6 max-w-3xl mx-auto">

From here on, we will abbreviate each neuron with a **circle**, indicating
the activation by color/label:

</div>

<div class="mt-6 flex justify-center gap-12">

<div class="text-center">
  <div class="w-20 h-20 rounded-full bg-emerald-500/30 border-2 border-emerald-400 flex items-center justify-center text-2xl mx-auto">+</div>
  <div class="mt-2 text-sm">Linear</div>
</div>

<div class="text-center">
  <div class="w-20 h-20 rounded-full bg-amber-500/30 border-2 border-amber-400 flex items-center justify-center text-2xl mx-auto">+</div>
  <div class="mt-2 text-sm">ReLU</div>
</div>

<div class="text-center">
  <div class="w-20 h-20 rounded-full bg-indigo-500/30 border-2 border-indigo-400 flex items-center justify-center text-2xl mx-auto">+</div>
  <div class="mt-2 text-sm">Sigmoid</div>
</div>

</div>

---

# Anatomy of a DNN

<div class="mt-2">
  <NeuralNetwork
    :layers="[4, 5, 5, 1]"
    :labels="['Input', 'Hidden 1', 'Hidden 2', 'Output']"
    animate
  />
</div>

<div class="mt-6 grid grid-cols-2 gap-x-10 gap-y-1 text-base max-w-4xl mx-auto">

<div><strong>Input layer:</strong> the variables <em>x₁, …, xₖ</em></div>
<div><strong>Hidden layers:</strong> transform the representation</div>
<div><strong>Output layer:</strong> produces the prediction</div>
<div><strong>Dense connections:</strong> each neuron connects to all neurons in the next layer</div>

</div>

<div class="mt-4 text-center text-sm opacity-80">
When all layers are dense, we say the network is <em>fully connected</em>.
</div>

---

# Designing a DNN: the choices

<div class="mt-6 max-w-3xl mx-auto">

When you design a feedforward (*vanilla*) neural network, you decide:

</div>

<div class="mt-6 grid grid-cols-2 gap-4">

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="font-bold text-indigo-300">How many hidden layers?</div>
  <div class="text-sm opacity-80 mt-1">depth</div>
</div>

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="font-bold text-indigo-300">How many neurons per layer?</div>
  <div class="text-sm opacity-80 mt-1">width</div>
</div>

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="font-bold text-indigo-300">Which activation in the hidden layers?</div>
  <div class="text-sm opacity-80 mt-1">typically ReLU/GELU</div>
</div>

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="font-bold text-indigo-300">Which activation in the output?</div>
  <div class="text-sm opacity-80 mt-1">depends on the task</div>
</div>

</div>

<div class="mt-6 max-w-3xl mx-auto text-sm opacity-80" v-click>

The choice of output activation is guided by the nature of $y$:
linear (regression), sigmoid (binary classification), softmax (multi-class).

</div>

---

# Applying to the interview classifier

<v-clicks>

- **Inputs**: 2 variables (GPA, experience)
- **Output**: probability $\in (0,1)$
- **Design decisions**:
  - 1 hidden layer with **3 ReLU neurons**
  - **Sigmoid** at the output (probability)

</v-clicks>

<div class="mt-4" v-click>
  <NeuralNetwork
    :layers="[2, 3, 1]"
    :labels="['Input', 'Hidden (ReLU)', 'Output (σ)']"
  />
</div>

---

# How many parameters does this network have?

<div class="mt-4">
  <NeuralNetwork
    :layers="[2, 3, 1]"
    :labels="['Input', 'Hidden', 'Output']"
    showCount
  />
</div>

<div class="mt-4 max-w-3xl mx-auto" v-click>

- From input to hidden: $2 \times 3 = 6$ weights $+ 3$ biases $= 9$
- From hidden to output:   $3 \times 1 = 3$ weights $+ 1$ bias $= 4$
- **Total**: <strong class="text-indigo-300">13 parameters</strong>

</div>

---

# Trained weights (assumption)

<div class="mt-2">
  <InterviewNet />
</div>

<div class="mt-4 text-sm opacity-80 text-center">

How these weights are <em>found</em> from data is the topic
of the <strong>next lecture</strong> (training, loss, backpropagation).

</div>

---

# *Forward pass* — computing a prediction

<div class="mt-2">

For a candidate with $x_1 = 2{,}3$ (GPA) and $x_2 = 10{,}2$ (experience*):

</div>

<div class="mt-4">
  <InterviewNet :x1="2.3" :x2="10.2" showActivations />
</div>

<div class="mt-3 text-xs opacity-60 text-center">
*Exaggerated value just to make the calculations didactic.
</div>

---

# Hidden layer: explicit calculations

<div class="mt-2 text-center text-base">

$$
\begin{aligned}
a_1 &= \mathrm{ReLU}\!\big(-0{,}3 + 0{,}5\cdot 2{,}3 + 0{,}1\cdot 10{,}2\big) = \max(0;\,1{,}87) = 1{,}87\\
a_2 &= \mathrm{ReLU}\!\big( \;\;0{,}2 - 0{,}1\cdot 2{,}3 + 0{,}3\cdot 10{,}2\big) = \max(0;\,3{,}03) = 3{,}03\\
a_3 &= \mathrm{ReLU}\!\big( \;\;0{,}5 + 0{,}2\cdot 2{,}3 - 0{,}1\cdot 10{,}2\big) = \max(0;\,-0{,}06) = 0
\end{aligned}
$$

</div>

<div class="mt-6 text-center" v-click>

And the output:

$$
y = \sigma\!\big(0{,}05 + (-0{,}2)\cdot 1{,}87 + (-0{,}3)\cdot 3{,}03 + (-0{,}15)\cdot 0\big)
\approx \sigma(-1{,}23) \approx 0{,}226
$$

</div>

<div class="mt-3 text-center text-amber-300" v-click>

For this candidate, the network estimates ≈ <strong>22.6%</strong> chance of being called.

</div>

---

# The network as a function

<div class="mt-4 max-w-4xl mx-auto">

The network's prediction can be written as **a single function** of the inputs:

$$
\hat y(\mathbf{x}) \;=\; \sigma\!\Big( \mathbf{w}^{(2)\top}\, \mathrm{ReLU}\big(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\big) + b^{(2)} \Big)
$$

</div>

<div class="mt-6 grid grid-cols-2 gap-6">

<div v-click>

**Pure logistic regression:**

$$
\hat y \;=\; \sigma\!\big(0{,}4 + 0{,}2 x_1 + 0{,}5 x_2\big)
$$

linear in the data before the sigmoid.

</div>

<div v-click>

**Network with 1 hidden layer:**

a much richer non-linear composition —
capable of representing complex relationships between $\mathbf{x}$ and $y$.

</div>

</div>

---

# Summary: a DNN

<div class="mt-2">
  <NeuralNetwork
    :layers="[4, 6, 6, 6, 2]"
    :labels="['Input', 'Hidden 1', 'Hidden 2', 'Hidden 3', 'Output']"
    animate
  />
</div>

<div class="mt-4 grid grid-cols-2 gap-8 max-w-4xl mx-auto text-sm">

<div>

- **Feedforward** (vanilla): data flows from left to right
- **Architecture** = layers + activations + connections
- **Depth** comes from many hidden layers

</div>

<div>

- *Deep Learning* is, in essence, **neural networks with many layers**
- Training, backpropagation, and optimization — next lecture

</div>

</div>

---
layout: center
class: text-center
---

# Recap

<div class="mt-6 grid grid-cols-2 gap-6 max-w-4xl mx-auto text-left text-base">

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">Hierarchy of ideas</strong>
<div class="mt-1">AI ⊃ ML ⊃ DL ⊃ Generative AI</div>
</div>

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">Why DL is special</strong>
<div class="mt-1">learns representations from unstructured data</div>
</div>

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">The combination that unlocked it</strong>
<div class="mt-1">algorithms + data + GPUs</div>
</div>

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">Neural network ≈ regression</strong>
<div class="mt-1">logistic stacked with non-linearities</div>
</div>

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">Components</strong>
<div class="mt-1">layers, neurons, weights, biases, activations</div>
</div>

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">Forward pass</strong>
<div class="mt-1">linear combination → non-linearity → repeat</div>
</div>

</div>

---

# Next lecture

<div class="mt-8 grid grid-cols-2 gap-8 max-w-4xl mx-auto">

<div>

<v-clicks>

- How do you **train** a neural network?
- What is a **loss function**?
- **Gradient descent** and its variants
- **Backpropagation** (intuition + a numerical example)

</v-clicks>

</div>

<div>

```mermaid {scale: 0.75}
flowchart TD
  D[Data] --> F[Forward pass]
  F --> L[Loss]
  L --> B[Backward pass]
  B --> U[Update weights]
  U --> F
  style L fill:#ef4444,stroke:#7f1d1d,color:#fff
  style B fill:#10b981,stroke:#065f46,color:#fff
```

</div>

</div>

---
layout: center
class: text-center
---

# Thank you! Questions?

<div class="mt-6 text-sm opacity-70">

Freely adapted from <em>15.773 Hands-on Deep Learning</em>
(MIT OpenCourseWare, 2024) — original English material by Vivek Farias and
Rama Ramakrishnan, distributed under the terms of MIT OCW.

</div>

<div class="mt-2 text-xs opacity-60">
For more information: https://ocw.mit.edu/terms
</div>
