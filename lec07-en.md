---
theme: seriph
background: https://cover.sli.dev
title: 'Lecture 7 — Pretraining and BERT'
info: |
  ## Advanced Topics in Artificial Intelligence
  Lecture 7: Self-Supervised Learning and BERT

  Course **Advanced Topics in AI** — Prof. Ronaldo Prati (UFABC).
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

# Lecture 7

## Pretraining and BERT

<div class="pt-12">
  <span class="px-2 py-1 rounded cursor-pointer" hover:bg="white op-10">
    Advanced Topics in Artificial Intelligence · UFABC
  </span>
</div>

---

# Lecture outline

<div class="grid grid-cols-2 gap-6 mt-5 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40">

**Part 1 — Representations as Knowledge**
Neural networks as representation learners

</div>

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40">

**Part 2 — Self-Supervised Learning**
Creating labels from the data itself

</div>

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40">

**Part 3 — BERT**
Architecture, pretraining, and fine-tuning

</div>

</div>

<div class="space-y-3">

<div class="p-3 rounded bg-cyan-900/30 border border-cyan-500/40">

**Part 4 — HuggingFace Hub**
Ecosystem of pretrained models

</div>

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40">

**Part 5 — Transformers Beyond NLP**
ViT, TabTransformer, and multimodality

</div>

</div>

</div>

---

# The Transformer transforms embeddings

<div class="text-sm text-slate-300 mb-3 italic">
"Jacob <strong class="text-amber-300 not-italic">Station</strong> lives nearby the train <strong class="text-sky-300 not-italic">station</strong> close to the radio <strong class="text-emerald-300 not-italic">station</strong>"
</div>

<div class="grid gap-x-3 gap-y-2 text-xs items-center" style="grid-template-columns: 7rem 1fr 1fr 7rem 1fr">

<div></div>
<div class="font-semibold text-blue-300 text-center">Init. embed.<br><span class="text-slate-500 font-normal">token → vector</span></div>
<div class="font-semibold text-violet-300 text-center">+ Positional enc.<br><span class="text-slate-500 font-normal">sine / cosine</span></div>
<div class="font-semibold text-orange-300 text-center text-center">Heads<br><span class="text-slate-500 font-normal">4 perspectives</span></div>
<div class="font-semibold text-emerald-300 text-center">Contextualized<br><span class="text-slate-500 font-normal">W_O + FFN</span></div>

<div class="text-center py-1 px-2 rounded bg-amber-900/40 border border-amber-500/40 font-mono text-amber-300 leading-tight">station[1]<br><span class="text-slate-400 text-xs">Jacob</span></div>
<div class="h-8 rounded" style="background:linear-gradient(to right,#3b82f6 0%,#ef4444 17%,#3b82f6 30%,#ef4444 44%,#3b82f6 57%,#ef4444 71%,#3b82f6 84%,#ef4444 100%)"></div>
<div class="h-8 rounded" style="background:linear-gradient(to right,#7c3aed 0%,#3b82f6 13%,#ef4444 26%,#3b82f6 39%,#ef4444 53%,#3b82f6 66%,#ef4444 80%,#7c3aed 100%)"></div>
<div class="grid grid-cols-4 gap-0.5 h-8">
<div class="rounded-sm" style="background:linear-gradient(to bottom,#ef4444,#f97316)"></div>
<div class="rounded-sm" style="background:linear-gradient(to bottom,#3b82f6,#ef4444)"></div>
<div class="rounded-sm" style="background:linear-gradient(to bottom,#f97316,#3b82f6)"></div>
<div class="rounded-sm" style="background:linear-gradient(to bottom,#ef4444,#f97316)"></div>
</div>
<div class="h-8 rounded" style="background:linear-gradient(to right,#ef4444 0%,#f97316 22%,#ef4444 40%,#f97316 57%,#ef4444 72%,#f97316 87%,#ef4444 100%)"></div>

<div class="text-center py-1 px-2 rounded bg-sky-900/40 border border-sky-500/40 font-mono text-sky-300 leading-tight">station[6]<br><span class="text-slate-400 text-xs">train</span></div>
<div class="h-8 rounded" style="background:linear-gradient(to right,#3b82f6 0%,#ef4444 17%,#3b82f6 30%,#ef4444 44%,#3b82f6 57%,#ef4444 71%,#3b82f6 84%,#ef4444 100%)"></div>
<div class="h-8 rounded" style="background:linear-gradient(to right,#3b82f6 0%,#ef4444 16%,#7c3aed 29%,#3b82f6 43%,#ef4444 57%,#7c3aed 71%,#3b82f6 85%,#ef4444 100%)"></div>
<div class="grid grid-cols-4 gap-0.5 h-8">
<div class="rounded-sm" style="background:linear-gradient(to bottom,#10b981,#3b82f6)"></div>
<div class="rounded-sm" style="background:linear-gradient(to bottom,#3b82f6,#10b981)"></div>
<div class="rounded-sm" style="background:linear-gradient(to bottom,#10b981,#3b82f6)"></div>
<div class="rounded-sm" style="background:linear-gradient(to bottom,#3b82f6,#10b981)"></div>
</div>
<div class="h-8 rounded" style="background:linear-gradient(to right,#10b981 0%,#3b82f6 22%,#10b981 40%,#3b82f6 57%,#10b981 72%,#3b82f6 86%,#10b981 100%)"></div>

<div class="text-center py-1 px-2 rounded bg-emerald-900/40 border border-emerald-500/40 font-mono text-emerald-300 leading-tight">station[11]<br><span class="text-slate-400 text-xs">radio</span></div>
<div class="h-8 rounded" style="background:linear-gradient(to right,#3b82f6 0%,#ef4444 17%,#3b82f6 30%,#ef4444 44%,#3b82f6 57%,#ef4444 71%,#3b82f6 84%,#ef4444 100%)"></div>
<div class="h-8 rounded" style="background:linear-gradient(to right,#ef4444 0%,#3b82f6 14%,#7c3aed 28%,#ef4444 43%,#3b82f6 58%,#7c3aed 72%,#ef4444 86%,#3b82f6 100%)"></div>
<div class="grid grid-cols-4 gap-0.5 h-8">
<div class="rounded-sm" style="background:linear-gradient(to bottom,#3b82f6,#10b981)"></div>
<div class="rounded-sm" style="background:linear-gradient(to bottom,#10b981,#3b82f6)"></div>
<div class="rounded-sm" style="background:linear-gradient(to bottom,#3b82f6,#10b981)"></div>
<div class="rounded-sm" style="background:linear-gradient(to bottom,#10b981,#ef4444)"></div>
</div>
<div class="h-8 rounded" style="background:linear-gradient(to right,#3b82f6 0%,#10b981 20%,#3b82f6 37%,#10b981 54%,#3b82f6 68%,#10b981 83%,#3b82f6 100%)"></div>

</div>

<div class="grid gap-x-3 mt-3 text-xs" style="grid-template-columns: 7rem 1fr 1fr 7rem 1fr">

<div></div>
<div class="p-1.5 rounded bg-blue-900/20 border border-blue-500/20 text-center text-slate-300">
Same token →<br><strong>same vector</strong>
</div>
<div class="p-1.5 rounded bg-violet-900/20 border border-violet-500/20 text-center text-slate-300">
Station[1] ≠ [6] ≠ [11]<br><strong>position encoded</strong>
</div>
<div class="p-1.5 rounded bg-orange-900/20 border border-orange-500/20 text-center text-slate-300">
Each head:<br><strong>one relation</strong>
</div>
<div class="p-1.5 rounded bg-emerald-900/20 border border-emerald-500/20 text-center text-slate-300">
Jacob ≠ train ≠ radio<br><strong>context integrated</strong>
</div>

</div>

<div class="text-right text-xs text-slate-600 mt-1 italic">illustrative bars — real values visualized in the notebook</div>

---
layout: section
---

# Part 1 — Representations as Knowledge

---

# Neural networks learn representations

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Each layer's output is a representation**

The original input is progressively transformed at every hidden layer.

Rather than a "final answer," each layer produces a new view of the data.

</div>

<div class="p-3 rounded bg-blue-900/20 border border-blue-500/30" v-click>

**Visual hierarchy in CNNs**

Early layers detect **lines and edges**

Middle layers detect **shapes and textures**

Deep layers detect **complex objects**

Final layer detects **semantic categories**

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Intuition:** A network trained to classify 1000 ImageNet categories did not memorize pixels — it learned a hierarchy of visual concepts. The intermediate representations encode general knowledge about the visual world.

</div>

---

# Transfer Learning: reusing representations

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Classic scenario**

1. Train ResNet on ImageNet (1.2M images, 1000 classes)
2. Remove the output layer
3. Use the rest as a **fixed encoder**
4. Add a new head for the target task

</div>

<div class="p-3 rounded bg-blue-900/20 border border-blue-500/30" v-click>

**Why does it work?**

With only 100 examples of bags vs shoes, the encoder already captured:
- shapes and contours
- textures and materials
- object parts

The new head only needs to learn to separate the new categories.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Transfer Learning flow**

```
ResNet (pretrained on ImageNet)
        ↓
  [remove last layer]
        ↓
  Representation encoder
        ↓
  Dense → softmax  ← train with 100 examples
```

Deep-layer representations generalize to related tasks without needing millions of labeled examples for the target task.

</div>

</div>

---

# For NLP: we need a text encoder

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**What we want**

An encoder that produces **rich representations** of text: capturing semantics, syntax, context, and relationships between words.

</div>

<div class="p-3 rounded bg-blue-900/20 border border-blue-500/30" v-click>

**The label problem**

Supervised training requires human-annotated labels.

Annotation is expensive and slow — and many tasks have very few examples available.

</div>

</div>

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40 mt-4 text-sm" v-click>

**The good news**

Raw text is available in practically unlimited quantities:

- Wikipedia (billions of words)
- Digital books (BooksCorpus, Project Gutenberg)
- Web pages (Common Crawl)

We can exploit this abundance without any human labels.

</div>

---
layout: section
---

# Part 2 — Self-Supervised Learning

---

# The costly label problem

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Traditional supervised learning**

Requires manually annotated (input, label) pairs.

Sentiment classification: a human reads a sentence and marks positive/negative.

NER: a human identifies each entity in the text.

**Cost:** hours of specialist work per thousands of examples.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**What we have in abundance**

Billions of unlabeled sentences: books, articles, forums, source code.

No human labels — but all the structure of language is there.

**Question:** can we learn useful representations without any labels?

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Self-Supervised Learning:** automatically create labels from the structure of the data itself. Use part of the input to predict another part of the same input.

</div>

---

# Masking: the central idea

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Masking procedure**

1. Take original text
2. Replace random tokens with [MASK]
3. Train model to predict the masked tokens
4. Labels = original tokens (extracted automatically)

No human annotation needed.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**Creating training pairs**

Masked input → Label = original token

Fully automatic generation from any text corpus.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Concrete example:**

Original: "The mission of the MIT Sloan School of Management is to develop principled, innovative leaders..."

Masked: "The [MASK] of the MIT Sloan School of [MASK] is to develop principled, innovative [MASK]..."

Labels: "mission", "Management", "leaders"

</div>

</div>

---

# Example: filling in the blanks

<div class="mt-4 text-sm">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40 mb-4" v-click>

**Input text (with masks):**

"The [MASK] of the MIT Sloan School of [MASK] is to develop principled, innovative [MASK] who improve the world and to generate ideas that advance management practice."

</div>

<div class="grid grid-cols-3 gap-4">

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**[MASK] 1**

To predict "mission," the model must understand that "The ___ of the MIT Sloan School" calls for a noun expressing institutional purpose.

Requires **semantics** and **syntactic structure**.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**[MASK] 2**

"MIT Sloan School of ___" — the model must know that schools have full names and that "Management" completes this one.

Requires **world knowledge**.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**[MASK] 3**

"develop principled, innovative ___" — the model infers that people (leaders/thinkers) complete the phrase.

Requires **semantic relations**.

</div>

</div>

</div>

---

# Why does masking work?

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**To fill in blanks, the model must:**

- Understand the meaning of surrounding words
- Capture long-range dependencies
- Model semantic and grammatical coherence
- Integrate bidirectional context (before and after)

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**Result**

The internal representations learned during masking automatically encode all this linguistic knowledge — with zero human labels.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**As a sequence labeling task:**

```
Masked input
      ↓
  Transformer Encoder
      ↓
  Per-token embedding
      ↓
  Dense(ReLU)
      ↓
  Softmax → predicted token
```

It is a sequence labeling problem: each masked token has a label (the original token). The model outputs a distribution over the vocabulary for each [MASK].

</div>

</div>

---
layout: section
---

# Part 3 — BERT

---

# BERT: overview

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**BERT**

Bidirectional Encoder Representations from Transformers

Devlin et al., 2018 — Google AI

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Architecture**

Transformer Encoder pretrained with:
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)

Corpus: Wikipedia + BooksCorpus (~3.3B words)

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Original variants**

BERT-base: 12 layers, 768 dim, 12 heads — 110M parameters

BERT-large: 24 layers, 1024 dim, 16 heads — 340M parameters

Pretraining: weeks on TPUs. Fine-tuning: minutes/hours on a single GPU.

**Core idea:** pretrain once on massive text, then quickly adapt to any task with a few labeled examples.

</div>

</div>

---

# BERT Architecture: encoder stack

<div class="grid grid-cols-2 gap-5 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2 rounded bg-slate-700/60 border border-slate-500/40 text-xs font-mono text-center">
[CLS] tok₁ tok₂ … tokₙ [SEP]
</div>

<div class="p-2 rounded bg-blue-900/30 border border-blue-500/30 text-xs">
<strong class="text-blue-300">Input embedding</strong><br>
Token emb. + Segment emb. + Position emb. (learnable)<br>
<span class="text-slate-400">→ element-wise sum → dim 768</span>
</div>

<div class="space-y-1">
<div class="p-1.5 rounded bg-amber-900/40 border border-amber-500/30 text-xs" v-click>
<strong class="text-amber-300">Encoder Block × 12 (base) / × 24 (large)</strong>
<div class="mt-1 grid grid-cols-2 gap-1 text-slate-300">
<div>① Multi-Head Self-Attention</div><div class="text-slate-500">12 heads, d_k = 64</div>
<div>② Add &amp; LayerNorm</div><div class="text-slate-500">residual + norm</div>
<div>③ FFN: Linear → GeLU → Linear</div><div class="text-slate-500">768 → 3072 → 768</div>
<div>④ Add &amp; LayerNorm</div><div class="text-slate-500">residual + norm</div>
</div>
</div>
<div class="p-1.5 rounded bg-amber-900/30 border border-amber-500/20 text-xs opacity-80">Encoder Block × 12 …</div>
<div class="p-1.5 rounded bg-amber-900/20 border border-amber-500/10 text-xs opacity-50">Encoder Block × 12 …</div>
</div>

<div class="p-2 rounded bg-slate-700/60 border border-slate-500/40 text-xs font-mono text-center">
h_CLS  h₁  h₂  …  hₙ  <span class="text-slate-400">(dim 768 each)</span>
</div>

</div>

<div class="space-y-3 text-xs" v-click>

<div class="p-2 rounded bg-amber-900/20 border border-amber-500/30">
<strong class="text-amber-300">BERT-base vs BERT-large</strong>
<table class="w-full mt-1 text-slate-300" style="border-collapse:collapse">
<tr class="border-b border-slate-600/40"><th class="text-left py-0.5 text-slate-400">Config</th><th class="text-right text-slate-400">base</th><th class="text-right text-slate-400">large</th></tr>
<tr><td class="py-0.5">Layers (L)</td><td class="text-right">12</td><td class="text-right">24</td></tr>
<tr><td class="py-0.5">Hidden size (H)</td><td class="text-right">768</td><td class="text-right">1024</td></tr>
<tr><td class="py-0.5">Heads (A)</td><td class="text-right">12</td><td class="text-right">16</td></tr>
<tr><td class="py-0.5">FFN dim</td><td class="text-right">3072</td><td class="text-right">4096</td></tr>
<tr class="border-t border-slate-600/40"><td class="py-0.5 font-semibold text-amber-300">Parameters</td><td class="text-right font-semibold text-amber-300">110M</td><td class="text-right font-semibold text-amber-300">340M</td></tr>
</table>
</div>

<div class="p-2 rounded bg-blue-900/20 border border-blue-500/30" v-click>
<strong class="text-blue-300">Three embeddings summed at input</strong>
<div class="mt-1 space-y-1 text-slate-300">
<div><span class="text-blue-400">Token:</span> lookup in 30k-token table (WordPiece)</div>
<div><span class="text-violet-400">Segment:</span> A or B — for sentence pairs</div>
<div><span class="text-emerald-400">Position:</span> learnable (≠ sine/cosine of original Transformer)</div>
</div>
</div>

<div class="p-2 rounded bg-slate-800/50 border border-slate-600/30" v-click>
<strong class="text-slate-300">Why GeLU in the FFN?</strong><br>
<span class="text-slate-400">GeLU (Gaussian Error Linear Unit) provides smoother gradients than ReLU and empirically converges better in large language models.</span>
</div>

</div>

</div>

---

# Fine-tuning BERT: strategies and hyperparameters

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Strategy 1 — Frozen encoder**

Freeze all BERT weights. Train only the classification head.

Advantage: fast, works with dozens of examples.

Disadvantage: does not adapt representations to the new task.

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Strategy 2 — Full fine-tuning**

Update all weights (BERT + head) with low lr (~2e-5).

Best performance. Requires care: high lr destroys pre-trained representations (**catastrophic forgetting**).

</div>

<div class="p-3 rounded bg-amber-800/20 border border-amber-400/30" v-click>

**Strategy 3 — Layer-wise LR decay**

Deeper layers = smaller lr. Output layers = larger lr.

Preserves general representations, adapts high-level ones.

</div>

</div>

<div class="space-y-3 text-sm">

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30" v-click>

**Recommended hyperparameters (Devlin et al., 2018)**

```
lr:           2e-5, 3e-5, 5e-5
batch size:   16, 32
epochs:       2, 3, 4
warmup:       10% of total steps
weight decay: 0.01
```

Fine-tuning on small datasets (<5k examples) rarely benefits from more than 3 epochs.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**When to use each strategy?**

| Available data | Strategy |
|---|---|
| < 100 examples | Frozen encoder |
| 100–5k examples | Full fine-tuning |
| > 5k examples | Full fine-tuning + decay |

</div>

</div>

</div>

---

# The [CLS] token: global representation

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Special token [CLS]**

Added at the beginning of every input sequence.

During pretraining, the [CLS] embedding learns to aggregate information from the entire sequence via self-attention.

By the end of the network, it represents the sentence as a whole — global context.

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**BERT input structure**

```text
[CLS] token1 token2 ... tokenN [SEP]
  ↓      ↓      ↓         ↓
 h_CLS  h_1   h_2  ...  h_N
```

h_CLS: global sentence embedding

h_i: contextualized embedding of token i

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Why does [CLS] work as a global representation?**

In the Transformer, all tokens communicate via self-attention at every layer. [CLS] carries no lexical meaning of its own, so it learns to aggregate information from all other tokens to fulfill the pretraining objective (NSP). After fine-tuning, it carries the semantic representation of the whole sentence.

</div>

---

# Sentence-BERT: beyond [CLS]

<div class="mt-2 text-sm">

The `[CLS]` embedding is great for classification, but **not ideal for semantic similarity between sentences**.

<div class="grid grid-cols-2 gap-3 mt-3 text-xs">

<div class="p-2 rounded bg-red-900/30 border border-red-500/30">

**Limitation of [CLS] for similarity**

Comparing N sentences with vanilla BERT requires N² inferences — infeasible for large-scale search. Moreover, cosine distance between `[CLS]` embeddings of two independent sequences **does not correlate well** with human semantic similarity.

</div>

<div class="p-2 rounded bg-emerald-900/30 border border-emerald-500/30" v-click>

**Solution: Sentence-BERT** *(Reimers & Gurevych, 2019)*

Fine-tunes BERT with a **siamese network** using sentence pairs labeled by similarity. Uses **mean pooling** (average of all token embeddings) instead of `[CLS]`.

Result: sentence embeddings directly comparable by cosine distance — N inferences, not N².

</div>

</div>

<div class="mt-3 p-2 rounded bg-slate-800/50 border border-slate-500/30 text-xs" v-click>

**Siamese architecture:**

```text
sentence A → BERT → mean pool → u ─┐
                                    ├─ concat(u, v, |u−v|) → similarity
sentence B → BERT → mean pool → v ─┘
       (shared weights)
```

Trained on NLI (entailment / contradiction / neutral).

</div>

<div class="mt-2 p-2 rounded bg-indigo-900/30 border border-indigo-500/30 text-xs" v-click>

**Why does mean pooling outperform `[CLS]`?** `[CLS]` is optimized for NSP — weak sentence-level supervision. Mean pooling preserves information from all tokens and, after fine-tuning with explicit similarity supervision, produces geometrically better-calibrated representations for cosine comparison.

</div>

</div>
---

# Fine-tuning for sequence classification

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Task: sentiment analysis**

Given a sentence, classify it as positive or negative.

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Fine-tuning architecture**

```
Input: [CLS] I loved this movie [SEP]
              ↓
    BERT Encoder (pretrained)
              ↓
     [CLS] embedding: h_CLS
              ↓
          Dense(ReLU)
              ↓
        sigmoid / softmax
              ↓
       positive / negative
```

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Fine-tuning strategies**

Head-only training: freeze BERT, train only Dense + softmax. Fast, works well with few examples.

Full fine-tuning: update all BERT weights together with the head. Better performance, needs more data and time.

</div>

</div>

---

# Fine-tuning for sequence labeling

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Tasks: NER, slot filling, POS tagging**

Each token receives a class label (e.g., person, organization, location, other).

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Fine-tuning architecture**

```text
[CLS] Barack  Obama  was  president [SEP]
        ↓       ↓     ↓       ↓
    BERT Encoder (pretrained)
        ↓       ↓     ↓       ↓
      h_1     h_2   h_3     h_4
        ↓       ↓     ↓       ↓
     Dense   Dense Dense   Dense
        ↓       ↓     ↓       ↓
     B-PER   I-PER   O       O
```

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**BERT's flexibility**

The same pretrained architecture handles both task types. Only the head added during fine-tuning changes:

- Classification: use h_CLS
- Labeling: use h_i for each token

</div>

</div>

---

# BERT variants

<div class="grid grid-cols-3 gap-4 mt-4 text-sm">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**RoBERTa**
(Liu et al., 2019 — Facebook)

Improvements over BERT:
- Trained longer
- Larger batch (8k samples)
- No NSP objective
- More training data

Better performance on NLU benchmarks.

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**DistilBERT**
(Sanh et al., 2019 — HuggingFace)

Knowledge distillation from BERT:
- 40% smaller
- 60% faster
- 97% of BERT's performance

Ideal for production with resource constraints.

</div>

<div class="p-3 rounded bg-amber-800/20 border border-amber-400/30" v-click>

**Multilingual BERT**
(Google, 2018)

- 104 languages
- Single model
- Trained on multilingual Wikipedia

Works for Portuguese, Spanish, Chinese, Arabic, etc. — no per-language fine-tuning needed.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Sentence Transformers (sbert.net):** BERT variants optimized for generating semantically similar sentence embeddings. Use contrastive training with sentence pairs. Ideal for semantic search, deduplication, and clustering.

</div>

---
layout: section
---

# Part 4 — HuggingFace Hub

---

# HuggingFace: the model ecosystem

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-cyan-900/30 border border-cyan-500/40" v-click>

**The Hub**

Central repository with 500k+ pretrained models, datasets, and demo spaces.

Models for NLP, vision, audio, multimodal — in dozens of languages.

Any researcher or company can publish and download models.

</div>

<div class="p-3 rounded bg-cyan-900/20 border border-cyan-500/30" v-click>

**Typical workflow**

1. Pick a model on the Hub for your task
2. Load tokenizer + model with AutoTokenizer / AutoModel
3. Add a task head (classification, NER, etc.)
4. Fine-tune with your few labeled examples

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Key concept**

For any text classification problem: use BERT (or a variant) as a pretrained encoder and train only the output head with your examples. No need for millions of data points — BERT already knows about language.

</div>

</div>

---

# Code: HuggingFace Transformers

```python {all|1-6|8-12|14-17}
from transformers import pipeline

# Sentiment analysis with BERT
classifier = pipeline("sentiment-analysis")
result = classifier("I loved the movie!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Fill-mask
fill = pipeline("fill-mask", model="bert-base-uncased")
fill("The capital of France is [MASK].")
# [{'token_str': 'paris', 'score': 0.9975}]

# NER
ner = pipeline("ner", aggregation_strategy="simple")
ner("Hugging Face is based in New York")
```

<div class="grid grid-cols-2 gap-4 mt-4 text-sm">

<div class="p-2 rounded bg-cyan-900/30 border border-cyan-500/40" v-click>

**pipeline()** automatically loads the model + tokenizer + post-processing for the chosen task. Just pass the task name.

</div>

<div class="p-2 rounded bg-cyan-900/20 border border-cyan-500/30" v-click>

**AutoTokenizer / AutoModelForSequenceClassification** allow loading any model from the Hub and adding custom heads for full fine-tuning.

</div>

</div>

---
layout: section
---

# Part 5 — Transformers Beyond NLP

---

# Vision Transformer (ViT)

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**Idea: treat an image as a sequence of patches**

1. Divide image into fixed-size patches (e.g., 16x16 pixels)
2. Flatten each patch into a vector
3. Linear projection → patch embedding
4. Add positional encoding
5. Feed into Transformer Encoder
6. Use [CLS] for classification

</div>

<div class="p-3 rounded bg-emerald-900/20 border border-emerald-500/30" v-click>

**Result**

Dosovitskiy et al., 2020 (Google Brain):

With sufficient pretraining data scale, ViT is competitive with or superior to CNNs for image classification.

Self-attention captures global dependencies that local convolutions cannot easily model.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**ViT flow:**

```
224x224 image
    ↓
Split into 16x16 patches → 196 patches
    ↓
Flatten + Linear Projection → 196 embeddings
    ↓
[CLS] + positional encoding
    ↓
Transformer Encoder (same as in NLP)
    ↓
h_CLS → Dense → image class
```

</div>

</div>

---

# TabTransformer: tabular data

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**The tabular data problem**

Categorical variables have complex interactions that linear models fail to capture.

Tree-based models capture interactions but do not generalize well with little data.

</div>

<div class="p-3 rounded bg-emerald-900/20 border border-emerald-500/30" v-click>

**TabTransformer architecture**
(Huang et al., 2020)

```
Categorical variables
       ↓
  Per-column embeddings
       ↓
  Transformer Encoder
  (captures column interactions)
       ↓
Numerical variables → concatenate
       ↓
    Dense layers → prediction
```

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Advantage:** the Transformer automatically discovers which combinations of columns matter for prediction — something manual feature engineering would try to replicate. Especially useful when categorical variables interact in non-obvious ways.

</div>

---

# Embeddings: the common language

<div class="grid grid-cols-3 gap-4 mt-4 text-sm">

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**Text**

Tokenize → per-token embedding

The same Transformer Encoder processes the sequence of embeddings.

</div>

<div class="p-3 rounded bg-emerald-900/20 border border-emerald-500/30" v-click>

**Image**

Split into patches → linear projection

Patches become tokens — same Transformer.

</div>

<div class="p-3 rounded bg-emerald-800/20 border border-emerald-400/30" v-click>

**Table / Audio / Graph**

Any data can be tokenized and projected into embeddings of the same dimension.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**The unifying power of embeddings**

The Transformer Encoder is domain-agnostic — it only needs a sequence of fixed-dimension vectors. What changes across domains is only the tokenization/projection module.

This enables **multimodal** models: combining text + image + table in the same architecture, mixing embeddings from different domains in the same sequence. CLIP, Flamingo, and GPT-4V are examples of this paradigm.

</div>

---

# Summary: what we learned today

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Representations as knowledge**

Deep neural networks learn representation hierarchies. Transfer learning reuses that knowledge for new tasks.

</div>

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Self-Supervised Learning**

Artificial labels created from data. Masking forces the model to learn language without human supervision.

</div>

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**BERT and fine-tuning**

Encoder pretrained on massive text. [CLS] token for classification; h_i per token for labeling. Adapt with few examples.

</div>

</div>

<div class="space-y-3">

<div class="p-3 rounded bg-cyan-900/30 border border-cyan-500/40" v-click>

**HuggingFace**

500k+ ready-to-use models. Pipeline in 3 lines. Workflow: pick model → tokenizer → head → fine-tune.

</div>

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**Transformers beyond NLP**

ViT (images as patches), TabTransformer (columns as tokens). Embeddings are the common language for multimodal data.

</div>

</div>

</div>
