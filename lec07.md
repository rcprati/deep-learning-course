---
theme: seriph
background: https://cover.sli.dev
title: 'Aula 7 — Pré-treinamento e BERT'
info: |
  ## Tópicos Avançados em Inteligência Artificial
  Aula 7: Self-Supervised Learning e BERT

  Disciplina **Tópicos Avançados em IA** — Prof. Ronaldo Prati (UFABC).
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

# Aula 7

## Pré-treinamento e BERT

<div class="pt-12">
  <span class="px-2 py-1 rounded cursor-pointer" hover:bg="white op-10">
    Tópicos Avançados em Inteligência Artificial · UFABC
  </span>
</div>

---

# Roteiro da aula

<div class="grid grid-cols-2 gap-6 mt-5 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40">

**Parte 1 — Representações como Conhecimento**
Redes neurais como aprendedores de representação

</div>

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40">

**Parte 2 — Self-Supervised Learning**
Criando labels a partir dos próprios dados

</div>

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40">

**Parte 3 — BERT**
Arquitetura, pré-treinamento e fine-tuning

</div>

</div>

<div class="space-y-3">

<div class="p-3 rounded bg-cyan-900/30 border border-cyan-500/40">

**Parte 4 — HuggingFace Hub**
Ecossistema de modelos pré-treinados

</div>

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40">

**Parte 5 — Transformers Além do NLP**
ViT, TabTransformer e multimodalidade

</div>

</div>

</div>

---

# O Transformer transforma embeddings

<div class="text-sm text-slate-300 mb-3 italic">
"Jacob <strong class="text-amber-300 not-italic">Station</strong> lives nearby the train <strong class="text-sky-300 not-italic">station</strong> close to the radio <strong class="text-emerald-300 not-italic">station</strong>"
</div>

<div class="grid gap-x-3 gap-y-2 text-xs items-center" style="grid-template-columns: 7rem 1fr 1fr 7rem 1fr">

<div></div>
<div class="font-semibold text-blue-300 text-center">Emb. inicial<br><span class="text-slate-500 font-normal">token → vetor</span></div>
<div class="font-semibold text-violet-300 text-center">+ Enc. posicional<br><span class="text-slate-500 font-normal">seno / cosseno</span></div>
<div class="font-semibold text-orange-300 text-center text-center">Cabeças<br><span class="text-slate-500 font-normal">4 perspectivas</span></div>
<div class="font-semibold text-emerald-300 text-center">Contextualizado<br><span class="text-slate-500 font-normal">W_O + FFN</span></div>

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
Mesmo token →<br><strong>mesmo vetor</strong>
</div>
<div class="p-1.5 rounded bg-violet-900/20 border border-violet-500/20 text-center text-slate-300">
Station[1] ≠ [6] ≠ [11]<br><strong>posição codificada</strong>
</div>
<div class="p-1.5 rounded bg-orange-900/20 border border-orange-500/20 text-center text-slate-300">
Cada cabeça:<br><strong>uma relação</strong>
</div>
<div class="p-1.5 rounded bg-emerald-900/20 border border-emerald-500/20 text-center text-slate-300">
Jacob ≠ trem ≠ rádio<br><strong>contexto integrado</strong>
</div>

</div>

<div class="text-right text-xs text-slate-600 mt-1 italic">barras ilustrativas — valores reais visualizados no notebook</div>

---
layout: section
---

# Parte 1 — Representações como Conhecimento

---

# Redes neurais aprendem representações

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**A saída de cada camada é uma representação**

A entrada original é transformada progressivamente a cada camada oculta.

Em vez de "resposta", cada camada produz uma nova visão dos dados.

</div>

<div class="p-3 rounded bg-blue-900/20 border border-blue-500/30" v-click>

**Hierarquia visual em CNNs**

Camadas iniciais detectam **linhas e bordas**

Camadas intermediárias detectam **formas e texturas**

Camadas profundas detectam **objetos complexos**

Camada final detecta **categorias semânticas**

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Intuição:** Uma rede treinada para classificar 1000 categorias ImageNet não decorou pixels — ela aprendeu uma hierarquia de conceitos visuais. As representações intermediárias codificam conhecimento geral sobre o mundo visual.

</div>

---

# Transfer Learning: aproveitando representações

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Cenário clássico**

1. Treinar ResNet em ImageNet (1.2M imagens, 1000 classes)
2. Remover a camada de saída
3. Usar o restante como **encoder fixo**
4. Adicionar nova cabeça para tarefa alvo

</div>

<div class="p-3 rounded bg-blue-900/20 border border-blue-500/30" v-click>

**Por que funciona?**

Com apenas 100 exemplos de bolsas vs sapatos, o encoder já capturou:
- formas e contornos
- texturas e materiais
- partes de objetos

A nova cabeça apenas aprende a distinguir as categorias novas.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Fluxo do Transfer Learning**

```
ResNet (pré-treinado ImageNet)
        ↓
  [remover última camada]
        ↓
  Encoder de representações
        ↓
  Dense → softmax  ← treinar com 100 exemplos
```

Representações de camadas profundas generalizam para tarefas relacionadas sem precisar de milhões de exemplos na tarefa alvo.

</div>

</div>

---

# Para NLP: precisamos de um encoder de texto

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**O que queremos**

Um encoder que produza **representações ricas** de texto: capture semântica, sintaxe, contexto, relações entre palavras.

</div>

<div class="p-3 rounded bg-blue-900/20 border border-blue-500/30" v-click>

**O problema das labels**

Para treinar com supervisão, precisamos de labels anotadas por humanos.

Anotação é cara e lenta — e muitas tarefas têm poucos exemplos disponíveis.

</div>

</div>

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40 mt-4 text-sm" v-click>

**A boa notícia**

Texto puro está disponível em quantidade praticamente ilimitada:

- Wikipedia (bilhões de palavras)
- Livros digitais (BooksCorpus, Project Gutenberg)
- Páginas da internet (Common Crawl)

Podemos explorar essa abundância sem precisar de labels humanas.

</div>

---
layout: section
---

# Parte 2 — Self-Supervised Learning

---

# O problema das labels custosas

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Aprendizado supervisionado tradicional**

Requer pares (entrada, label) anotados manualmente.

Classificar sentimento: humano lê frase e marca positivo/negativo.

NER: humano identifica cada entidade no texto.

**Custo:** horas de trabalho especializado por milhares de exemplos.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**O que temos em abundância**

Bilhões de frases não rotuladas: livros, artigos, fóruns, código-fonte.

Nenhuma label humana — mas toda a estrutura da linguagem está presente.

**Pergunta:** podemos aprender representações úteis sem labels?

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Self-Supervised Learning:** criar labels artificiais automaticamente a partir da estrutura dos próprios dados. Usar parte da entrada para prever outra parte da mesma entrada.

</div>

---

# Mascaramento: a ideia central

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Procedimento de mascaramento**

1. Pegar texto original
2. Substituir tokens aleatórios por [MASK]
3. Treinar modelo para prever os tokens mascarados
4. Labels = tokens originais (extraídos automaticamente)

Nenhuma anotação humana necessária.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**Criação de pares de treinamento**

Entrada mascarada → Label = token original

Geração totalmente automática a partir de qualquer corpus de texto.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Exemplo concreto:**

Original: "The mission of the MIT Sloan School of Management is to develop principled, innovative leaders..."

Mascarado: "The [MASK] of the MIT Sloan School of [MASK] is to develop principled, innovative [MASK]..."

Labels: "mission", "Management", "leaders"

</div>

</div>

---

# Exemplo: preenchendo lacunas

<div class="mt-4 text-sm">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40 mb-4" v-click>

**Texto de entrada (com máscaras):**

"The [MASK] of the MIT Sloan School of [MASK] is to develop principled, innovative [MASK] who improve the world and to generate ideas that advance management practice."

</div>

<div class="grid grid-cols-3 gap-4">

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**[MASK] 1**

Para prever "mission", o modelo precisa entender que "The ___ of the MIT Sloan School" pede um substantivo de propósito institucional.

Requer **semântica** e **estrutura sintática**.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**[MASK] 2**

"MIT Sloan School of ___" — o modelo precisa conhecer que escolas têm nomes completos e que "Management" completa esse nome.

Requer **conhecimento de mundo**.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**[MASK] 3**

"develop principled, innovative ___" — o modelo infere que pessoas (leaders/thinkers) completam a frase.

Requer **relações semânticas**.

</div>

</div>

</div>

---

# Por que o mascaramento funciona?

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Para preencher lacunas, o modelo precisa:**

- Entender o significado das palavras ao redor
- Capturar dependências de longa distância
- Modelar coerência semântica e gramatical
- Integrar contexto bidirecional (antes e depois)

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**Resultado**

As representações internas aprendidas durante o mascaramento codificam automaticamente todo esse conhecimento linguístico — sem nenhuma label humana.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Como tarefa de sequência:**

```
Entrada mascarada
      ↓
  Transformer Encoder
      ↓
  Embedding por token
      ↓
  Dense(ReLU)
      ↓
  Softmax → token previsto
```

É um problema de rotulagem de sequência: cada token mascarado tem um label (o token original). O modelo computa a distribuição sobre o vocabulário para cada [MASK].

</div>

</div>

---
layout: section
---

# Parte 3 — BERT

---

# BERT: visão geral

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**BERT**

Bidirectional Encoder Representations from Transformers

Devlin et al., 2018 — Google AI

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Arquitetura**

Transformer Encoder pré-treinado com:
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)

Corpus: Wikipedia + BooksCorpus (~3.3B palavras)

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Variantes originais**

BERT-base: 12 camadas, 768 dim, 12 heads — 110M parâmetros

BERT-large: 24 camadas, 1024 dim, 16 heads — 340M parâmetros

Pré-treinamento: semanas em TPUs. Fine-tuning: minutos/horas em GPU.

**Ideia central:** pré-treinar uma vez em texto massivo, depois adaptar rapidamente para qualquer tarefa com poucos exemplos rotulados.

</div>

</div>

---

# Arquitetura BERT: encoder stack

<div class="grid grid-cols-2 gap-5 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2 rounded bg-slate-700/60 border border-slate-500/40 text-xs font-mono text-center">
[CLS] tok₁ tok₂ … tokₙ [SEP]
</div>

<div class="p-2 rounded bg-blue-900/30 border border-blue-500/30 text-xs">
<strong class="text-blue-300">Embedding de entrada</strong><br>
Token emb. + Segment emb. + Position emb. (learnable)<br>
<span class="text-slate-400">→ soma elemento a elemento → dim 768</span>
</div>

<div class="space-y-1">
<div class="p-1.5 rounded bg-amber-900/40 border border-amber-500/30 text-xs" v-click>
<strong class="text-amber-300">Bloco Encoder × 12 (base) / × 24 (large)</strong>
<div class="mt-1 grid grid-cols-2 gap-1 text-slate-300">
<div>① Multi-Head Self-Attention</div><div class="text-slate-500">12 cabeças, d_k = 64</div>
<div>② Add &amp; LayerNorm</div><div class="text-slate-500">residual + norm</div>
<div>③ FFN: Linear → GeLU → Linear</div><div class="text-slate-500">768 → 3072 → 768</div>
<div>④ Add &amp; LayerNorm</div><div class="text-slate-500">residual + norm</div>
</div>
</div>
<div class="p-1.5 rounded bg-amber-900/30 border border-amber-500/20 text-xs opacity-80">Bloco Encoder × 12 …</div>
<div class="p-1.5 rounded bg-amber-900/20 border border-amber-500/10 text-xs opacity-50">Bloco Encoder × 12 …</div>
</div>

<div class="p-2 rounded bg-slate-700/60 border border-slate-500/40 text-xs font-mono text-center">
h_CLS  h₁  h₂  …  hₙ  <span class="text-slate-400">(dim 768 cada)</span>
</div>

</div>

<div class="space-y-3 text-xs" v-click>

<div class="p-2 rounded bg-amber-900/20 border border-amber-500/30">
<strong class="text-amber-300">BERT-base vs BERT-large</strong>
<table class="w-full mt-1 text-slate-300" style="border-collapse:collapse">
<tr class="border-b border-slate-600/40"><th class="text-left py-0.5 text-slate-400">Config</th><th class="text-right text-slate-400">base</th><th class="text-right text-slate-400">large</th></tr>
<tr><td class="py-0.5">Camadas (L)</td><td class="text-right">12</td><td class="text-right">24</td></tr>
<tr><td class="py-0.5">Dimensão (H)</td><td class="text-right">768</td><td class="text-right">1024</td></tr>
<tr><td class="py-0.5">Cabeças (A)</td><td class="text-right">12</td><td class="text-right">16</td></tr>
<tr><td class="py-0.5">FFN dim</td><td class="text-right">3072</td><td class="text-right">4096</td></tr>
<tr class="border-t border-slate-600/40"><td class="py-0.5 font-semibold text-amber-300">Parâmetros</td><td class="text-right font-semibold text-amber-300">110M</td><td class="text-right font-semibold text-amber-300">340M</td></tr>
</table>
</div>

<div class="p-2 rounded bg-blue-900/20 border border-blue-500/30" v-click>
<strong class="text-blue-300">Três embeddings somados na entrada</strong>
<div class="mt-1 space-y-1 text-slate-300">
<div><span class="text-blue-400">Token:</span> lookup na tabela de 30k tokens (WordPiece)</div>
<div><span class="text-violet-400">Segment:</span> A ou B — para pares de sentenças</div>
<div><span class="text-emerald-400">Position:</span> learnable (≠ senos/cossenos do Transformer original)</div>
</div>
</div>

<div class="p-2 rounded bg-slate-800/50 border border-slate-600/30" v-click>
<strong class="text-slate-300">Por que GeLU no FFN?</strong><br>
<span class="text-slate-400">GeLU (Gaussian Error Linear Unit) proporciona gradientes mais suaves que ReLU e empiricamente converge melhor em modelos de linguagem de grande escala.</span>
</div>

</div>

</div>

---

# Fine-tuning BERT: estratégias e hiperparâmetros

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Estratégia 1 — Encoder congelado**

Congela todos os pesos do BERT. Treina apenas a cabeça de classificação.

Vantagem: rápido, funciona com dezenas de exemplos.

Desvantagem: não adapta representações à nova tarefa.

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Estratégia 2 — Full fine-tuning**

Atualiza todos os pesos (BERT + cabeça) com lr baixo (~2e-5).

Melhor desempenho. Requer cuidado: lr alto destrói representações pré-treinadas (**catastrophic forgetting**).

</div>

<div class="p-3 rounded bg-amber-800/20 border border-amber-400/30" v-click>

**Estratégia 3 — Layer-wise LR decay**

Camadas mais profundas = lr menor. Camadas de saída = lr maior.

Preserva representações gerais, adapta representações de alto nível.

</div>

</div>

<div class="space-y-3 text-sm">

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30" v-click>

**Hiperparâmetros recomendados (Devlin et al., 2018)**

```
lr:           2e-5, 3e-5, 5e-5
batch size:   16, 32
epochs:       2, 3, 4
warmup:       10% dos passos totais
weight decay: 0.01
```

Fine-tuning em datasets pequenos (<5k exemplos) raramente beneficia de mais de 3 épocas.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**Quando usar cada estratégia?**

| Dados disponíveis | Estratégia |
|---|---|
| < 100 exemplos | Encoder congelado |
| 100–5k exemplos | Full fine-tuning |
| > 5k exemplos | Full fine-tuning + decay |

</div>

</div>

</div>

---

# O token [CLS]: representação global

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Token especial [CLS]**

Adicionado no início de cada sequência de entrada.

Ao longo do pré-treinamento, o embedding de [CLS] aprende a agregar informação de toda a sequência via self-attention.

Ao final, representa a sentença como um todo — contexto global.

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Estrutura de entrada BERT**

```text
[CLS] token1 token2 ... tokenN [SEP]
  ↓      ↓      ↓         ↓
 h_CLS  h_1   h_2  ...  h_N
```

h_CLS: embedding global da sentença

h_i: embedding contextualizado do token i

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Por que [CLS] funciona como representação global?**

No Transformer, todos os tokens se comunicam via self-attention em cada camada. O [CLS] não tem significado lexical próprio, então aprende a agregar informação de todos os outros tokens para cumprir o objetivo de pré-treinamento (NSP). Depois do fine-tuning, carrega a representação semântica da sentença inteira.

</div>

---

# Sentence-BERT: além do [CLS]

<div class="mt-2 text-sm">

O embedding de `[CLS]` é ótimo para classificação, mas **não é ideal para similaridade semântica entre sentenças**.

<div class="grid grid-cols-2 gap-3 mt-3 text-xs">

<div class="p-2 rounded bg-red-900/30 border border-red-500/30">

**Limitação do [CLS] para similaridade**

Comparar N sentenças com BERT puro exige N² inferências — inviável para busca em larga escala. Além disso, a distância coseno entre embeddings de `[CLS]` de duas sequências independentes **não correlaciona bem** com similaridade semântica humana.

</div>

<div class="p-2 rounded bg-emerald-900/30 border border-emerald-500/30" v-click>

**Solução: Sentence-BERT** *(Reimers & Gurevych, 2019)*

Treina BERT com uma **rede siamesa** usando pares de sentenças rotulados por similaridade. Usa **mean pooling** (média dos embeddings de todos os tokens) em vez de `[CLS]`.

Resultado: embeddings de sentença comparáveis diretamente por distância coseno — N inferências, não N².

</div>

</div>

<div class="mt-3 p-2 rounded bg-slate-800/50 border border-slate-500/30 text-xs" v-click>

**Arquitetura siamesa:**

```text
sentença A → BERT → mean pool → u ─┐
                                    ├─ concat(u, v, |u−v|) → similaridade
sentença B → BERT → mean pool → v ─┘
       (pesos compartilhados)
```

Treinado com NLI (entailment / contradiction / neutral).

</div>

<div class="mt-2 p-2 rounded bg-indigo-900/30 border border-indigo-500/30 text-xs" v-click>

**Por que mean pooling supera `[CLS]`?** O `[CLS]` é otimizado para NSP — supervisão fraca de nível de sentença. Mean pooling preserva informação de todos os tokens e, após fine-tuning com supervisão explícita de similaridade, produz representações geometricamente melhor calibradas para comparação por coseno.

</div>

</div>
---

# Fine-tuning para classificação de sequência

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Tarefa: análise de sentimento**

Dada uma frase, classificar como positivo ou negativo.

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Arquitetura de fine-tuning**

```
Entrada: [CLS] I loved this movie [SEP]
              ↓
    BERT Encoder (pré-treinado)
              ↓
     Embedding de [CLS]: h_CLS
              ↓
          Dense(ReLU)
              ↓
        sigmoid / softmax
              ↓
        positivo / negativo
```

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Estratégias de fine-tuning**

Treinar apenas a cabeça: BERT congelado, treinar somente Dense + softmax. Rápido, bom com poucos exemplos.

Full fine-tuning: atualizar todos os pesos do BERT junto com a cabeça. Melhor desempenho, precisa de mais dados e tempo.

</div>

</div>

---

# Fine-tuning para rotulagem de sequência

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Tarefas: NER, slot filling, POS tagging**

Cada token recebe uma classe (ex: pessoa, organização, local, outro).

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Arquitetura de fine-tuning**

```text
[CLS] Barack  Obama  foi  presidente [SEP]
        ↓       ↓     ↓       ↓
    BERT Encoder (pré-treinado)
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

**Flexibilidade do BERT**

A mesma arquitetura pré-treinada serve para ambas as tarefas. O que muda é apenas a cabeça adicionada no fine-tuning:

- Classificação: usar h_CLS
- Rotulagem: usar h_i de cada token

</div>

</div>

---

# Variações do BERT

<div class="grid grid-cols-3 gap-4 mt-4 text-sm">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**RoBERTa**
(Liu et al., 2019 — Facebook)

Melhorias sobre BERT:
- Treinado por mais tempo
- Batch maior (8k amostras)
- Remove NSP
- Mais dados de treinamento

Melhor desempenho em benchmarks de NLU.

</div>

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**DistilBERT**
(Sanh et al., 2019 — HuggingFace)

Destilação do conhecimento de BERT:
- 40% menor
- 60% mais rápido
- 97% do desempenho

Ideal para produção com restrição de recursos.

</div>

<div class="p-3 rounded bg-amber-800/20 border border-amber-400/30" v-click>

**BERT Multilingual**
(Google, 2018)

- 104 idiomas
- Um único modelo
- Treinado em Wikipedia multilíngue

Funciona para português, espanhol, chinês, árabe, etc. — sem fine-tuning por língua.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Sentence Transformers (sbert.net):** variantes de BERT otimizadas para gerar embeddings de sentenças semanticamente similares. Usam treinamento contrastivo com pares de sentenças. Ideais para busca semântica, deduplicação e clustering.

</div>

---
layout: section
---

# Parte 4 — HuggingFace Hub

---

# HuggingFace: o ecossistema de modelos

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-cyan-900/30 border border-cyan-500/40" v-click>

**O Hub**

Repositório central com 500k+ modelos pré-treinados, datasets e spaces de demonstração.

Modelos para NLP, visão, áudio, multimodal — em dezenas de idiomas.

Qualquer pesquisador ou empresa pode publicar e baixar modelos.

</div>

<div class="p-3 rounded bg-cyan-900/20 border border-cyan-500/30" v-click>

**Workflow típico**

1. Escolher modelo no Hub para sua tarefa
2. Carregar tokenizer + modelo com AutoTokenizer / AutoModel
3. Adicionar cabeça de tarefa (classificação, NER, etc.)
4. Fine-tuning com seus poucos exemplos rotulados

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Conceito chave**

Para qualquer problema de classificação de texto: use BERT (ou variante) como encoder pré-treinado e treine apenas a cabeça de saída com seus exemplos. Não precisa de milhões de dados — o BERT já sabe sobre linguagem.

</div>

</div>

---

# Código: HuggingFace Transformers

```python {all|1-6|8-12|14-17}
from transformers import pipeline

# Análise de sentimento com BERT
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

**pipeline()** carrega automaticamente modelo + tokenizer + pós-processamento para a tarefa escolhida. Basta passar o nome da tarefa.

</div>

<div class="p-2 rounded bg-cyan-900/20 border border-cyan-500/30" v-click>

**AutoTokenizer / AutoModelForSequenceClassification** permitem carregar qualquer modelo do Hub e adicionar cabeças personalizadas para fine-tuning completo.

</div>

</div>

---
layout: section
---

# Parte 5 — Transformers Além do NLP

---

# Vision Transformer (ViT)

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**Ideia: tratar imagem como sequência de patches**

1. Dividir imagem em patches de tamanho fixo (ex: 16x16 pixels)
2. Achatar cada patch em um vetor
3. Projeção linear → embedding de patch
4. Adicionar positional encoding
5. Alimentar Transformer Encoder
6. Usar [CLS] para classificação

</div>

<div class="p-3 rounded bg-emerald-900/20 border border-emerald-500/30" v-click>

**Resultado**

Dosovitskiy et al., 2020 (Google Brain):

Com escala suficiente de dados de pré-treinamento, ViT é competitivo ou superior a CNNs em classificação de imagens.

Self-attention captura dependências globais que convoluções locais não capturam facilmente.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 text-sm" v-click>

**Fluxo do ViT:**

```
Imagem 224x224
    ↓
Dividir em patches 16x16 → 196 patches
    ↓
Achatar + Linear Projection → 196 embeddings
    ↓
[CLS] + positional encoding
    ↓
Transformer Encoder (igual ao de NLP)
    ↓
h_CLS → Dense → classe da imagem
```

</div>

</div>

---

# TabTransformer: dados tabulares

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**Problema com dados tabulares**

Variáveis categóricas têm interações complexas que modelos lineares não capturam bem.

Árvores capturam interações mas não generalizam com poucos dados.

</div>

<div class="p-3 rounded bg-emerald-900/20 border border-emerald-500/30" v-click>

**Arquitetura TabTransformer**
(Huang et al., 2020)

```
Variáveis categóricas
       ↓
  Embeddings por coluna
       ↓
  Transformer Encoder
  (captura interações entre colunas)
       ↓
Variáveis numéricas → concatenar
       ↓
    Dense layers → predição
```

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Vantagem:** o Transformer captura automaticamente quais combinações de colunas são relevantes para a predição — algo que engenharia de features manual tentaria replicar. Especialmente útil quando variáveis categóricas interagem de forma não óbvia.

</div>

---

# Embeddings: a linguagem comum

<div class="grid grid-cols-3 gap-4 mt-4 text-sm">

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**Texto**

Tokenizar → embedding por token

O mesmo Transformer Encoder processa a sequência de embeddings.

</div>

<div class="p-3 rounded bg-emerald-900/20 border border-emerald-500/30" v-click>

**Imagem**

Dividir em patches → projeção linear

Patches viram tokens — mesmo Transformer.

</div>

<div class="p-3 rounded bg-emerald-800/20 border border-emerald-400/30" v-click>

**Tabela / Áudio / Grafo**

Qualquer dado pode ser tokenizado e projetado em embeddings de mesma dimensão.

</div>

</div>

<div class="p-3 rounded bg-slate-800/50 border border-slate-600/30 mt-4 text-sm" v-click>

**Poder unificador dos embeddings**

O Transformer Encoder é agnóstico ao tipo de dado — só precisa de uma sequência de vetores de dimensão fixa. O que muda entre domínios é apenas o módulo de tokenização/projeção.

Isso habilita modelos **multimodais**: combinar texto + imagem + tabela na mesma arquitetura, misturando embeddings de diferentes domínios na mesma sequência. CLIP, Flamingo, GPT-4V são exemplos desse paradigma.

</div>

---

# Resumo: o que aprendemos hoje

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Representações como conhecimento**

Redes neurais profundas aprendem hierarquias de representação. Transfer learning reutiliza esse conhecimento em novas tarefas.

</div>

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Self-Supervised Learning**

Labels artificiais criadas a partir dos dados. Mascaramento força o modelo a aprender linguagem sem supervisão humana.

</div>

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**BERT e fine-tuning**

Encoder pré-treinado em texto massivo. Token [CLS] para classificação; h_i por token para rotulagem. Adaptar com poucos exemplos.

</div>

</div>

<div class="space-y-3">

<div class="p-3 rounded bg-cyan-900/30 border border-cyan-500/40" v-click>

**HuggingFace**

500k+ modelos prontos. Pipeline em 3 linhas. Workflow: escolher modelo → tokenizer → cabeça → fine-tune.

</div>

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**Transformers além do NLP**

ViT (imagens como patches), TabTransformer (colunas como tokens). Embeddings são a linguagem comum para dados multimodais.

</div>

</div>

</div>
