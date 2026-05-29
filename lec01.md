---
theme: seriph
background: https://cover.sli.dev
title: 'Aula 1 — Introdução a Redes Neurais e Deep Learning'
info: |
  ## Tópicos Avançados em Inteligência Artificial
  Aula 1: Introdução a Redes Neurais e Deep Learning

  Adaptação livre, em português, do material *15.773 Hands-on Deep Learning*
  (MIT OpenCourseWare, Farias & Ramakrishnan, 2024) para a disciplina
  **Tópicos Avançados em IA** — Prof. Ronaldo Prati (UFABC).
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

# Aula 1

## Introdução a Redes Neurais e *Deep Learning*

<div class="pt-12">
  <span class="px-2 py-1 rounded cursor-pointer" hover:bg="white op-10">
    Tópicos Avançados em Inteligência Artificial · UFABC
  </span>
</div>

<div class="abs-br m-6 text-sm opacity-60">
  Adaptado de MIT 15.773 (Farias, Ramakrishnan) — OCW
</div>

---
layout: two-cols
---

# Pré-requisitos

<v-clicks>

- Familiaridade com **Python** em nível intermediário
- Conhecimento de **conceitos fundamentais de Machine Learning**:
  - treino / validação / teste
  - *overfitting* e *underfitting*
  - regularização
- Boa intuição com **álgebra linear e cálculo**
  (vetores, matrizes, derivadas parciais)
- Disposição para **escrever código** e experimentar

</v-clicks>

::right::

<div class="mt-12 ml-8">

```mermaid {scale: 0.8}
flowchart TD
  P[Python] --> ML[ML básico]
  ML --> DL[Deep Learning]
  M[Matemática] --> ML
  M --> DL
  C[Código + experimentos] --> DL
  style DL fill:#4f46e5,stroke:#312e81,color:#fff
```

</div>

---
layout: section
---

# Filosofia do curso

Conceitos antes da matemática · Mão na massa antes da teoria

---

# Como vamos abordar o conteúdo

<v-clicks>

- **Foco nas ideias-chave** que sustentam o *Deep Learning*
- A matemática aparece **quando ajuda**, não como obstáculo
- Aprender DL é como **aprender a nadar**: não dá só assistindo
  - Vamos escrever, treinar e depurar modelos reais
- O objetivo não é formar engenheiros de ML, mas dar autonomia
  para que vocês construam um **modelo V1.0** sem depender de terceiros

</v-clicks>

<div class="mt-8 text-sm opacity-70" v-click>

> Se você procura uma abordagem fortemente teórica/matemática, há outras disciplinas mais adequadas.

</div>

---
layout: center
class: text-center
---

# IA, ML, DL e IA Generativa

Antes de mergulhar nas redes neurais, vamos entender as relações entre essas ideias.

<div class="mt-8">
  <AIHierarchy />
</div>

---

# A inteligência artificial nasceu em **1956**

<div class="grid grid-cols-2 gap-8 mt-6">

<div>

<v-clicks>

- O termo **"Inteligência Artificial"** foi cunhado em uma oficina histórica realizada na **Dartmouth College** (verão de 1956)
- Reunia nomes como **John McCarthy, Marvin Minsky, Claude Shannon, Allen Newell, Herbert Simon**
- Definia uma agenda otimista: máquinas que aprendessem, raciocinassem e usassem linguagem
- Desde então, a IA passou por vários **invernos** e **renascimentos**

</v-clicks>

</div>

<div>
  <Timeline />
</div>

</div>

---

# A abordagem clássica de IA

<div class="grid grid-cols-2 gap-8 mt-4">

<div class="text-lg">

<v-clicks>

- **Meta**: dar ao computador a capacidade de fazer tarefas que só humanos faziam bem
- **Estratégia clássica**:
  perguntar a *especialistas humanos* como eles fazem,
  transcrever em **regras `SE…ENTÃO`**, programá-las explicitamente
- Funcionou em **alguns domínios bem delimitados**
  (sistemas especialistas, xadrez baseado em regras)

</v-clicks>

</div>

<div>

```mermaid {scale: 0.85}
flowchart LR
  E[Especialista humano] -->|entrevista| R[Regras SE-ENTÃO]
  R --> P[Programa]
  D[Dado] --> P
  P --> S[Saída]
  style R fill:#fbbf24,stroke:#92400e
```

</div>
</div>

---

# Por que isso é tão difícil?

<div class="mt-8 text-xl">

<v-click>

> *"Sabemos mais do que conseguimos contar."*
> — **Paradoxo de Polanyi**

</v-click>

</div>

<div class="mt-6">

<v-clicks>

- Reconhecemos um rosto, andamos de bicicleta, identificamos sarcasmo —
  mas é **muito difícil escrever as regras** que descrevem como fazemos isso
- Regras explícitas não cobrem **casos extremos** nem generalizam para situações novas
- Resultado: a IA simbólica esbarrou na **complexidade do mundo real**

</v-clicks>

</div>

---
layout: center
---

# Mudança de paradigma

<div class="text-2xl mt-6 max-w-3xl mx-auto text-center">

Em vez de **dizer** ao computador o que fazer…

<div class="mt-8" v-click>

…<strong class="text-indigo-400">mostre</strong> a ele <em>muitos exemplos</em> de entrada e saída,
e deixe que <strong class="text-indigo-400">técnicas estatísticas</strong> aprendam a relação.

</div>

</div>

<div class="mt-12 text-center" v-click>
  <span class="px-4 py-2 bg-indigo-500/20 rounded-lg text-indigo-200">
    Isto é Machine Learning
  </span>
</div>

---

# Machine Learning, em uma figura

<div class="mt-4">
  <MLDiagram />
</div>

<div class="mt-4 text-sm opacity-80">

Algoritmos clássicos: regressão linear, regressão logística, árvores, *random forests*,
*gradient boosting*, SVMs, redes neurais rasas…

</div>

---

# ML brilha com **dados estruturados**

<div class="grid grid-cols-2 gap-6 mt-4">

<div>

<v-clicks>

- Dados estruturados = aqueles que cabem **naturalmente em uma planilha**
- Cada coluna é uma *feature* numérica/categórica significativa
- ML clássico funciona **muito bem** nesses casos:
  - *score* de crédito
  - previsão de demanda
  - detecção de fraude
  - diagnóstico baseado em exames laboratoriais

</v-clicks>

</div>

<div>

| idade | renda | tem_imovel | inadimplente |
|------:|------:|:----------:|:------------:|
| 32    | 5.4k  | sim        | 0            |
| 47    | 3.1k  | não        | 1            |
| 25    | 2.8k  | não        | 0            |
| 51    | 9.2k  | sim        | 0            |

<div class="mt-2 text-xs opacity-60 text-center">Exemplo ilustrativo (sintético)</div>

</div>

</div>

---

# Mas e os **dados não estruturados**?

<div class="grid grid-cols-3 gap-6 mt-6 text-center">

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="text-4xl">🖼️</div>
  <div class="mt-2 font-bold">Imagens</div>
  <div class="text-sm opacity-70 mt-1">pixels RGB sem significado isolado</div>
</div>

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="text-4xl">📝</div>
  <div class="mt-2 font-bold">Texto</div>
  <div class="text-sm opacity-70 mt-1">sequências de caracteres/tokens</div>
</div>

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="text-4xl">🔊</div>
  <div class="mt-2 font-bold">Áudio</div>
  <div class="text-sm opacity-70 mt-1">amostras temporais brutas</div>
</div>

</div>

<div class="mt-8" v-click>

A **forma bruta** desses dados não tem significado intrínseco para os algoritmos
clássicos. Um pixel `(R=128, G=64, B=200)` não diz nada sozinho sobre haver um
gato na imagem.

</div>

---

# A dificuldade dos dados não estruturados

<div class="mt-4">
  <PixelGrid />
</div>

<div class="mt-2 text-sm opacity-70 text-center">
Um classificador <em>não vê</em> o gato — vê uma matriz de números.
</div>

---

# A solução pré-DL: **engenharia de atributos**

<div class="grid grid-cols-2 gap-6">

<div>

<v-clicks>

- Especialistas projetavam **representações artesanais** dos dados:
  - **SIFT**, **HOG**, **SURF** para imagens
  - **MFCC** para áudio
  - **TF-IDF**, *bag-of-words* para texto
- A representação extraída era então alimentada
  em um modelo **clássico** (geralmente regressão logística!)

</v-clicks>

</div>

<div>

```mermaid {scale: 0.85}
flowchart LR
  R[Dado bruto<br/>imagem/texto/áudio] --> F[Engenharia<br/>de features<br/>SIFT, HOG, MFCC...]
  F --> M[Modelo clássico<br/>logística, SVM...]
  M --> S[Predição]
  style F fill:#f59e0b,stroke:#92400e,color:#000
```

</div>

</div>

<div class="mt-6" v-click>

Isso exigia **muito esforço humano** — um *gargalo* que limitou drasticamente
o alcance de ML em dados ricos como imagens, voz e linguagem natural.

</div>

---
layout: center
class: text-center
---

# Surge o **Deep Learning**

<div class="mt-6 text-xl max-w-3xl mx-auto">

Modelos que **aprendem a representação** diretamente dos dados brutos —
eliminando o gargalo da engenharia manual de atributos.

</div>

<div class="mt-12">
  <AIHierarchy showDL />
</div>

---

# O que **DL faz** que ML clássico não fazia?

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

<v-clicks>

- **Extrai automaticamente** representações úteis dos dados não estruturados
- Essas representações podem alimentar até modelos triviais (uma regressão
  logística no topo já entrega resultados impressionantes)
- Resolve o **gargalo humano** que limitava ML em imagens, texto e áudio

</v-clicks>

</div>

<div>

```mermaid {scale: 0.85}
flowchart LR
  R[Dado bruto] --> DL[Deep Learning]
  DL -->|representação<br/>aprendida| C[Classificador]
  C --> Y[Predição]
  style DL fill:#4f46e5,stroke:#312e81,color:#fff
  style C fill:#10b981,stroke:#065f46,color:#fff
```

<div class="mt-6 text-sm opacity-80">
A engenharia de features deixa de ser <em>humana</em>
e passa a ser <em>aprendida</em>.
</div>

</div>

</div>

---

# Por que aconteceu **agora**?

<div class="grid grid-cols-3 gap-6 mt-6 text-center">

<div v-click class="p-6 rounded-lg bg-gradient-to-b from-blue-500/10 to-blue-700/20 border border-blue-500/30">
  <div class="text-5xl">💡</div>
  <div class="mt-3 font-bold text-lg">Algoritmos</div>
  <div class="text-sm opacity-80 mt-2">
    ReLU, dropout, batch norm,
    arquiteturas convolucionais e
    transformers, otimizadores
    (Adam, etc.)
  </div>
</div>

<div v-click class="p-6 rounded-lg bg-gradient-to-b from-emerald-500/10 to-emerald-700/20 border border-emerald-500/30">
  <div class="text-5xl">📊</div>
  <div class="mt-3 font-bold text-lg">Dados</div>
  <div class="text-sm opacity-80 mt-2">
    Digitalização de tudo:
    fotos, vídeos, redes sociais,
    sensores, logs.
    <br/>ImageNet, Common Crawl…
  </div>
</div>

<div v-click class="p-6 rounded-lg bg-gradient-to-b from-amber-500/10 to-amber-700/20 border border-amber-500/30">
  <div class="text-5xl">⚡</div>
  <div class="mt-3 font-bold text-lg">Computação</div>
  <div class="text-sm opacity-80 mt-2">
    GPUs, depois TPUs.
    Treinos paralelos massivos
    em precisão reduzida.
  </div>
</div>

</div>

<div class="mt-8 text-center text-lg" v-click>

…aplicados a uma ideia <strong>antiga</strong>: as <strong>redes neurais artificiais</strong>.

</div>

---

# Aplicação imediata: **percepção**

<div class="mt-4">

Cada **sensor** pode ganhar a capacidade de detectar, reconhecer e
classificar o que está percebendo. *Acoplar* DL a câmeras, microfones e
sensores cria produtos qualitativamente diferentes.

</div>

<div class="grid grid-cols-2 md:grid-cols-4 gap-3 mt-6 text-center">

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">📷</div>
  <div class="text-sm mt-1">Detecção de objetos</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🩺</div>
  <div class="text-sm mt-1">Diagnóstico por imagem</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🚗</div>
  <div class="text-sm mt-1">Direção autônoma</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🏭</div>
  <div class="text-sm mt-1">Inspeção industrial</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🎤</div>
  <div class="text-sm mt-1">Reconhecimento de fala</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🔬</div>
  <div class="text-sm mt-1">Análise de microscopia</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🛰️</div>
  <div class="text-sm mt-1">Sensoriamento remoto</div>
</div>

<div v-click class="p-3 rounded bg-slate-800/40">
  <div class="text-3xl">🦅</div>
  <div class="text-sm mt-1">Bio-acústica</div>
</div>

</div>

---
layout: center
---

# E quanto à **saída**?

<div class="text-xl mt-4 max-w-3xl mx-auto text-center">

DL clássico previa **saídas estruturadas** com facilidade
(um número, um rótulo, um vetor de probabilidades).

</div>

<div class="mt-8" v-click>

Mas **gerar** texto, imagens, áudio, código? Por muito tempo, esse era
um **território difícil**.

</div>

---

# Saídas que ML/DL conseguia prever bem

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

<v-clicks>

- **Um número**
  - probabilidade de inadimplência
  - demanda da próxima semana
- **Poucos números**
  - distribuição sobre 1 000 classes do ImageNet
  - coordenadas de um *bounding box*
- **Um rótulo discreto**
  - sentimento (positivo/negativo)
  - categoria de produto

</v-clicks>

</div>

<div>

```mermaid {scale: 0.85}
flowchart LR
  X[Entrada<br/>estrut. ou não] --> M[Deep Learning]
  M --> Y1[Número]
  M --> Y2[Vetor de classes]
  M --> Y3[Caixa delimitadora]
  style M fill:#4f46e5,stroke:#312e81,color:#fff
```

</div>

</div>

---
layout: center
class: text-center
---

# Então surgiu a **IA Generativa**

<div class="mt-6 max-w-3xl mx-auto text-xl">

Modelos capazes de **produzir** conteúdo não estruturado:
imagens, texto, código, áudio, vídeo.

</div>

<div class="mt-10">
  <AIHierarchy showGenAI />
</div>

---

# A "matriz" $X \to Y$ da IA Generativa

<div class="mt-4 text-center text-lg">

Tanto **$X$** quanto **$Y$** podem ser texto, imagem, áudio, vídeo, código…
e até **multimodais**.

</div>

<div class="mt-6">
  <XYMatrix />
</div>

---

# Resumo: $X \to Y$

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

> Toda a empolgação atual com IA é, em essência,
> o sucesso do **Deep Learning** (e da IA Generativa que veio sobre ele).

</div>

---
layout: section
---

# O que é uma rede neural?

Vamos começar do começo.

---

# Voltando à **regressão logística**

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

A regressão logística mapeia um vetor de entradas em uma **probabilidade**:

$$
p(y=1\mid \mathbf{x}) \;=\; \sigma\!\left(b + \sum_{i=1}^{n} w_i\, x_i\right)
$$

onde $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ é a função **sigmoide**.

<v-click>

Vamos olhar para essa expressão como uma **rede de operações matemáticas**.

</v-click>

</div>

<div>
  <SigmoidPlot />
</div>

</div>

---

# Exemplo: prever quem é chamado para entrevista

<div class="grid grid-cols-2 gap-6 mt-4">

<div class="text-base">

<v-clicks>

- **Entradas**:
  - GPA (coeficiente de rendimento)
  - Anos de experiência
- **Saída**: probabilidade de ser chamado para a entrevista
  ($1$ = chamado, $0$ = não chamado)
- **Modelo**: regressão logística treinada nos dados históricos

</v-clicks>

</div>

<div>

| # | GPA | Exp. | Chamado |
|--:|----:|-----:|:-------:|
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

# Suponha que treinamos e obtivemos:

<div class="mt-4 text-center text-2xl">

$$
p(\text{chamado}) = \sigma\!\big(0{,}4 + 0{,}2\cdot\text{GPA} + 0{,}5\cdot\text{Exp}\big)
$$

</div>

<div class="mt-8">

Reescrevendo como uma **rede** com fluxo da esquerda para a direita:

</div>

<div class="mt-4">
  <LogRegNetwork />
</div>

---

# Predizendo com a "rede"

<div class="mt-2">
Considere um candidato com <strong>GPA = 3,8</strong> e <strong>1,2 anos</strong> de experiência.
</div>

<div class="mt-4">
  <LogRegNetwork :gpa="3.8" :exp="1.2" showValues />
</div>

<div class="mt-4 text-sm">

$$
\sigma(0{,}4 + 0{,}2\cdot 3{,}8 + 0{,}5\cdot 1{,}2) \;=\; \sigma(1{,}76) \;\approx\; 0{,}85
$$

A probabilidade estimada de chamada para entrevista é cerca de **85 %**.

</div>

---

# Vocabulário básico

<div class="mt-6">
  <NeuralNetwork
    :layers="[3, 1]"
    :labels="['Entradas', 'Saída']"
    showWeights
  />
</div>

<div class="mt-6 grid grid-cols-2 gap-x-12 gap-y-2 text-base">

<div><strong class="text-indigo-300">Pesos</strong> — multiplicadores nas conexões entre neurônios</div>
<div><strong class="text-indigo-300">Bias</strong> — termo aditivo (intercepto) em cada neurônio</div>
<div><strong class="text-indigo-300">Neurônio</strong> — unidade que faz $\sum + \text{bias}$ e aplica ativação</div>
<div><strong class="text-indigo-300">Camada</strong> — pilha vertical de neurônios</div>

</div>

---

# Por que a "lente de rede" é útil?

<div class="mt-4 text-lg max-w-3xl mx-auto">

<v-clicks>

- Ela nos permite **transformar** os dados de entrada antes da decisão final
- Em vez de alimentar a regressão logística com os dados crus,
  podemos primeiro **aprender uma melhor representação** deles
- Esse é o coração do *Deep Learning*: empilhar transformações aprendidas

</v-clicks>

</div>

---

# Empilhando funções lineares

<div class="mt-4 max-w-4xl mx-auto">

Antes da decisão final, **insira** uma camada de funções lineares
que combine as entradas:

</div>

<div class="mt-4">
  <NeuralNetwork :layers="[3, 3, 1]" :labels="['Entrada', 'Oculta', 'Saída']" />
</div>

<div class="mt-4 text-sm opacity-80 text-center">

Cada nó da camada intermediária é $z_j = b_j + \sum_i w_{ji} x_i$
— uma **transformação linear** das entradas.

</div>

---

# Empilhando ainda mais

<div class="mt-2">
  <NeuralNetwork :layers="[3, 3, 3, 1]" :labels="['Entrada', 'Oculta 1', 'Oculta 2', 'Saída']" />
</div>

<div class="mt-4 max-w-3xl mx-auto">

Podemos empilhar **quantas camadas** quisermos. No final, alimentamos
a regressão logística (sigmoide) com o vetor transformado.

</div>

<div class="mt-4 text-amber-300 text-center" v-click>

⚠ Mas só com camadas <em>lineares</em> a composição segue sendo linear —
precisamos de algo a mais.

</div>

---

# Funções de ativação **não lineares**

<div class="mt-4 max-w-3xl mx-auto">

Para que empilhar camadas faça diferença, cada neurônio aplica uma
**função de ativação não linear** depois da combinação linear:

$$
a_j = \phi\!\Big(b_j + \sum_i w_{ji}\, x_i\Big)
$$

</div>

<div class="mt-6">
  <ActivationFunctions />
</div>

---

# Funções de ativação comuns

<div class="grid grid-cols-3 gap-4 mt-4">

<div class="p-4 rounded-lg bg-slate-800/40">

**Sigmoide**

$\sigma(z) = \dfrac{1}{1+e^{-z}}$

<div class="text-xs opacity-70 mt-2">
Saída em (0,1). Usada na <strong>camada de saída</strong> para classificação binária. Sofre de <em>vanishing gradients</em> em camadas profundas.
</div>

</div>

<div class="p-4 rounded-lg bg-slate-800/40">

**Tanh**

$\tanh(z) = \dfrac{e^z - e^{-z}}{e^z + e^{-z}}$

<div class="text-xs opacity-70 mt-2">
Saída em (−1,1), centrada em zero. Mais bem comportada que sigmoide, mas ainda satura.
</div>

</div>

<div class="p-4 rounded-lg bg-slate-800/40">

**ReLU**

$\mathrm{ReLU}(z) = \max(0,\, z)$

<div class="text-xs opacity-70 mt-2">
Padrão moderno em camadas ocultas. Barata, evita saturação para z &gt; 0, permite redes muito mais profundas.
</div>

</div>

</div>

<div class="mt-6 text-center text-sm opacity-80">
Outras: Leaky ReLU, ELU, GELU, Swish, Softmax (na saída multi-classe)…
</div>

---

# Notação visual

<div class="mt-6 max-w-3xl mx-auto">

A partir daqui, vamos abreviar cada neurônio com um **círculo**, indicando
a ativação por cor/rótulo:

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
  <div class="mt-2 text-sm">Sigmoide</div>
</div>

</div>

---

# Anatomia de uma DNN

<div class="mt-2">
  <NeuralNetwork
    :layers="[4, 5, 5, 1]"
    :labels="['Entrada', 'Oculta 1', 'Oculta 2', 'Saída']"
    animate
  />
</div>

<div class="mt-6 grid grid-cols-2 gap-x-10 gap-y-1 text-base max-w-4xl mx-auto">

<div><strong>Camada de entrada:</strong> as variáveis <em>x₁, …, xₖ</em></div>
<div><strong>Camadas ocultas:</strong> transformam a representação</div>
<div><strong>Camada de saída:</strong> produz a predição</div>
<div><strong>Conexões densas:</strong> cada neurônio se liga a todos da próxima camada</div>

</div>

<div class="mt-4 text-center text-sm opacity-80">
Quando todas as camadas são densas, dizemos que a rede é <em>fully connected</em>.
</div>

---

# Projetando uma DNN: as escolhas

<div class="mt-6 max-w-3xl mx-auto">

Quando você desenha uma rede neural feedforward (*vanilla*), você decide:

</div>

<div class="mt-6 grid grid-cols-2 gap-4">

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="font-bold text-indigo-300">Quantas camadas ocultas?</div>
  <div class="text-sm opacity-80 mt-1">profundidade</div>
</div>

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="font-bold text-indigo-300">Quantos neurônios em cada camada?</div>
  <div class="text-sm opacity-80 mt-1">largura</div>
</div>

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="font-bold text-indigo-300">Qual ativação nas ocultas?</div>
  <div class="text-sm opacity-80 mt-1">tipicamente ReLU/GELU</div>
</div>

<div v-click class="p-4 rounded bg-slate-800/40">
  <div class="font-bold text-indigo-300">Qual ativação na saída?</div>
  <div class="text-sm opacity-80 mt-1">depende da tarefa</div>
</div>

</div>

<div class="mt-6 max-w-3xl mx-auto text-sm opacity-80" v-click>

A escolha da ativação de saída é guiada pela natureza de $y$:
linear (regressão), sigmoide (classificação binária), softmax (multiclasse).

</div>

---

# Aplicando ao classificador de entrevistas

<v-clicks>

- **Entradas**: 2 variáveis (GPA, experiência)
- **Saída**: probabilidade $\in (0,1)$
- **Decisões de projeto**:
  - 1 camada oculta com **3 neurônios** ReLU
  - **Sigmoide** na saída (probabilidade)

</v-clicks>

<div class="mt-4" v-click>
  <NeuralNetwork
    :layers="[2, 3, 1]"
    :labels="['Entrada', 'Oculta (ReLU)', 'Saída (σ)']"
  />
</div>

---

# Quantos parâmetros tem essa rede?

<div class="mt-4">
  <NeuralNetwork
    :layers="[2, 3, 1]"
    :labels="['Entrada', 'Oculta', 'Saída']"
    showCount
  />
</div>

<div class="mt-4 max-w-3xl mx-auto" v-click>

- Da entrada para a oculta: $2 \times 3 = 6$ pesos $+ 3$ biases $= 9$
- Da oculta para a saída:   $3 \times 1 = 3$ pesos $+ 1$ bias $= 4$
- **Total**: <strong class="text-indigo-300">13 parâmetros</strong>

</div>

---

# Pesos treinados (suposição)

<div class="mt-2">
  <InterviewNet />
</div>

<div class="mt-4 text-sm opacity-80 text-center">

Como esses pesos são <em>encontrados</em> a partir dos dados é o tema
da <strong>próxima aula</strong> (treinamento, perda, retropropagação).

</div>

---

# *Forward pass* — calculando uma predição

<div class="mt-2">

Para um candidato com $x_1 = 2{,}3$ (GPA) e $x_2 = 10{,}2$ (experiência*):

</div>

<div class="mt-4">
  <InterviewNet :x1="2.3" :x2="10.2" showActivations />
</div>

<div class="mt-3 text-xs opacity-60 text-center">
*Valor exagerado apenas para tornar as contas didáticas.
</div>

---

# Camada oculta: contas explícitas

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

E a saída:

$$
y = \sigma\!\big(0{,}05 + (-0{,}2)\cdot 1{,}87 + (-0{,}3)\cdot 3{,}03 + (-0{,}15)\cdot 0\big)
\approx \sigma(-1{,}23) \approx 0{,}226
$$

</div>

<div class="mt-3 text-center text-amber-300" v-click>

Para este candidato, a rede estima ≈ <strong>22,6 %</strong> de chance de chamada.

</div>

---

# A rede como uma função

<div class="mt-4 max-w-4xl mx-auto">

A predição da rede pode ser escrita como **uma única função** das entradas:

$$
\hat y(\mathbf{x}) \;=\; \sigma\!\Big( \mathbf{w}^{(2)\top}\, \mathrm{ReLU}\big(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\big) + b^{(2)} \Big)
$$

</div>

<div class="mt-6 grid grid-cols-2 gap-6">

<div v-click>

**Regressão logística pura:**

$$
\hat y \;=\; \sigma\!\big(0{,}4 + 0{,}2 x_1 + 0{,}5 x_2\big)
$$

linear nos dados antes da sigmoide.

</div>

<div v-click>

**Rede com 1 camada oculta:**

uma composição não linear muito mais rica —
capaz de representar relações complexas entre $\mathbf{x}$ e $y$.

</div>

</div>

---

# Resumo: uma DNN

<div class="mt-2">
  <NeuralNetwork
    :layers="[4, 6, 6, 6, 2]"
    :labels="['Entrada', 'Oculta 1', 'Oculta 2', 'Oculta 3', 'Saída']"
    animate
  />
</div>

<div class="mt-4 grid grid-cols-2 gap-8 max-w-4xl mx-auto text-sm">

<div>

- **Feedforward** (vanilla): dados fluem da esquerda para a direita
- **Arquitetura** = camadas + ativações + conexões
- **Profundidade** vem de muitas camadas ocultas

</div>

<div>

- *Deep Learning* é, em essência, **redes neurais com muitas camadas**
- Treinamento, retropropagação e otimização — próxima aula

</div>

</div>

---
layout: center
class: text-center
---

# Recapitulando

<div class="mt-6 grid grid-cols-2 gap-6 max-w-4xl mx-auto text-left text-base">

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">Hierarquia de ideias</strong>
<div class="mt-1">IA ⊃ ML ⊃ DL ⊃ IA Generativa</div>
</div>

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">Por que DL é especial</strong>
<div class="mt-1">aprende representações de dados não estruturados</div>
</div>

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">A combinação que destravou</strong>
<div class="mt-1">algoritmos + dados + GPUs</div>
</div>

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">Rede neural ≈ regressão</strong>
<div class="mt-1">logística empilhada com não linearidades</div>
</div>

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">Componentes</strong>
<div class="mt-1">camadas, neurônios, pesos, biases, ativações</div>
</div>

<div class="p-4 rounded-lg bg-slate-800/40">
<strong class="text-indigo-300">Forward pass</strong>
<div class="mt-1">composição linear → não linearidade → repetir</div>
</div>

</div>

---

# Próxima aula

<div class="mt-8 grid grid-cols-2 gap-8 max-w-4xl mx-auto">

<div>

<v-clicks>

- Como **treinar** uma rede neural?
- O que é uma **função de perda**?
- **Gradiente descendente** e suas variantes
- **Retropropagação** (intuição + um exemplo numérico)

</v-clicks>

</div>

<div>

```mermaid {scale: 0.75}
flowchart TD
  D[Dados] --> F[Forward pass]
  F --> L[Loss]
  L --> B[Backward pass]
  B --> U[Atualiza pesos]
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

# Obrigado! Perguntas?

<div class="mt-6 text-sm opacity-70">

Adaptado livremente de <em>15.773 Hands-on Deep Learning</em>
(MIT OpenCourseWare, 2024) — material original em inglês de Vivek Farias e
Rama Ramakrishnan, distribuído sob os termos do MIT OCW.

</div>

<div class="mt-2 text-xs opacity-60">
Para mais informações: https://ocw.mit.edu/terms
</div>
