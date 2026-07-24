---
theme: seriph
background: https://cover.sli.dev
title: 'Aula 8 — Grandes Modelos de Linguagem (LLMs)'
info: |
  ## Tópicos Avançados em Inteligência Artificial
  Aula 8: LLMs — Arquitetura, Adaptação e Eficiência

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

# Aula 8

## Grandes Modelos de Linguagem (LLMs)

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

**Parte 1 — Do Encoder ao Decoder**
Causal Self-Attention e a família GPT

</div>

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40">

**Parte 2 — Tokenização Moderna**
Byte Pair Encoding (BPE)

</div>

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40">

**Parte 3 — Estratégias de Decodificação**
Greedy, Top-K, Top-P, Temperatura

</div>

</div>

<div class="space-y-3">

<div class="p-3 rounded bg-cyan-900/30 border border-cyan-500/40">

**Parte 4 — Adaptação de LLMs**
Zero-Shot, Few-Shot, RAG, Fine-Tuning

</div>

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40">

**Parte 5 — RLHF e Instruction Tuning**
Como o ChatGPT aprendeu a seguir instruções

</div>

<div class="p-3 rounded bg-rose-900/30 border border-rose-500/40">

**Parte 6 — Fine-Tuning Eficiente**
LoRA: Low-Rank Adaptation

</div>

</div>

</div>

---

# Mapa do território: encoder vs decoder

<div class="grid grid-cols-3 gap-4 mt-4 text-sm">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40 text-center" v-click>

**Encoder (BERT)**

Atenção **bidirecional**

Vê o contexto inteiro

✅ Classificação, NER, Q&A extrativo

❌ Não gera texto

</div>

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40 text-center" v-click>

**Decoder (GPT)**

Atenção **causal** (máscara)

Só vê tokens passados

✅ Geração de texto, chatbots, código

❌ Não é bidirecional

</div>

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40 text-center" v-click>

**Encoder-Decoder (T5)**

Encoder bidirecional

Decoder causal

✅ Tradução, sumarização

✅ Q&A generativo

</div>

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30 mt-4 text-sm" v-click>

**Esta aula**: focamos nos modelos **decoder-only** (GPT, LLaMA, Gemini) — a arquitetura dominante nos LLMs modernos. A diferença essencial em relação ao BERT é **uma modificação na atenção**: a máscara causal.

</div>

---
layout: section
---

# Parte 1 — Do Encoder ao Decoder

---

# Previsão da próxima palavra (Causal LM)

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Tarefa**: dado um prefixo de texto, prever o próximo token

Pares de treinamento criados **automaticamente**:

```
Entrada: "The cat sat on the"
Label:   "mat"

Entrada: "The cat sat on"
Label:   "the"
```

Nenhuma anotação humana — o corpus é a própria supervisão.

</div>

<div class="p-3 rounded bg-blue-900/20 border border-blue-500/30" v-click>

**Perda**: cross-entropy sobre o vocabulário em cada posição

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(w_t \mid w_1, \ldots, w_{t-1})$$

GPT-3 foi treinado assim em ~30 bilhões de frases da internet e livros.

</div>

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30 text-sm" v-click>

**Por que o encoder bidirecional não funciona para isto?**

No BERT, o token na posição *t* já "vê" os tokens *t+1, t+2, …* na atenção.

Se usarmos isso para prever a próxima palavra, o modelo **cola** na resposta — aprende apenas a copiar, não a gerar.

**Solução:** bloquear a atenção para o futuro — a **máscara causal**.

</div>

</div>

---

# Causal Self-Attention: a modificação-chave

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Atenção padrão (BERT)**

Ao calcular o embedding contextualizado de "sat", a atenção pondera **todos** os tokens: "the", "cat", "sat", "on", "the", "mat".

O futuro é visível — OK para MLM, errado para geração.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**Causal Self-Attention (GPT)**

Ao calcular o embedding de "sat" (posição 3), somente "the", "cat" e "sat" são ponderados.

Os tokens futuros recebem peso **−∞** antes do softmax → probabilidade zero.

</div>

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30 text-sm" v-click>

**Implementação:** basta somar uma máscara triangular inferior aos scores de atenção antes do softmax:

```python
# scores: (batch, heads, T, T)
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
attn = torch.softmax(scores, dim=-1)
```

**Custo zero** — o mesmo mecanismo de atenção, apenas com uma máscara triangular.

</div>

</div>

---

# Transformer Decoder: arquitetura completa

<div class="grid grid-cols-2 gap-4 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2.5 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Bloco Transformer Decoder** (mesmo esqueleto que o encoder):

```
Token embeddings + Positional embeddings
        ↓ (× N blocos)
┌─────────────────────────────┐
│  Causal Multi-Head Attention │  ← com máscara
│  + Add & LayerNorm           │
│  Feed-Forward Network (FFN)  │
│  + Add & LayerNorm           │
└─────────────────────────────┘
        ↓
    Dense → Softmax (vocab)
```

</div>

<div class="p-2.5 rounded bg-blue-900/20 border border-blue-500/30" v-click>

**Encoder-Decoder**: o decoder tem um terceiro sub-bloco — **cross-attention** sobre as saídas do encoder. Usado em T5, BART, modelos de tradução.

Nos **decoder-only** (GPT, LLaMA): só causal self-attention — sem encoder.

</div>

</div>

<div class="space-y-2">

<div class="p-2.5 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**GPT-3 em números**

| Componente | Valor |
|---|---|
| Blocos (camadas) | 96 |
| Cabeças de atenção | 96 |
| Dimensão do modelo | 12 288 |
| Parâmetros totais | **175 bilhões** |
| Tokens de treinamento | ~300 bilhões |
| Custo estimado de treino | >$4M |

</div>

<div class="p-2.5 rounded bg-amber-900/20 border border-amber-500/30 text-xs" v-click>

**Capacidades emergentes**: Com escala suficiente, o modelo aprende tarefas nunca vistas explicitamente — tradução, aritmética, raciocínio lógico — apenas a partir do próximo-token-prediction em texto.

</div>

</div>

</div>

---
layout: section
---

# Parte 2 — Tokenização Moderna (BPE)

---

# O problema com tokenização por palavras

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Tokenização por palavras inteiras**

```
"jogar" → id 1842
"jogando" → id 9371  (palavra diferente!)
"jogador" → id 4521  (palavra diferente!)
"jogarei" → id ???   (out-of-vocabulary)
```

Vocabulário gigante: inglês tem >500k palavras. Palavras raras (nomes próprios, jargão técnico, erros de digitação) ficam fora do vocabulário.

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**Tokenização por caracteres**

```
"jogar" → ['j','o','g','a','r']
```

Vocabulário minúsculo (~256 caracteres). Mas sequências ficam muito longas e a atenção escala com O(T²).

</div>

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30 mt-4 text-sm" v-click>

**Byte Pair Encoding (BPE)**: encontra um ponto intermediário ótimo. Vocabulário final contém caracteres, subpalavras frequentes e palavras completas comuns. GPT-2/3: ~50k tokens. GPT-4: ~100k tokens. BERT usa variante chamada **WordPiece**.

</div>

</div>

---

# BPE: como funciona

<div class="grid grid-cols-2 gap-4 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2.5 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Algoritmo (treinamento)**

1. Iniciar com cada caractere como token
2. Contar **pares de tokens adjacentes** mais frequentes
3. **Fundir** o par mais frequente em um novo token
4. Repetir até o vocabulário atingir o tamanho desejado

</div>

<div class="p-2.5 rounded bg-violet-900/20 border border-violet-500/30 font-mono text-xs" v-click>

**Exemplo:** corpus = "the cat sat on the mat"

```
Vocab inicial: [t][h][e][c][a][s][o][n][m]
Corpus:  [t,h,e,_,c,a,t,_,s,a,t,_,o,n,_,t,h,e,_,m,a,t]

Par mais frequente: [a,t] → 3 ocorrências
Novo vocab: [t][h][e][c][a][s][o][n][m][at]
Corpus:  [t,h,e,_,c,at,_,s,at,_,o,n,_,t,h,e,_,m,at]

Par mais frequente: [t,h] → 2 ocorrências
Novo vocab: [...][at][th]
...
```

</div>

</div>

<div class="space-y-2">

<div class="p-2.5 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Resultado prático**

Texto novo é tokenizado aplicando as fusões aprendidas na mesma ordem.

```
"playing"  → ["play", "ing"]
"unplayed" → ["un", "play", "ed"]
"GPT-4"    → ["G", "PT", "-", "4"]
```

Nunca há tokens fora do vocabulário: o pior caso é um caractere isolado.

</div>

<div class="p-2.5 rounded bg-amber-900/20 border border-amber-500/30 text-xs" v-click>

**Na prática (HuggingFace):**

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok.encode("Aprendizado profundo é fascinante!")
print(tok.convert_ids_to_tokens(ids))
# ['Ap', 'rend', 'izado', 'Ġprofundo',
#  'Ġé', 'Ġfas', 'cin', 'ante', '!']
```

</div>

</div>

</div>

---
layout: section
---

# Parte 3 — Estratégias de Decodificação

---

# O problema da decodificação

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**A saída do LLM**: uma distribuição de probabilidade sobre o vocabulário a cada passo

```
Prompt: "It was a dark and ____"

stormy  → 0.60
rainy   → 0.20
foggy   → 0.10
cold    → 0.05
dreary  → 0.03
...     → 0.02
```

Como escolher o próximo token?

</div>

<div class="space-y-3">

<div class="p-3 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Greedy Decoding**

Sempre escolher o token de maior probabilidade.

✅ Determinístico e rápido

❌ Pode prender o modelo em loops ("the the the…")

❌ Subótimo: a escolha localmente ótima pode bloquear sequências melhores globalmente

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30" v-click>

**Beam Search** (variante do greedy)

Manter os *k* melhores prefixos a cada passo. Mais robusto, mas ainda pode ser repetitivo e não é adequado para respostas criativas.

</div>

</div>

</div>

---

# Sampling: aleatoriedade controlada

<div class="grid grid-cols-2 gap-4 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2.5 rounded bg-cyan-900/30 border border-cyan-500/40" v-click>

**Random Sampling**

Amostrar proporcionalmente à probabilidade.

✅ Diversidade e criatividade

❌ Pode amostrar tokens ruins da "cauda longa"

Um token raro pode desviar toda a geração subsequente.

</div>

<div class="p-2.5 rounded bg-cyan-900/30 border border-cyan-500/40" v-click>

**Top-K Sampling (K=2)**

Considerar apenas os *K* tokens mais prováveis. Renormalizar e amostrar.

```
stormy  0.60 → 0.75
rainy   0.20 → 0.25
--- descartados ---
foggy   0.10
cold    0.05
```

✅ Elimina a cauda longa

❌ *K* fixo pode ser muito restritivo ou liberal dependendo da distribuição

</div>

</div>

<div class="space-y-2">

<div class="p-2.5 rounded bg-cyan-900/20 border border-cyan-500/30" v-click>

**Top-P (Nucleus) Sampling (P=0.9)**

Considerar o menor conjunto de tokens cuja probabilidade acumulada ≥ P. Renormalizar e amostrar.

```
stormy  0.60 → soma = 0.60
rainy   0.20 → soma = 0.80
foggy   0.10 → soma = 0.90  ← parar aqui (P=0.9)
```

✅ Adapta-se automaticamente à entropia da distribuição

✅ Preferido na prática (ChatGPT, LLaMA)

</div>

<div class="p-2.5 rounded bg-violet-900/20 border border-violet-500/30 text-xs" v-click>

**Temperature (τ)**

Escalar os logits antes do softmax: `logits / τ`

- **τ < 1**: distribução mais "afiada" → mais determinístico
- **τ = 1**: distribuição original
- **τ > 1**: distribuição mais uniforme → mais criativo/aleatório

Combinado com Top-P: parâmetros padrão ChatGPT ≈ `top_p=1.0, temperature=0.7`

</div>

</div>

</div>

---

# Quando usar cada estratégia

<div class="grid grid-cols-2 gap-4 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Greedy / Beam Search**

Use quando a **precisão factual** importa mais que a diversidade:
- Extração de dados estruturados
- Completar código com resposta única correta
- Tradução de terminologia técnica

</div>

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Top-K / Top-P + Temperature**

Use quando **variedade e naturalidade** são importantes:
- Chatbots e assistentes de conversação
- Escrita criativa e brainstorming
- Geração de múltiplas alternativas para revisão humana

</div>

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30 text-sm mt-4 col-span-2" v-click>

**Receita prática:** `top_p=0.9, temperature=0.7` cobre a maioria dos casos. Aumente temperature para textos mais criativos; reduza para respostas mais precisas. Use greedy apenas para tarefas de extração determinística.

</div>

</div>

---
layout: section
---

# Parte 4 — Adaptação de LLMs

---

# A escada de adaptação

<div class="mt-4 text-sm">

<div class="grid grid-cols-4 gap-3 mb-4">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40 text-center" v-click>

**Zero-Shot**

Instrução clara sem exemplos

Prompt = tarefa descrita em linguagem natural

🔧 Esforço: mínimo

</div>

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40 text-center" v-click>

**Few-Shot**

Alguns exemplos no prompt

Prompt = instrução + 2–5 exemplos

🔧 Esforço: baixo

</div>

<div class="p-3 rounded bg-amber-900/30 border border-amber-500/40 text-center" v-click>

**RAG**

Recuperar documentos relevantes e injetar no prompt

🔧 Esforço: médio

✅ Dados proprietários

</div>

<div class="p-3 rounded bg-emerald-900/30 border border-emerald-500/40 text-center" v-click>

**Fine-Tuning**

Retreinar o modelo com exemplos do domínio alvo

🔧 Esforço: alto

✅ Máxima especialização

</div>

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30 text-sm" v-click>

**Regra prática:** comece sempre pelo zero-shot. Se o resultado for insuficiente, adicione exemplos (few-shot). Se o domínio for proprietário/específico, considere RAG. Fine-tuning somente quando as outras abordagens não atingirem a qualidade necessária — o custo é alto.

</div>

</div>

---

# Prompt Engineering

<div class="grid grid-cols-2 gap-4 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2.5 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Zero-Shot: instrução clara**

```
Prompt:
Classifique o sentimento desta avaliação
como Positivo, Negativo ou Neutro.

Avaliação: "O produto chegou rápido mas
a qualidade deixou a desejar."

Sentimento:
```

O LLM responde diretamente sem exemplos. Funciona bem para LLMs grandes.

</div>

<div class="p-2.5 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Few-Shot: exemplos no contexto**

```
Avaliação: "Ótimo produto!" → Positivo
Avaliação: "Péssima experiência." → Negativo
Avaliação: "Entregou o que prometeu." → Neutro

Avaliação: "Chegou rápido mas qualidade
baixa." →
```

Guia o comportamento sem retreinar o modelo.

</div>

</div>

<div class="space-y-2">

<div class="p-2.5 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Chain-of-Thought (CoT)**

```
Prompt: "Roger tem 5 bolas de tênis. Compra
mais 2 latas com 3 bolas cada. Quantas bolas
ele tem? Pense passo a passo."

Resposta: "Roger começa com 5 bolas.
2 latas × 3 bolas = 6 bolas novas.
5 + 6 = 11 bolas."
```

Apenas a frase "pense passo a passo" melhora drasticamente tarefas de raciocínio em modelos grandes.

</div>

<div class="p-2.5 rounded bg-slate-800/40 border border-slate-500/30 text-xs" v-click>

**Limitação: context window**

Todo o histórico de conversa + exemplos + resposta devem caber na janela de contexto. GPT-4: 128k tokens. LLaMA 3: 8k–128k tokens. Para grandes volumes de dados proprietários → RAG.

</div>

</div>

</div>

---

# RAG: Retrieval-Augmented Generation

<div class="grid grid-cols-2 gap-4 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2.5 rounded bg-cyan-900/30 border border-cyan-500/40" v-click>

**Motivação**

Uma empresa tem 50.000 documentos de suporte ao cliente. Impossível inserir tudo na context window. O LLM também não foi treinado nesse conteúdo proprietário.

**Ideia:** buscar os documentos relevantes para *cada pergunta* e injetá-los no prompt.

</div>

<div class="p-2.5 rounded bg-cyan-900/20 border border-cyan-500/30" v-click>

**Pipeline RAG**

```
Pergunta do usuário
        ↓
  Embedding da pergunta
        ↓
  Busca vetorial no banco
  de documentos (FAISS)
        ↓
  Top-K documentos relevantes
        ↓
  Prompt = pergunta + documentos
        ↓
       LLM
        ↓
  Resposta fundamentada
```

</div>

</div>

<div class="space-y-2">

<div class="p-2.5 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**Vantagens do RAG**

- Dados atualizados sem retreinar o modelo
- Transparência: citar as fontes usadas
- Funciona com qualquer LLM (incluindo APIs)
- Custo muito menor que fine-tuning

</div>

<div class="p-2.5 rounded bg-emerald-900/20 border border-emerald-500/30 text-xs" v-click>

**Implementação mínima:**

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embs = model.encode(documents)
index = faiss.IndexFlatL2(doc_embs.shape[1])
index.add(doc_embs)

q_emb = model.encode([query])
_, ids = index.search(q_emb, k=3)
context = "\n".join([documents[i] for i in ids[0]])
# → inserir `context` no prompt do LLM
```

</div>

</div>

</div>

---
layout: section
---

# Parte 5 — RLHF e Instruction Tuning

---

# O problema do GPT-3 base

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-rose-900/30 border border-rose-500/40" v-click>

**GPT-3 foi treinado para prever a próxima palavra**

Se o prompt for "Explique a lua para uma criança de 6 anos", o modelo pode:

- Continuar como se fosse um artigo científico
- Adicionar mais perguntas ao invés de responder
- Gerar conteúdo prejudicial encontrado no corpus

O modelo aprendeu *a distribuição do texto da internet*, não *a seguir instruções*.

</div>

<div class="p-3 rounded bg-rose-900/20 border border-rose-500/30" v-click>

**O objetivo real**

Queremos um modelo que:

- Siga as **instruções** do usuário
- Seja **útil** e **honesto**
- Evite respostas **prejudiciais**

Esses critérios não emergem automaticamente do next-word-prediction.

</div>

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30 mt-4 text-sm" v-click>

**Solução: Instruction Tuning = SFT + RLHF**

A OpenAI desenvolveu esse processo para transformar GPT-3 em InstructGPT/ChatGPT. O mesmo princípio é usado em LLaMA 2-Chat, Gemini, Claude e virtualmente todos os assistentes modernos.

</div>

</div>

---

# Passo 1: Supervised Fine-Tuning (SFT)

<div class="grid grid-cols-2 gap-4 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2.5 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Coletar dados de alta qualidade**

~12.500 pares (instrução, resposta) escritos por humanos contratados:

| Instrução | Resposta humana |
|---|---|
| "Explique o pouso na lua para uma criança de 6 anos em poucas frases." | "Pessoas foram à lua num foguete gigante, caminharam por lá e voltaram para a Terra. Tiraram fotos para todos verem!" |
| "Escreva um e-mail profissional de agradecimento após entrevista." | *resposta exemplar...* |

</div>

<div class="p-2.5 rounded bg-amber-900/20 border border-amber-500/30" v-click>

**Fine-tuning com next-word-prediction**

Usar esses pares como corpus de treinamento adicional — o mesmo objetivo de previsão da próxima palavra, mas agora em dados curados.

O modelo aprende o *estilo* de responder instruções, não apenas prever texto genérico.

**Resultado:** GPT-3 passa a seguir instruções muito melhor.

</div>

</div>

<div class="p-2.5 rounded bg-slate-800/40 border border-slate-500/30 text-sm" v-click>

**Limitação:** escrever 12.500 respostas de alta qualidade é caro e lento. Para fazer muito mais SFT, precisamos de uma forma mais barata de sinalizar qualidade — *é mais fácil ranquear respostas do que escrevê-las*.

</div>

</div>

---

# Passo 2: Reward Model (RM)

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Coletar dados de preferência**

Para 33.000 instruções, gerar **múltiplas respostas** com o GPT-3 (após SFT).

Humanos **ranqueiam** as respostas de melhor a pior — muito mais rápido do que escrever.

Par de treinamento:
- Instrução + Resposta preferida A + Resposta preterida B

</div>

<div class="p-3 rounded bg-violet-900/20 border border-violet-500/30" v-click>

**Treinar o Reward Model**

Um LLM adaptado para produzir um **score numérico único** para qualquer par (instrução, resposta).

Função de perda que penaliza quando a resposta preterida recebe score maior:

$$\mathcal{L} = -\log\bigl(\sigma(r_{\text{pref}} - r_{\text{out}})\bigr)$$

</div>

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30 mt-4 text-sm" v-click>

**O Reward Model aprendeu a emular preferências humanas.** Pode agora dar feedback automático e contínuo para qualquer resposta gerada — sem precisar de humanos a cada passo.

</div>

</div>

---

# Passo 3: RLHF — treinar com o Reward Model

<div class="grid grid-cols-2 gap-4 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2.5 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**Loop de Reinforcement Learning**

```
Instrução aleatória do dataset
        ↓
  GPT-3 gera resposta
        ↓
  Reward Model avalia: score R
        ↓
  Atualizar GPT-3 via PPO
  para maximizar E[R]
        ↓
  (repetir milhares de vezes)
```

Sem intervenção humana no loop — o RM faz o papel do crítico.

</div>

<div class="p-2.5 rounded bg-emerald-900/20 border border-emerald-500/30" v-click>

**Resultado: InstructGPT → ChatGPT**

GPT-3 → **SFT** → InstructGPT (GPT-3.5)

Com RLHF adicional e escala → **ChatGPT**

Avaliadores humanos preferiram InstructGPT (1.3B parâmetros com RLHF) sobre GPT-3 (175B sem RLHF) em 85% dos casos — o alinhamento importa mais que o tamanho.

</div>

</div>

<div class="p-2.5 rounded bg-rose-900/20 border border-rose-500/30 text-xs" v-click>

**Limitações do RLHF:** processo complexo e instável (PPO é sensível a hiperparâmetros); **reward hacking** (modelo aprende a "enganar" o RM); alucinações persistem; vieses dos anotadores são amplificados. Alternativas: DPO (Direct Preference Optimization), RLAIF (feedback de outro AI).

</div>

</div>

---
layout: section
---

# Parte 6 — Fine-Tuning Eficiente com LoRA

---

# O custo do fine-tuning completo

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-rose-900/30 border border-rose-500/40" v-click>

**LLaMA 2-70B: memória necessária**

| Componente | Memória |
|---|---|
| Parâmetros (fp16) | 140 GB |
| Gradientes | ~140 GB |
| Estado otimizador (Adam) | 140–560 GB |
| **Total** | **420–840 GB** |

Uma GPU A100 tem 80 GB. Precisaríamos de 6–11 GPUs **só para carregar o modelo**.

</div>

<div class="p-3 rounded bg-rose-900/20 border border-rose-500/30" v-click>

**Custo de tempo**

LLaMA-2-70B foi treinado em 2 trilhões de tokens com 2048 GPUs por ~28 dias.

Fine-tuning completo com dataset menor (Alpaca: 50k pares) em 7 GPUs leva ~20 horas — ainda inacessível para a maioria.

**Precisamos reduzir drasticamente a memória.**

</div>

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30 mt-4 text-sm" v-click>

**Insight fundamental:** fine-tuning muda os pesos do modelo em quantidades **muito pequenas**. A matriz de pesos resultante está próxima da original. Isso sugere que a mudança ΔW é *intrínsecamente de baixa complexidade*.

</div>

</div>

---

# LoRA: Low-Rank Adaptation

<div class="grid grid-cols-2 gap-4 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2.5 rounded bg-cyan-900/30 border border-cyan-500/40" v-click>

**Ideia central**

Em vez de atualizar a matriz completa $W \in \mathbb{R}^{d \times d}$, forçar que a mudança $\Delta W$ seja de **baixo posto** (*rank r*):

$$\Delta W = B \cdot C, \quad B \in \mathbb{R}^{d \times r},\; C \in \mathbb{R}^{r \times d}, \quad r \ll d$$

A matriz atualizada na inferência é:

$$W' = W + \Delta W = W + B \cdot C$$

</div>

<div class="p-2.5 rounded bg-cyan-900/20 border border-cyan-500/30" v-click>

**Redução de parâmetros**

Para $A_K$ no LLaMA-2-70B: $d = 8192$

- Original: $8192 \times 8192 = 67M$ parâmetros
- LoRA (r=4): $2 \times 4 \times 8192 = 65k$ parâmetros
- **Redução: 0.097% dos parâmetros originais**

Com r=4, treinamos apenas ~0.1% dos pesos — o restante fica **congelado**.

</div>

</div>

<div class="space-y-2">

<div class="p-2.5 rounded bg-violet-900/30 border border-violet-500/40 text-xs" v-click>

**Treinamento LoRA**

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,                    # posto das matrizes adaptadoras
    lora_alpha=16,          # escala (normalmente 2×r)
    target_modules=["q_proj", "v_proj"],  # matrizes alvo
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 (0.24% de 1.7B)
```

</div>

<div class="p-2.5 rounded bg-violet-900/20 border border-violet-500/30 text-xs" v-click>

**Memória com LoRA (LLaMA-2-70B)**

| Componente | Sem LoRA | Com LoRA |
|---|---|---|
| Pesos do modelo | 140 GB | 140 GB |
| Gradientes | 140 GB | ~0 GB* |
| Otimizador | 140–560 GB | ~0 GB** |
| **Total** | **420–840 GB** | **~140 GB** |

*Gradientes apenas para B e C (minúsculos). **Otimizador só para B e C.

</div>

</div>

</div>

---

# A curva esforço-benefício

<div class="grid grid-cols-2 gap-4 mt-4 text-sm">

<div class="space-y-3">

<div class="p-3 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Resumo das estratégias de adaptação**

| Estratégia | Esforço | Quando usar |
|---|---|---|
| Zero-Shot | ⭐ | Tarefas gerais |
| Few-Shot | ⭐⭐ | Formato específico |
| RAG | ⭐⭐⭐ | Dados proprietários |
| LoRA | ⭐⭐⭐⭐ | Domínio especializado |
| Full FT | ⭐⭐⭐⭐⭐ | Raramente necessário |

</div>

<div class="p-3 rounded bg-blue-900/20 border border-blue-500/30" v-click>

**Modelos open-source para fine-tuning**

- **LLaMA 3** (Meta): 8B e 70B, licença permissiva
- **Mistral / Mixtral**: 7B e MoE, eficientes
- **Phi-3** (Microsoft): modelos pequenos mas capazes
- **Gemma 2** (Google): 2B, 9B, 27B

Todos disponíveis via HuggingFace Hub + `peft` para LoRA.

</div>

</div>

<div class="p-3 rounded bg-slate-800/40 border border-slate-500/30 text-sm" v-click>

**QLoRA**: combina LoRA com **quantização em 4-bits** (NF4). Permite fine-tuning de LLaMA-2-70B em **uma única GPU de 48GB**. Danylo Nika e Tim Dettmers, 2023.

```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config)
```

</div>

</div>

---

# Resumo: o que aprendemos hoje

<div class="grid grid-cols-2 gap-4 mt-3 text-sm">

<div class="space-y-2">

<div class="p-2.5 rounded bg-blue-900/30 border border-blue-500/40" v-click>

**Parte 1 — Arquitetura Decoder**

Causal Self-Attention = atenção mascarada (máscara triangular). Transformer Decoder gera texto autoregressivamente. GPT-3: 175B parâmetros, capacidades emergentes.

</div>

<div class="p-2.5 rounded bg-violet-900/30 border border-violet-500/40" v-click>

**Parte 2 — BPE**

Tokenização por fusão iterativa de subpalavras frequentes. Elimina OOV, balanceia vocabulário vs. comprimento de sequência.

</div>

<div class="p-2.5 rounded bg-amber-900/30 border border-amber-500/40" v-click>

**Parte 3 — Decodificação**

Greedy (determinístico), Top-K (trunca cauda), Top-P/Nucleus (adapta-se à distribuição), Temperature (controla aleatoriedade).

</div>

</div>

<div class="space-y-2">

<div class="p-2.5 rounded bg-cyan-900/30 border border-cyan-500/40" v-click>

**Parte 4 — Adaptação**

Escada: Zero-Shot → Few-Shot → RAG → Fine-Tuning. RAG injeta documentos relevantes no prompt. Chain-of-Thought melhora raciocínio.

</div>

<div class="p-2.5 rounded bg-emerald-900/30 border border-emerald-500/40" v-click>

**Parte 5 — RLHF**

SFT em dados curados + Reward Model treinado com preferências humanas + RL para otimizar o reward. GPT-3 → ChatGPT.

</div>

<div class="p-2.5 rounded bg-rose-900/30 border border-rose-500/40" v-click>

**Parte 6 — LoRA**

ΔW ≈ B·C com r ≪ d. Treina <1% dos parâmetros. Reduz memória de gradientes/otimizador de 840GB para ~140GB. QLoRA adiciona quantização 4-bit.

</div>

</div>

</div>
