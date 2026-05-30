<template>
  <div class="softmax-viz">
    <svg viewBox="0 0 540 280" class="w-full max-w-4xl mx-auto">
      <!-- Input bars (raw scores z) -->
      <g>
        <text x="80" y="20" text-anchor="middle" fill="#cbd5e1" font-size="11" font-weight="600">scores brutos (z)</text>
        <g v-for="(z, i) in scores" :key="'z' + i">
          <rect :x="40" :y="40 + i * 22" :width="Math.abs(z) * 12 + 2" :height="16"
                :fill="z >= 0 ? 'rgba(34,211,238,0.5)' : 'rgba(248,113,113,0.5)'"
                :stroke="z >= 0 ? '#22d3ee' : '#f87171'" stroke-width="1.2" rx="2" />
          <text :x="z >= 0 ? 50 + Math.abs(z) * 12 : 36" :y="52 + i * 22"
                :text-anchor="z >= 0 ? 'start' : 'end'" fill="#e2e8f0" font-size="10">
            {{ z.toFixed(1) }}
          </text>
          <text x="32" :y="52 + i * 22" text-anchor="end" fill="#94a3b8" font-size="10">c{{ i }}</text>
        </g>
      </g>

      <!-- Softmax box -->
      <g>
        <rect x="220" y="100" width="100" height="50" rx="10"
              fill="rgba(79,70,229,0.4)" stroke="#a78bfa" stroke-width="2" />
        <text x="270" y="125" text-anchor="middle" fill="#e0e7ff" font-size="14" font-weight="700">softmax</text>
        <text x="270" y="142" text-anchor="middle" fill="#c4b5fd" font-size="9">e^z / Σ e^z</text>
      </g>

      <!-- Arrows in -->
      <g stroke="#94a3b8" stroke-width="1.4" fill="none" marker-end="url(#a)">
        <line v-for="(z, i) in scores" :key="'in' + i"
              :x1="160" :y1="48 + i * 22" :x2="218" :y2="125" />
      </g>

      <!-- Arrows out -->
      <g stroke="#94a3b8" stroke-width="1.4" fill="none" marker-end="url(#a)">
        <line v-for="(_, i) in scores" :key="'out' + i"
              :x1="322" :y1="125" :x2="380" :y2="48 + i * 22" />
      </g>

      <!-- Output bars (probabilities) -->
      <g>
        <text x="450" y="20" text-anchor="middle" fill="#cbd5e1" font-size="11" font-weight="600">probabilidades</text>
        <g v-for="(p, i) in probs" :key="'p' + i">
          <rect :x="385" :y="40 + i * 22" :width="p * 130 + 2" :height="16"
                fill="rgba(16,185,129,0.5)" stroke="#10b981" stroke-width="1.2" rx="2" />
          <text :x="389 + p * 130" :y="52 + i * 22" fill="#d1fae5" font-size="10">
            {{ (p * 100).toFixed(1) }}%
          </text>
        </g>
      </g>

      <!-- Sum verification -->
      <text x="270" y="245" text-anchor="middle" fill="#6ee7b7" font-size="11">
        Σ probabilidades = {{ sumDisplay }}
      </text>

      <defs>
        <marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
          <path d="M0,0 L0,6 L7,3 z" fill="#94a3b8" />
        </marker>
      </defs>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const scores = [2.1, -0.5, 1.4, -1.2, 0.8, 3.2, -0.1, 0.3]

const probs = computed(() => {
  const max = Math.max(...scores)
  const exps = scores.map(z => Math.exp(z - max))  // numerical stability
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map(e => e / sum)
})

const sumDisplay = computed(() => probs.value.reduce((a, b) => a + b, 0).toFixed(3))
</script>
