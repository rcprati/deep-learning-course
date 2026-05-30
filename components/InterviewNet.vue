<template>
  <div class="interview-net">
    <svg viewBox="0 0 760 360" class="w-full max-w-4xl mx-auto">
      <!-- Connections input -> hidden -->
      <g>
        <g v-for="(w, i) in W1" :key="'w1' + i">
          <line
            :x1="80 + 28" :y1="inputY(w.from)"
            :x2="320 - 28" :y2="hiddenY(w.to)"
            :stroke="weightColor(w.value)"
            :stroke-width="1.4 + Math.min(2, Math.abs(w.value) * 4)"
            stroke-opacity="0.7"
          />
        </g>
      </g>

      <!-- Connections hidden -> output -->
      <g>
        <g v-for="(w, i) in W2" :key="'w2' + i">
          <line
            :x1="320 + 28" :y1="hiddenY(w.from)"
            :x2="540 - 28" :y2="180"
            :stroke="weightColor(w.value)"
            :stroke-width="1.4 + Math.min(2, Math.abs(w.value) * 4)"
            stroke-opacity="0.7"
          />
        </g>
      </g>

      <!-- Bias labels (hidden) -->
      <g class="bias">
        <text :x="320" :y="hiddenY(0) - 32" text-anchor="middle" fill="#fbbf24" font-size="10">b₁ = -0,3</text>
        <text :x="320" :y="hiddenY(1) - 32" text-anchor="middle" fill="#fbbf24" font-size="10">b₂ = 0,2</text>
        <text :x="320" :y="hiddenY(2) - 32" text-anchor="middle" fill="#fbbf24" font-size="10">b₃ = 0,5</text>
      </g>

      <!-- Bias output -->
      <text :x="540" y="148" text-anchor="middle" fill="#fbbf24" font-size="10">b = 0,05</text>

      <!-- Input nodes -->
      <g>
        <circle :cx="80" :cy="inputY(0)" r="28" fill="rgba(14,116,144,0.4)" stroke="#22d3ee" stroke-width="2" />
        <text :x="80" :y="inputY(0)" text-anchor="middle" fill="#cffafe" font-size="12" font-weight="600">x₁</text>
        <text v-if="showActivations" :x="32" :y="inputY(0) + 4" text-anchor="end" fill="#fde68a" font-size="11">{{ x1 }}</text>
      </g>
      <g>
        <circle :cx="80" :cy="inputY(1)" r="28" fill="rgba(14,116,144,0.4)" stroke="#22d3ee" stroke-width="2" />
        <text :x="80" :y="inputY(1)" text-anchor="middle" fill="#cffafe" font-size="12" font-weight="600">x₂</text>
        <text v-if="showActivations" :x="32" :y="inputY(1) + 4" text-anchor="end" fill="#fde68a" font-size="11">{{ x2 }}</text>
      </g>

      <!-- Hidden nodes -->
      <g v-for="i in 3" :key="'h' + i">
        <circle :cx="320" :cy="hiddenY(i - 1)" r="28" fill="rgba(245,158,11,0.3)" stroke="#f59e0b" stroke-width="2" />
        <text :x="320" :y="hiddenY(i - 1) - 1" text-anchor="middle" fill="#fef3c7" font-size="11" font-weight="600">a{{ i }}</text>
        <text :x="320" :y="hiddenY(i - 1) + 12" text-anchor="middle" fill="#fbbf24" font-size="9">ReLU</text>
      </g>

      <!-- Activation values -->
      <g v-if="showActivations">
        <text :x="362" :y="hiddenY(0) + 4" fill="#fde68a" font-size="11">= {{ a1.toFixed(2) }}</text>
        <text :x="362" :y="hiddenY(1) + 4" fill="#fde68a" font-size="11">= {{ a2.toFixed(2) }}</text>
        <text :x="362" :y="hiddenY(2) + 4" fill="#fde68a" font-size="11">= {{ a3.toFixed(2) }}</text>
      </g>

      <!-- Output node -->
      <g>
        <circle :cx="540" :cy="180" r="32" fill="rgba(6,95,70,0.4)" stroke="#10b981" stroke-width="2" />
        <text :x="540" :y="178" text-anchor="middle" fill="#d1fae5" font-size="11" font-weight="600">y</text>
        <text :x="540" :y="194" text-anchor="middle" fill="#6ee7b7" font-size="9">σ</text>
      </g>

      <!-- Final value -->
      <g v-if="showActivations">
        <line x1="572" y1="180" x2="660" y2="180" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr2)" />
        <text :x="690" :y="184" text-anchor="middle" fill="#fde68a" font-size="14" font-weight="700">{{ y.toFixed(3) }}</text>
      </g>

      <!-- Layer labels -->
      <text :x="80"  y="40" text-anchor="middle" fill="#cbd5e1" font-size="12" font-weight="600">Entrada</text>
      <text :x="320" y="40" text-anchor="middle" fill="#cbd5e1" font-size="12" font-weight="600">Camada oculta</text>
      <text :x="540" y="40" text-anchor="middle" fill="#cbd5e1" font-size="12" font-weight="600">Saída</text>

      <!-- Weight labels (selected) -->
      <text :x="200" :y="105" fill="#a78bfa" font-size="9">0,5</text>
      <text :x="200" :y="180" fill="#a78bfa" font-size="9">-0,1</text>
      <text :x="200" :y="265" fill="#a78bfa" font-size="9">0,2</text>
      <text :x="430" :y="155" fill="#a78bfa" font-size="9">-0,2</text>
      <text :x="430" :y="200" fill="#a78bfa" font-size="9">-0,3</text>
      <text :x="430" :y="245" fill="#a78bfa" font-size="9">-0,15</text>

      <defs>
        <marker id="arr2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
        </marker>
      </defs>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  x1?: number
  x2?: number
  showActivations?: boolean
}>(), {
  x1: 0,
  x2: 0,
  showActivations: false,
})

function inputY(i: number) { return 130 + i * 100 }
function hiddenY(i: number) { return 100 + i * 80 }

// Hardcoded illustrative weights
const W1 = [
  // from input 0 (x1) to hidden 0..2
  { from: 0, to: 0, value:  0.5 },
  { from: 0, to: 1, value: -0.1 },
  { from: 0, to: 2, value:  0.2 },
  // from input 1 (x2) to hidden 0..2
  { from: 1, to: 0, value:  0.1 },
  { from: 1, to: 1, value:  0.3 },
  { from: 1, to: 2, value: -0.1 },
]
const W2 = [
  { from: 0, value: -0.2  },
  { from: 1, value: -0.3  },
  { from: 2, value: -0.15 },
]
const b1 = [-0.3, 0.2, 0.5]
const b2 = 0.05

function weightColor(v: number) {
  return v >= 0 ? '#a78bfa' : '#f87171'
}

const a1 = computed(() => Math.max(0, b1[0] + 0.5 * props.x1 + 0.1 * props.x2))
const a2 = computed(() => Math.max(0, b1[1] + (-0.1) * props.x1 + 0.3 * props.x2))
const a3 = computed(() => Math.max(0, b1[2] + 0.2 * props.x1 + (-0.1) * props.x2))

const y = computed(() => {
  const z = b2 + W2[0].value * a1.value + W2[1].value * a2.value + W2[2].value * a3.value
  return 1 / (1 + Math.exp(-z))
})
</script>
