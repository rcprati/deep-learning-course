<template>
  <div class="logreg-network">
    <svg viewBox="0 0 700 280" class="w-full max-w-3xl mx-auto">
      <!-- Connections -->
      <line x1="120" y1="80"  x2="320" y2="135" :stroke="lineColor" stroke-width="2" />
      <line x1="120" y1="200" x2="320" y2="155" :stroke="lineColor" stroke-width="2" />

      <!-- Bias -->
      <text x="320" y="60" text-anchor="middle" fill="#fbbf24" font-size="14" font-weight="600">bias = 0,4</text>
      <line x1="320" y1="70" x2="335" y2="125" stroke="#fbbf24" stroke-width="2" stroke-dasharray="3,3" />

      <!-- Input nodes -->
      <g>
        <circle cx="100" cy="80" r="32" fill="#0e7490" fill-opacity="0.4" stroke="#22d3ee" stroke-width="2" />
        <text x="100" y="78" text-anchor="middle" fill="#cffafe" font-size="13" font-weight="600">GPA</text>
        <text x="100" y="94" text-anchor="middle" fill="#67e8f9" font-size="11" v-if="showValues">{{ gpa }}</text>
      </g>
      <g>
        <circle cx="100" cy="200" r="32" fill="#0e7490" fill-opacity="0.4" stroke="#22d3ee" stroke-width="2" />
        <text x="100" y="198" text-anchor="middle" fill="#cffafe" font-size="12" font-weight="600">Exp.</text>
        <text x="100" y="214" text-anchor="middle" fill="#67e8f9" font-size="11" v-if="showValues">{{ exp }}</text>
      </g>

      <!-- Edge labels (weights) -->
      <text x="200" y="100" fill="#a78bfa" font-size="13" font-weight="600">×0,2</text>
      <text x="200" y="190" fill="#a78bfa" font-size="13" font-weight="600">×0,5</text>

      <!-- Sum node -->
      <g>
        <circle cx="350" cy="140" r="34" fill="#1e293b" stroke="#94a3b8" stroke-width="2" />
        <text x="350" y="148" text-anchor="middle" fill="#e2e8f0" font-size="22" font-weight="700">+</text>
      </g>

      <text v-if="showValues" x="430" y="120" fill="#fde68a" font-size="13">
        z = {{ z.toFixed(2) }}
      </text>

      <!-- Sigmoid arrow -->
      <line x1="384" y1="140" x2="450" y2="140" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)" />

      <!-- Sigmoid box -->
      <g>
        <rect x="450" y="105" width="100" height="70" rx="10"
              fill="#4f46e5" fill-opacity="0.3" stroke="#a78bfa" stroke-width="2" />
        <text x="500" y="138" text-anchor="middle" fill="#c4b5fd" font-size="14" font-weight="600">σ</text>
        <text x="500" y="158" text-anchor="middle" fill="#e0e7ff" font-size="11">sigmoide</text>
      </g>

      <!-- Output arrow -->
      <line x1="550" y1="140" x2="610" y2="140" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)" />

      <!-- Output -->
      <g>
        <circle cx="640" cy="140" r="28" fill="#065f46" fill-opacity="0.4" stroke="#10b981" stroke-width="2" />
        <text x="640" y="138" text-anchor="middle" fill="#d1fae5" font-size="11" font-weight="600">p</text>
        <text v-if="showValues" x="640" y="156" text-anchor="middle" fill="#6ee7b7" font-size="11">{{ p.toFixed(2) }}</text>
      </g>

      <defs>
        <marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
        </marker>
      </defs>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  gpa?: number
  exp?: number
  showValues?: boolean
}>(), {
  gpa: 0,
  exp: 0,
  showValues: false,
})

const lineColor = '#94a3b8'

const z = computed(() => 0.4 + 0.2 * props.gpa + 0.5 * props.exp)
const p = computed(() => 1 / (1 + Math.exp(-z.value)))
</script>
