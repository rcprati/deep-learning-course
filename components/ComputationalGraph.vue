<template>
  <div class="comp-graph">
    <svg viewBox="0 0 480 280" class="w-full">
      <!-- Connections -->
      <g stroke="#94a3b8" stroke-width="1.4" fill="none">
        <line v-for="(e, i) in edges" :key="'e' + i"
              :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2" marker-end="url(#arrow)" />
      </g>

      <!-- Backward arrows (for full variant) -->
      <g v-if="variant === 'full'" stroke="#fb923c" stroke-width="1.4" fill="none" stroke-dasharray="4 3">
        <line v-for="(e, i) in edges" :key="'b' + i"
              :x1="e.x2" :y1="e.y2 + 4" :x2="e.x1" :y2="e.y1 + 4" marker-end="url(#arrowOrange)" />
      </g>

      <!-- Nodes -->
      <g v-for="(n, i) in nodes" :key="'n' + i">
        <circle :cx="n.x" :cy="n.y" :r="22" :fill="n.fill" :stroke="n.stroke" stroke-width="2" />
        <text :x="n.x" :y="n.y + 4" text-anchor="middle" fill="#f8fafc" font-size="11" font-weight="700">{{ n.label }}</text>
      </g>

      <!-- Forward / backward legend (full variant) -->
      <g v-if="variant === 'full'">
        <line x1="20" y1="260" x2="50" y2="260" stroke="#94a3b8" stroke-width="1.6" marker-end="url(#arrow)" />
        <text x="56" y="263" fill="#cbd5e1" font-size="10">forward (valores)</text>
        <line x1="200" y1="260" x2="230" y2="260" stroke="#fb923c" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#arrowOrange)" />
        <text x="236" y="263" fill="#fdba74" font-size="10">backward (gradientes)</text>
      </g>

      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
        </marker>
        <marker id="arrowOrange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#fb923c" />
        </marker>
      </defs>
    </svg>
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{ variant?: 'simple' | 'full' }>(), { variant: 'simple' })

// Layout: x1, x2 -> a1, a2 -> y -> loss
const nodes = [
  { x: 60,  y: 80,  label: 'x₁',   fill: 'rgba(14,116,144,0.45)', stroke: '#22d3ee' },
  { x: 60,  y: 180, label: 'x₂',   fill: 'rgba(14,116,144,0.45)', stroke: '#22d3ee' },
  { x: 180, y: 80,  label: 'a₁',   fill: 'rgba(245,158,11,0.35)', stroke: '#f59e0b' },
  { x: 180, y: 180, label: 'a₂',   fill: 'rgba(245,158,11,0.35)', stroke: '#f59e0b' },
  { x: 300, y: 130, label: 'ŷ',    fill: 'rgba(99,102,241,0.4)',  stroke: '#a78bfa' },
  { x: 420, y: 130, label: 'loss', fill: 'rgba(239,68,68,0.4)',   stroke: '#ef4444' },
]

const edges = [
  { x1: 82,  y1: 80,  x2: 158, y2: 80  }, // x1 -> a1
  { x1: 82,  y1: 180, x2: 158, y2: 180 }, // x2 -> a2
  { x1: 202, y1: 80,  x2: 280, y2: 124 }, // a1 -> y
  { x1: 202, y1: 180, x2: 280, y2: 136 }, // a2 -> y
  { x1: 322, y1: 130, x2: 398, y2: 130 }, // y -> loss
]
</script>
