<template>
  <div class="backprop-example">
    <svg viewBox="0 0 600 280" class="w-full max-w-5xl mx-auto">
      <!-- Forward edges (blue) -->
      <g stroke="#22d3ee" stroke-width="1.6" fill="none">
        <line v-for="(e, i) in edges" :key="'fe' + i"
              :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2" marker-end="url(#fwd)" />
      </g>

      <!-- Backward edges (orange, dashed, slightly offset) -->
      <g stroke="#fb923c" stroke-width="1.6" fill="none" stroke-dasharray="5 4" stroke-opacity="0.85">
        <line v-for="(e, i) in edges" :key="'be' + i"
              :x1="e.x2" :y1="e.y2 + 8" :x2="e.x1" :y2="e.y1 + 8" marker-end="url(#bwd)" />
      </g>

      <!-- Nodes -->
      <g v-for="(n, i) in nodes" :key="'n' + i">
        <circle :cx="n.x" :cy="n.y" :r="24" :fill="n.fill" :stroke="n.stroke" stroke-width="2" />
        <text :x="n.x" :y="n.y - 1" text-anchor="middle" fill="#f8fafc" font-size="11" font-weight="700">{{ n.label }}</text>
        <text :x="n.x" :y="n.y + 12" text-anchor="middle" fill="#cbd5e1" font-size="9">{{ n.value }}</text>
      </g>

      <!-- Edge labels (gradients on the way back) -->
      <g>
        <text x="120" y="60"  fill="#fdba74" font-size="10">∂L/∂x₁ = 2(ŷ−y)·w₁₁</text>
        <text x="120" y="240" fill="#fdba74" font-size="10">∂L/∂x₂ = 2(ŷ−y)·w₂₂</text>
        <text x="270" y="65"  fill="#fdba74" font-size="10">∂L/∂a₁ = 2(ŷ−y)</text>
        <text x="270" y="235" fill="#fdba74" font-size="10">∂L/∂a₂ = 2(ŷ−y)</text>
        <text x="430" y="125" fill="#fdba74" font-size="10">∂L/∂ŷ = 2(ŷ−y)</text>
      </g>

      <!-- Legend -->
      <g>
        <line x1="20" y1="265" x2="55" y2="265" stroke="#22d3ee" stroke-width="2" marker-end="url(#fwd)" />
        <text x="62" y="269" fill="#67e8f9" font-size="11">forward</text>
        <line x1="170" y1="265" x2="205" y2="265" stroke="#fb923c" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#bwd)" />
        <text x="212" y="269" fill="#fdba74" font-size="11">backward (gradiente)</text>
      </g>

      <defs>
        <marker id="fwd" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#22d3ee" />
        </marker>
        <marker id="bwd" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#fb923c" />
        </marker>
      </defs>
    </svg>
  </div>
</template>

<script setup lang="ts">
const nodes = [
  { x: 60,  y: 80,  label: 'x₁',   value: '= 1.5', fill: 'rgba(14,116,144,0.45)', stroke: '#22d3ee' },
  { x: 60,  y: 200, label: 'x₂',   value: '= 0.8', fill: 'rgba(14,116,144,0.45)', stroke: '#22d3ee' },
  { x: 220, y: 80,  label: 'a₁',   value: 'w₁₁·x₁',fill: 'rgba(245,158,11,0.35)', stroke: '#f59e0b' },
  { x: 220, y: 200, label: 'a₂',   value: 'w₂₂·x₂',fill: 'rgba(245,158,11,0.35)', stroke: '#f59e0b' },
  { x: 380, y: 140, label: 'ŷ',    value: 'b+a₁+a₂', fill: 'rgba(99,102,241,0.4)',stroke: '#a78bfa' },
  { x: 540, y: 140, label: 'L',    value: '(ŷ−y)²', fill: 'rgba(239,68,68,0.4)', stroke: '#ef4444' },
]

const edges = [
  { x1: 84,  y1: 80,  x2: 196, y2: 80  },
  { x1: 84,  y1: 200, x2: 196, y2: 200 },
  { x1: 244, y1: 80,  x2: 358, y2: 134 },
  { x1: 244, y1: 200, x2: 358, y2: 146 },
  { x1: 404, y1: 140, x2: 516, y2: 140 },
]
</script>
