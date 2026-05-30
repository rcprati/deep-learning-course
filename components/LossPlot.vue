<template>
  <div class="loss-plot">
    <svg viewBox="0 0 360 220" class="w-full">
      <!-- Axes -->
      <line x1="40" y1="180" x2="340" y2="180" stroke="#94a3b8" stroke-width="1" />
      <line x1="40" y1="20"  x2="40"  y2="180" stroke="#94a3b8" stroke-width="1" />

      <!-- Tick at 0 -->
      <line x1="190" y1="178" x2="190" y2="182" stroke="#94a3b8" />
      <text x="190" y="195" text-anchor="middle" fill="#94a3b8" font-size="9">0</text>

      <!-- Curve -->
      <path :d="path" fill="none" stroke="#a78bfa" stroke-width="2.5" />

      <!-- Highlight at zero -->
      <circle cx="190" cy="180" r="4" fill="#10b981" />

      <!-- Labels -->
      <text x="335" y="174" fill="#94a3b8" font-size="10">erro</text>
      <text x="50"  y="30"  fill="#94a3b8" font-size="10">loss</text>
      <text x="190" y="14"  text-anchor="middle" fill="#cbd5e1" font-size="11" font-weight="600">{{ title }}</text>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{ kind?: 'mse' | 'mae' }>(), { kind: 'mse' })

const title = computed(() => props.kind === 'mse' ? 'MSE = (y − ŷ)²' : 'MAE = |y − ŷ|')

const path = computed(() => {
  const pts: string[] = []
  for (let i = 0; i <= 60; i++) {
    const e = -3 + (i * 6) / 60
    const x = 190 + e * 50
    const v = props.kind === 'mae' ? Math.abs(e) : e * e
    const y = 180 - Math.min(160, v * 20)
    pts.push((i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1))
  }
  return pts.join(' ')
})
</script>
