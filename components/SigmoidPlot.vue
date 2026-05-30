<template>
  <div class="sigmoid-plot">
    <svg viewBox="0 0 400 280" class="w-full max-w-md mx-auto">
      <!-- Grid -->
      <g stroke="#334155" stroke-width="0.5" opacity="0.5">
        <line v-for="i in 10" :key="'v'+i" :x1="40 + i * 32" y1="20" :x2="40 + i * 32" y2="240" />
        <line v-for="i in 6" :key="'h'+i" x1="40" :y1="20 + i * 36" x2="360" :y2="20 + i * 36" />
      </g>

      <!-- Axes -->
      <line x1="40" y1="130" x2="360" y2="130" stroke="#94a3b8" stroke-width="1.5" />
      <line x1="200" y1="20" x2="200" y2="240" stroke="#94a3b8" stroke-width="1.5" />

      <!-- Axis labels -->
      <text x="350" y="125" fill="#94a3b8" font-size="11">z</text>
      <text x="206" y="28" fill="#94a3b8" font-size="11">σ(z)</text>
      <text x="35" y="32" fill="#94a3b8" font-size="10" text-anchor="end">1</text>
      <text x="35" y="135" fill="#94a3b8" font-size="10" text-anchor="end">0.5</text>
      <text x="35" y="240" fill="#94a3b8" font-size="10" text-anchor="end">0</text>

      <!-- Sigmoid curve -->
      <path
        :d="path"
        fill="none"
        stroke="#a78bfa"
        stroke-width="2.5"
      />

      <!-- 0.5 dashed line -->
      <line x1="40" y1="130" x2="360" y2="130" stroke="#fbbf24" stroke-width="1" stroke-dasharray="4,4" opacity="0.6" />
    </svg>
    <div class="text-center text-xs opacity-70 mt-1">
      σ(z) = 1 / (1 + e<sup>−z</sup>) — saída em (0, 1)
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

function sig(z: number) { return 1 / (1 + Math.exp(-z)) }

const path = computed(() => {
  const pts: string[] = []
  // x range: -8 .. 8 -> mapped to 40..360
  // y range: 0..1 -> mapped to 240..20
  for (let i = 0; i <= 80; i++) {
    const z = -8 + (i * 16) / 80
    const x = 40 + ((z + 8) / 16) * 320
    const y = 240 - sig(z) * 220
    pts.push((i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1))
  }
  return pts.join(' ')
})
</script>
