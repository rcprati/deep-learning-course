<template>
  <div class="simple-polynomial">
    <svg viewBox="0 0 320 240" class="w-full">
      <!-- Axes -->
      <line x1="20" y1="200" x2="300" y2="200" stroke="#94a3b8" stroke-width="1" />
      <line x1="160" y1="20" x2="160" y2="220" stroke="#94a3b8" stroke-width="1" />
      <text x="295" y="195" fill="#94a3b8" font-size="10">w</text>
      <text x="166" y="22"  fill="#94a3b8" font-size="10">g(w)</text>

      <!-- Curve -->
      <path :d="path" fill="none" stroke="#a78bfa" stroke-width="2.5" />

      <!-- Min point -->
      <circle :cx="160 + minW * 30" cy="200" r="4" fill="#10b981" />
      <text :x="160 + minW * 30 + 8" y="215" fill="#6ee7b7" font-size="10">mínimo</text>
    </svg>
    <div class="text-center text-xs opacity-70 mt-1">
      Como achar o mínimo de uma função qualquer?
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

// g(w) = 0.4 * (w - 1)^2 + 0.5  (a parabola with minimum at w=1)
function g(w: number) { return 0.4 * (w - 1) * (w - 1) + 0.5 }

const minW = 1

const path = computed(() => {
  const pts: string[] = []
  for (let i = 0; i <= 60; i++) {
    const w = -3 + (i * 6) / 60
    const x = 160 + w * 30
    const y = 200 - g(w) * 30
    if (y > 20 && y < 220) pts.push((pts.length === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1))
  }
  return pts.join(' ')
})
</script>
