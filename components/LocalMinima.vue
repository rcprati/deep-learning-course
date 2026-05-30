<template>
  <div class="local-minima">
    <svg viewBox="0 0 360 220" class="w-full">
      <!-- Curve with multiple minima -->
      <path :d="path" fill="none" stroke="#a78bfa" stroke-width="2.5" />

      <!-- Axes -->
      <line x1="20" y1="200" x2="340" y2="200" stroke="#94a3b8" stroke-width="1" />

      <!-- Local min -->
      <circle :cx="localMinX" :cy="localMinY" r="5" fill="#f59e0b" />
      <text :x="localMinX" :y="localMinY - 10" text-anchor="middle" fill="#fbbf24" font-size="10">mín. local</text>

      <!-- Global min -->
      <circle :cx="globalMinX" :cy="globalMinY" r="5" fill="#10b981" />
      <text :x="globalMinX" :y="globalMinY - 10" text-anchor="middle" fill="#6ee7b7" font-size="10">mín. global</text>

      <!-- Saddle-ish flat point -->
      <circle :cx="saddleX" :cy="saddleY" r="5" fill="#22d3ee" />
      <text :x="saddleX" :y="saddleY - 10" text-anchor="middle" fill="#67e8f9" font-size="10">platô</text>
    </svg>
    <div class="text-center text-xs opacity-70 mt-1">
      Paisagens reais têm mínimos locais e platôs — nem sempre o GD acha o mínimo global.
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

// f(x) = 0.04*x^2 - 0.5*sin(x) + 1.2
function f(x: number) { return 0.04 * x * x - 0.5 * Math.sin(x) + 1.2 }

function toX(x: number) { return 180 + x * 20 }
function toY(v: number) { return 200 - v * 50 }

const path = computed(() => {
  const pts: string[] = []
  for (let i = 0; i <= 100; i++) {
    const x = -8 + (i * 16) / 100
    const px = toX(x)
    const py = toY(f(x))
    pts.push((i === 0 ? 'M' : 'L') + px.toFixed(1) + ',' + py.toFixed(1))
  }
  return pts.join(' ')
})

// Approximate min locations in x: from -df/dx = 0:  0.08x = 0.5 cos(x)
// Hardcoded points for visual purposes
const localMinX  = toX(-4.7)
const localMinY  = toY(f(-4.7))
const globalMinX = toX(1.5)
const globalMinY = toY(f(1.5))
const saddleX    = toX(5)
const saddleY    = toY(f(5))
</script>
