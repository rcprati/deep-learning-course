<template>
  <div class="early-stopping">
    <svg viewBox="0 0 360 240" class="w-full">
      <!-- Axes -->
      <line x1="40" y1="200" x2="340" y2="200" stroke="#94a3b8" stroke-width="1.5" />
      <line x1="40" y1="20"  x2="40"  y2="200" stroke="#94a3b8" stroke-width="1.5" />

      <text x="190" y="225" text-anchor="middle" fill="#cbd5e1" font-size="11">épocas</text>
      <text x="28"  y="110" text-anchor="middle" transform="rotate(-90 28 110)" fill="#cbd5e1" font-size="11">loss</text>

      <!-- Train loss: monotonically decreasing -->
      <path :d="trainPath" fill="none" stroke="#22d3ee" stroke-width="2.5" />
      <text x="320" y="180" fill="#67e8f9" font-size="11">treino</text>

      <!-- Val loss: U-shape -->
      <path :d="valPath" fill="none" stroke="#f59e0b" stroke-width="2.5" />
      <text x="320" y="100" fill="#fbbf24" font-size="11">validação</text>

      <!-- Stop point -->
      <line x1="170" y1="20" x2="170" y2="200" stroke="#10b981" stroke-width="1.5" stroke-dasharray="5 4" />
      <circle cx="170" cy="80" r="5" fill="#10b981" stroke="#0f172a" stroke-width="1.5" />
      <text x="180" y="35" fill="#6ee7b7" font-size="11" font-weight="600">early stop</text>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

function tx(x: number) { return 40 + x * 300 }
function ty(y: number) { return 200 - y * 180 }

function train(x: number) { return 0.9 * Math.exp(-2 * x) + 0.05 }
function val(x: number) {
  return Math.min(0.95, 1.4 * (x - 0.43) * (x - 0.43) + 0.18)
}

function buildPath(f: (x: number) => number) {
  const pts: string[] = []
  for (let i = 0; i <= 80; i++) {
    const x = i / 80
    pts.push((i === 0 ? 'M' : 'L') + tx(x).toFixed(1) + ',' + ty(f(x)).toFixed(1))
  }
  return pts.join(' ')
}

const trainPath = computed(() => buildPath(train))
const valPath   = computed(() => buildPath(val))
</script>
