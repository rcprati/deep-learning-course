<template>
  <div class="overfit-curve">
    <svg viewBox="0 0 480 260" class="w-full max-w-3xl mx-auto">
      <!-- Axes -->
      <line x1="50" y1="220" x2="450" y2="220" stroke="#94a3b8" stroke-width="1.5" />
      <line x1="50" y1="20"  x2="50"  y2="220" stroke="#94a3b8" stroke-width="1.5" />

      <!-- Axis labels -->
      <text x="250" y="248" text-anchor="middle" fill="#cbd5e1" font-size="11">complexidade do modelo →</text>
      <text x="38"  y="120" text-anchor="middle" transform="rotate(-90 38 120)" fill="#cbd5e1" font-size="11">erro</text>

      <!-- Training error: monotonically decreasing -->
      <path :d="trainPath" fill="none" stroke="#22d3ee" stroke-width="2.5" />
      <text x="420" y="195" fill="#67e8f9" font-size="11">treino</text>

      <!-- Validation error: U-shaped -->
      <path :d="valPath" fill="none" stroke="#f59e0b" stroke-width="2.5" />
      <text x="420" y="100" fill="#fbbf24" font-size="11">validação</text>

      <!-- Sweet spot vertical line -->
      <line x1="240" y1="20" x2="240" y2="220" stroke="#10b981" stroke-width="1.2" stroke-dasharray="4 4" />
      <text x="240" y="14" text-anchor="middle" fill="#6ee7b7" font-size="10">"sweet spot"</text>

      <!-- Underfit / overfit zones -->
      <text x="120" y="40" text-anchor="middle" fill="#fbbf24" font-size="10" opacity="0.85">underfit</text>
      <text x="370" y="40" text-anchor="middle" fill="#fb7185" font-size="10" opacity="0.85">overfit</text>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

// x-domain 0..1 mapped to 50..450 (400 px wide)
// y-domain 0..1 mapped to 220..20 (200 px tall)
function tx(x: number) { return 50 + x * 400 }
function ty(y: number) { return 220 - y * 200 }

// Training error: starts high, decreases monotonically to ~0.05
// Approximate with: 0.95 * exp(-3x) + 0.05
function trainErr(x: number) { return 0.95 * Math.exp(-3 * x) + 0.05 }
// Validation error: U-shape. f(x) = 0.6 * (x - 0.475)^2 + 0.25  shifted/scaled
function valErr(x: number) {
  const base = 1.6 * (x - 0.475) * (x - 0.475) + 0.25
  // Add penalty on the left (underfit) and rise on the right (overfit)
  return Math.min(0.95, base)
}

function buildPath(f: (x: number) => number) {
  const pts: string[] = []
  for (let i = 0; i <= 80; i++) {
    const x = i / 80
    const px = tx(x), py = ty(f(x))
    pts.push((i === 0 ? 'M' : 'L') + px.toFixed(1) + ',' + py.toFixed(1))
  }
  return pts.join(' ')
}

const trainPath = computed(() => buildPath(trainErr))
const valPath   = computed(() => buildPath(valErr))
</script>
