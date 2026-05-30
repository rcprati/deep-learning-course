<template>
  <div class="activation-fns flex justify-center gap-4">
    <div v-for="fn in fns" :key="fn.name" class="text-center">
      <div class="font-bold text-indigo-300 mb-1">{{ fn.name }}</div>
      <svg viewBox="0 0 200 160" class="w-56">
        <!-- Grid -->
        <line x1="20" y1="80" x2="180" y2="80" stroke="#64748b" stroke-width="1" />
        <line x1="100" y1="20" x2="100" y2="140" stroke="#64748b" stroke-width="1" />

        <!-- Curve -->
        <path
          :d="fn.path"
          fill="none"
          :stroke="fn.color"
          stroke-width="2.5"
        />

        <!-- Range -->
        <text x="100" y="155" text-anchor="middle" fill="#94a3b8" font-size="9">{{ fn.range }}</text>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
function buildPath(f: (z: number) => number, ymin: number, ymax: number) {
  const pts: string[] = []
  for (let i = 0; i <= 80; i++) {
    const z = -5 + (i * 10) / 80
    const x = 20 + ((z + 5) / 10) * 160
    const yNorm = (f(z) - ymin) / (ymax - ymin)
    const y = 140 - yNorm * 120
    pts.push((i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + Math.max(20, Math.min(140, y)).toFixed(1))
  }
  return pts.join(' ')
}

const sig = (z: number) => 1 / (1 + Math.exp(-z))
const tanh = (z: number) => Math.tanh(z)
const relu = (z: number) => Math.max(0, z)

const fns = [
  { name: 'Sigmoide',  color: '#22d3ee', path: buildPath(sig,  0, 1),  range: '(0, 1)' },
  { name: 'Tanh',      color: '#a78bfa', path: buildPath(tanh, -1, 1), range: '(-1, 1)' },
  { name: 'ReLU',      color: '#f59e0b', path: buildPath(relu,  0, 5), range: '[0, ∞)' },
]
</script>
