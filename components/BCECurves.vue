<template>
  <div class="bce-curves grid grid-cols-2 gap-6 max-w-4xl mx-auto">
    <div v-for="c in curves" :key="c.title" class="text-center">
      <div class="text-sm font-bold text-indigo-300 mb-2">{{ c.title }}</div>
      <svg viewBox="0 0 280 200" class="w-full">
        <!-- Axes -->
        <line x1="40" y1="160" x2="260" y2="160" stroke="#94a3b8" stroke-width="1" />
        <line x1="40" y1="20"  x2="40"  y2="160" stroke="#94a3b8" stroke-width="1" />

        <!-- Ticks -->
        <text x="40"  y="175" text-anchor="middle" fill="#94a3b8" font-size="9">0</text>
        <text x="260" y="175" text-anchor="middle" fill="#94a3b8" font-size="9">1</text>
        <text x="32"  y="160" text-anchor="end"    fill="#94a3b8" font-size="9">0</text>
        <text x="32"  y="25"  text-anchor="end"    fill="#94a3b8" font-size="9">∞</text>

        <!-- Curve -->
        <path :d="c.path" fill="none" :stroke="c.color" stroke-width="2.5" />

        <!-- Axis labels -->
        <text x="150" y="195" text-anchor="middle" fill="#cbd5e1" font-size="10">probabilidade prevista (p̂)</text>
        <text x="20"  y="100" text-anchor="middle" transform="rotate(-90 20 100)" fill="#cbd5e1" font-size="10">loss</text>
      </svg>
      <div class="text-xs opacity-80 mt-2">{{ c.formula }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
function buildPath(f: (p: number) => number) {
  const pts: string[] = []
  for (let i = 1; i <= 99; i++) {
    const p = i / 100
    const x = 40 + p * 220
    const v = f(p)
    const y = 160 - Math.min(140, v * 30)
    pts.push((i === 1 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1))
  }
  return pts.join(' ')
}

const curves = [
  {
    title: 'Loss para y = 1',
    formula: '−log(p̂) — punir p̂ baixo',
    color:   '#22d3ee',
    path:    buildPath(p => -Math.log(p)),
  },
  {
    title: 'Loss para y = 0',
    formula: '−log(1 − p̂) — punir p̂ alto',
    color:   '#f59e0b',
    path:    buildPath(p => -Math.log(1 - p)),
  },
]
</script>
