<template>
  <div class="dropout-viz">
    <svg viewBox="0 0 380 240" class="w-full">
      <!-- Connections -->
      <g stroke="#cbd5e1" stroke-width="1.2" stroke-opacity="0.55">
        <line v-for="(c, i) in connections" :key="'c' + i"
              :x1="c.x1" :y1="c.y1" :x2="c.x2" :y2="c.y2"
              :stroke-opacity="c.dropped ? 0.1 : 0.6" />
      </g>

      <!-- Layer labels -->
      <text x="60"  y="20" text-anchor="middle" fill="#cbd5e1" font-size="11" font-weight="600">camada n</text>
      <text x="200" y="20" text-anchor="middle" fill="#cbd5e1" font-size="11" font-weight="600">dropout (p = 0.5)</text>
      <text x="340" y="20" text-anchor="middle" fill="#cbd5e1" font-size="11" font-weight="600">camada n + 1</text>

      <!-- Neurons -->
      <g v-for="(n, i) in neurons" :key="'n' + i">
        <circle :cx="n.x" :cy="n.y" :r="n.r"
                :fill="n.dropped ? 'rgba(71,85,105,0.4)' : n.fill"
                :stroke="n.dropped ? '#475569' : n.stroke" stroke-width="2"
                :class="{ dropped: n.dropped }" />
        <text v-if="n.dropped" :x="n.x" :y="n.y + 4" text-anchor="middle" fill="#94a3b8" font-size="11">×</text>
        <text v-else :x="n.x" :y="n.y + 4" text-anchor="middle" fill="#f8fafc" font-size="11" font-weight="700">+</text>
      </g>
    </svg>
    <div class="text-center text-xs opacity-70 mt-1">
      Em cada passo de treino, sorteamos quais neurônios "desligar".
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue'

const tick = ref(0)

const fixed = [
  // left layer
  { x: 60, y: 60,  r: 14, fill: 'rgba(14,116,144,0.45)',  stroke: '#22d3ee', layer: 0 },
  { x: 60, y: 110, r: 14, fill: 'rgba(14,116,144,0.45)',  stroke: '#22d3ee', layer: 0 },
  { x: 60, y: 160, r: 14, fill: 'rgba(14,116,144,0.45)',  stroke: '#22d3ee', layer: 0 },
  // middle (dropout) layer — 6 neurons
  { x: 200, y: 50,  r: 14, fill: 'rgba(245,158,11,0.35)', stroke: '#f59e0b', layer: 1 },
  { x: 200, y: 90,  r: 14, fill: 'rgba(245,158,11,0.35)', stroke: '#f59e0b', layer: 1 },
  { x: 200, y: 130, r: 14, fill: 'rgba(245,158,11,0.35)', stroke: '#f59e0b', layer: 1 },
  { x: 200, y: 170, r: 14, fill: 'rgba(245,158,11,0.35)', stroke: '#f59e0b', layer: 1 },
  { x: 200, y: 210, r: 14, fill: 'rgba(245,158,11,0.35)', stroke: '#f59e0b', layer: 1 },
  // right layer
  { x: 340, y: 80,  r: 14, fill: 'rgba(6,95,70,0.45)',    stroke: '#10b981', layer: 2 },
  { x: 340, y: 140, r: 14, fill: 'rgba(6,95,70,0.45)',    stroke: '#10b981', layer: 2 },
]

const neurons = computed(() => {
  void tick.value
  return fixed.map(n => ({
    ...n,
    dropped: n.layer === 1 && Math.random() < 0.5,
  }))
})

const connections = computed(() => {
  void tick.value
  const left   = neurons.value.filter(n => n.layer === 0)
  const middle = neurons.value.filter(n => n.layer === 1)
  const right  = neurons.value.filter(n => n.layer === 2)
  const out: { x1: number; y1: number; x2: number; y2: number; dropped: boolean }[] = []
  for (const a of left) for (const b of middle)
    out.push({ x1: a.x + a.r, y1: a.y, x2: b.x - b.r, y2: b.y, dropped: b.dropped })
  for (const a of middle) for (const b of right)
    out.push({ x1: a.x + a.r, y1: a.y, x2: b.x - b.r, y2: b.y, dropped: a.dropped })
  return out
})

let timer: any
onMounted(() => { timer = setInterval(() => tick.value++, 1300) })
onUnmounted(() => clearInterval(timer))
</script>

<style scoped>
.dropped { transition: all 0.3s; }
</style>
