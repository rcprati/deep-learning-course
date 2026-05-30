<template>
  <div class="minibatch">
    <svg viewBox="0 0 360 260" class="w-full">
      <!-- Dataset grid -->
      <g v-for="(p, i) in dots" :key="'p' + i">
        <circle :cx="p.x" :cy="p.y" :r="6" :fill="p.selected ? '#fbbf24' : '#475569'"
                :stroke="p.selected ? '#92400e' : '#1e293b'" stroke-width="1.2"
                :class="{ 'pulse': p.selected }" />
      </g>

      <!-- Frame around minibatch -->
      <text x="180" y="20" text-anchor="middle" fill="#cbd5e1" font-size="11" font-weight="600">
        {{ batchSize }} de {{ dots.length }} exemplos sorteados
      </text>

      <!-- Arrow showing batch -> gradient -->
      <line x1="180" y1="220" x2="180" y2="248" stroke="#fbbf24" stroke-width="2" marker-end="url(#a)" />
      <text x="180" y="258" text-anchor="middle" fill="#fde68a" font-size="11">∇ aproximado</text>

      <defs>
        <marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#fbbf24" />
        </marker>
      </defs>
    </svg>
    <div class="text-center text-xs opacity-70 mt-1">
      A cada passo, sorteamos um <strong>minibatch</strong> diferente.
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue'

const COLS = 18
const ROWS = 9
const BATCH = 12

const tick = ref(0)

const dots = computed(() => {
  // re-evaluate when tick changes
  void tick.value
  const all: { x: number; y: number; selected: boolean }[] = []
  // Pick BATCH random indices
  const total = COLS * ROWS
  const sel = new Set<number>()
  while (sel.size < BATCH) sel.add(Math.floor(Math.random() * total))
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const idx = r * COLS + c
      all.push({
        x: 30 + c * 17,
        y: 50 + r * 18,
        selected: sel.has(idx),
      })
    }
  }
  return all
})

const batchSize = BATCH

let timer: any
onMounted(() => { timer = setInterval(() => tick.value++, 1300) })
onUnmounted(() => clearInterval(timer))
</script>

<style scoped>
.pulse { animation: p 1.3s ease-in-out infinite; }
@keyframes p {
  0%, 100% { filter: drop-shadow(0 0 1px #fbbf24); }
  50%      { filter: drop-shadow(0 0 5px #fbbf24); }
}
</style>
