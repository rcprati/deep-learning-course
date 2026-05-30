<template>
  <div class="tensor-ranks grid grid-cols-5 gap-3 max-w-5xl mx-auto">
    <div v-for="t in ranks" :key="t.rank" class="text-center">
      <div class="font-bold text-indigo-300 text-sm">rank {{ t.rank }}</div>
      <div class="text-xs opacity-70 mb-1">{{ t.name }}</div>

      <svg viewBox="0 0 100 100" class="w-full">
        <component :is="t.draw" />
      </svg>

      <div class="font-mono text-xs mt-1 text-amber-300">{{ t.example }}</div>
      <div class="text-xs opacity-60">{{ t.hint }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { h } from 'vue'

function Scalar()  { return h('rect', { x: 40, y: 40, width: 20, height: 20, fill: 'rgba(34,211,238,0.4)', stroke: '#22d3ee', 'stroke-width': 1.5, rx: 3 }) }
function Vector()  {
  return [0, 1, 2, 3].map(i =>
    h('rect', { key: i, x: 25 + i * 12, y: 40, width: 12, height: 20, fill: 'rgba(34,211,238,0.4)', stroke: '#22d3ee', 'stroke-width': 1, rx: 2 }))
}
function Matrix() {
  const cells: any[] = []
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++)
      cells.push(h('rect', { key: r + '-' + c, x: 22 + c * 14, y: 18 + r * 14, width: 14, height: 14, fill: 'rgba(34,211,238,0.4)', stroke: '#22d3ee', 'stroke-width': 0.8, rx: 1 }))
  return cells
}
function Cube() {
  const front: any[] = []
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      front.push(h('rect', { key: 'f' + r + c, x: 22 + c * 14, y: 30 + r * 14, width: 14, height: 14, fill: 'rgba(34,211,238,0.4)', stroke: '#22d3ee', 'stroke-width': 0.8 }))
  // back face (offset)
  const back: any[] = []
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      back.push(h('rect', { key: 'b' + r + c, x: 32 + c * 14, y: 20 + r * 14, width: 14, height: 14, fill: 'rgba(34,211,238,0.2)', stroke: '#22d3ee', 'stroke-width': 0.6 }))
  return [...back, ...front]
}
function Hyper() {
  // Stack of cubes
  const els: any[] = []
  for (let s = 0; s < 3; s++) {
    const ox = s * 6, oy = -s * 6
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++)
        els.push(h('rect', {
          key: s + '-' + r + '-' + c,
          x: 18 + c * 12 + ox, y: 38 + r * 12 + oy, width: 12, height: 12,
          fill: `rgba(34,211,238,${0.45 - s * 0.12})`,
          stroke: '#22d3ee', 'stroke-width': 0.5,
        }))
  }
  return els
}

const ranks = [
  { rank: 0, name: 'escalar',  example: '42',           hint: 'um número',                draw: Scalar },
  { rank: 1, name: 'vetor',    example: '[a, b, c, d]', hint: 'lista de números',         draw: Vector },
  { rank: 2, name: 'matriz',   example: 'imagem cinza', hint: 'H × W',                    draw: Matrix },
  { rank: 3, name: 'cubo',     example: 'imagem RGB',   hint: 'H × W × 3',                draw: Cube  },
  { rank: 4, name: 'rank-4',   example: 'batch RGB',    hint: 'N × H × W × 3',            draw: Hyper },
]
</script>
