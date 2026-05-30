<template>
  <div class="rgb-image flex justify-center items-center gap-6">
    <!-- Combined RGB -->
    <div class="text-center">
      <div class="text-xs opacity-70 mb-2">imagem RGB</div>
      <div class="grid" :style="gridStyle">
        <div v-for="(p, i) in pixels" :key="i"
             :style="{ width: cell + 'px', height: cell + 'px',
                       background: `rgb(${p[0]}, ${p[1]}, ${p[2]})` }" />
      </div>
    </div>

    <div class="text-3xl opacity-50">=</div>

    <!-- Three channels stacked -->
    <div class="flex flex-col gap-2">
      <div v-for="ch in channels" :key="ch.name" class="flex items-center gap-2">
        <div class="grid" :style="gridStyle">
          <div v-for="(p, i) in pixels" :key="i"
               :style="{ width: cell + 'px', height: cell + 'px',
                         background: ch.color(p) }" />
        </div>
        <div class="text-xs font-bold" :style="{ color: ch.label }">{{ ch.name }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const cols = 10
const cell = 14

// procedural color image: gradient + colored dot
function makePixels() {
  const out: [number, number, number][] = []
  for (let r = 0; r < cols; r++) {
    for (let c = 0; c < cols; c++) {
      const dx = c - 5
      const dy = r - 5
      const dist = Math.sqrt(dx * dx + dy * dy)
      let rgb: [number, number, number] = [50 + c * 18, 80 + r * 14, 180]
      if (dist < 3.5) rgb = [240, 90, 90]   // red dot
      if (dist < 1.5) rgb = [255, 200, 60]  // yellow center
      out.push(rgb)
    }
  }
  return out
}

const pixels = makePixels()

const gridStyle = computed(() => ({
  gridTemplateColumns: `repeat(${cols}, ${cell}px)`,
  gap: '1px',
}))

const channels = [
  { name: 'R', label: '#f87171', color: (p: number[]) => `rgb(${p[0]}, 0, 0)` },
  { name: 'G', label: '#4ade80', color: (p: number[]) => `rgb(0, ${p[1]}, 0)` },
  { name: 'B', label: '#60a5fa', color: (p: number[]) => `rgb(0, 0, ${p[2]})` },
]
</script>
