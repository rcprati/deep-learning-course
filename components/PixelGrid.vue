<template>
  <div class="pixel-grid flex justify-center items-center gap-8">
    <!-- The "image" - cat shape made of pixels -->
    <div>
      <div class="text-xs opacity-70 mb-2 text-center">"O que vemos"</div>
      <div
        class="grid"
        :style="{
          gridTemplateColumns: `repeat(${cols}, ${cell}px)`,
          gap: '1px',
        }"
      >
        <div
          v-for="(p, i) in pixels"
          :key="i"
          :style="{
            width: cell + 'px',
            height: cell + 'px',
            background: `rgb(${p[0]}, ${p[1]}, ${p[2]})`,
          }"
        />
      </div>
    </div>

    <!-- Arrow -->
    <div class="text-3xl opacity-50">→</div>

    <!-- The numbers -->
    <div>
      <div class="text-xs opacity-70 mb-2 text-center">"O que o computador vê"</div>
      <div class="font-mono text-[10px] leading-snug bg-slate-900/60 p-3 rounded border border-slate-700 max-h-72 overflow-hidden">
        <div v-for="(row, ri) in matrixRows" :key="ri" class="whitespace-nowrap flex gap-2">
          <span
            v-for="(p, ci) in row"
            :key="ci"
            class="inline-block tabular-nums"
            :style="{ color: `rgb(${p[0]}, ${p[1]}, ${p[2]})`, minWidth: '5.5rem' }"
          >({{ p[0] }},{{ p[1] }},{{ p[2] }})</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const cols = 18
const rows = 14
const cell = 16

// Generate a simple cat-like silhouette using a procedural rule
function makePixels() {
  const pixels: [number, number, number][] = []
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      // background sky
      let rgb: [number, number, number] = [120 + Math.floor(Math.random() * 30), 170 + Math.floor(Math.random() * 30), 220]

      // ears (triangles)
      const earL = (r >= 2 && r <= 5 && c >= 4 && c <= 6 && (c - r) >= -2)
      const earR = (r >= 2 && r <= 5 && c >= 11 && c <= 13 && (c + r) <= 17)

      // head ellipse
      const dx = (c - 8.5) / 5
      const dy = (r - 7) / 4
      const head = dx * dx + dy * dy < 1

      // eyes
      const eye1 = (r === 6 && c === 6)
      const eye2 = (r === 6 && c === 11)

      // nose
      const nose = (r === 8 && c === 8)

      if (earL || earR || head) rgb = [220 + Math.floor(Math.random() * 20), 150 + Math.floor(Math.random() * 20), 80]
      if (eye1 || eye2) rgb = [20, 30, 80]
      if (nose) rgb = [220, 100, 110]

      pixels.push(rgb)
    }
  }
  return pixels
}

const pixels = makePixels()

const matrixRows = computed(() => {
  const out: [number, number, number][][] = []
  for (let r = 0; r < 8; r++) {
    out.push(pixels.slice(r * cols, r * cols + 6))
  }
  return out
})
</script>

<style scoped>
</style>
