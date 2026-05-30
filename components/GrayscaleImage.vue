<template>
  <div class="grayscale flex justify-center items-center gap-6">
    <!-- The "image" - smiley face -->
    <div>
      <div class="text-xs opacity-70 mb-2 text-center">imagem</div>
      <div class="grid" :style="gridStyle">
        <div
          v-for="(p, i) in pixels"
          :key="i"
          :style="{
            width: cell + 'px',
            height: cell + 'px',
            background: `rgb(${p}, ${p}, ${p})`,
          }"
        />
      </div>
    </div>

    <div class="text-3xl opacity-50">→</div>

    <!-- The matrix of values -->
    <div>
      <div class="text-xs opacity-70 mb-2 text-center">matriz de intensidades (0–255)</div>
      <div class="font-mono text-[10px] leading-snug bg-slate-900/60 p-2 rounded border border-slate-700">
        <div v-for="(row, ri) in rows" :key="ri" class="whitespace-nowrap flex gap-2">
          <span
            v-for="(p, ci) in row"
            :key="ci"
            class="inline-block tabular-nums w-7 text-right"
            :style="{ color: `rgb(${Math.max(p, 60)}, ${Math.max(p, 60)}, ${Math.max(p, 60)})` }"
          >{{ p }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const cols = 12
const cell = 18

// Procedural smiley face on a light background
function makePixels() {
  const out: number[] = []
  for (let r = 0; r < cols; r++) {
    for (let c = 0; c < cols; c++) {
      let v = 235 // background
      const dx = c - 5.5
      const dy = r - 5.5
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dist < 5)        v = 200
      if (dist < 4.6)      v = 250 // face
      if (r === 4 && (c === 3 || c === 8)) v = 30 // eyes
      if (r === 7 && c >= 3 && c <= 8 && Math.abs(c - 5.5) > 1) v = 30 // smile
      out.push(v)
    }
  }
  return out
}

const pixels = makePixels()

const gridStyle = computed(() => ({
  gridTemplateColumns: `repeat(${cols}, ${cell}px)`,
  gap: '1px',
}))

const rows = computed(() => {
  // show top 6 rows × 6 cols
  const out: number[][] = []
  for (let r = 0; r < 6; r++) out.push(pixels.slice(r * cols, r * cols + 6))
  return out
})
</script>
