<template>
  <div class="nn-viz flex justify-center">
    <svg
      :viewBox="viewBox"
      :width="width"
      :height="height"
      preserveAspectRatio="xMidYMid meet"
      class="nn-svg"
      :style="{ maxWidth: '100%', maxHeight: '52vh', height: 'auto' }"
      :class="{ animated: animate }"
    >
      <!-- Connections drawn FIRST so neurons sit on top -->
      <g class="connections">
        <line
          v-for="(conn, i) in connections"
          :key="'c' + i"
          :x1="conn.x1" :y1="conn.y1"
          :x2="conn.x2" :y2="conn.y2"
          stroke="#cbd5e1"
          stroke-width="1.4"
          stroke-opacity="0.6"
        />
      </g>

      <!-- Layer labels -->
      <g v-if="labels && labels.length">
        <text
          v-for="(lbl, i) in labels"
          :key="'lbl' + i"
          :x="layerX(i)"
          y="14"
          text-anchor="middle"
          fill="#cbd5e1"
          font-size="11"
          font-weight="600"
        >{{ lbl }}</text>
      </g>

      <!-- Neurons -->
      <g class="neurons">
        <g v-for="(n, i) in neurons" :key="'n' + i">
          <circle
            :cx="n.x" :cy="n.y" :r="radius"
            :fill="n.fill"
            :stroke="n.stroke"
            stroke-width="2"
            class="neuron"
          />
          <text
            :x="n.x" :y="n.y + 4"
            text-anchor="middle"
            fill="#f8fafc"
            font-size="13"
            font-weight="700"
          >+</text>
        </g>
      </g>

      <!-- Ellipsis dots for layers that exceed the cap -->
      <g v-for="ellipsis in ellipses" :key="'el' + ellipsis.layer">
        <circle :cx="ellipsis.x" :cy="ellipsis.y - 6" r="1.6" fill="#94a3b8" />
        <circle :cx="ellipsis.x" :cy="ellipsis.y"     r="1.6" fill="#94a3b8" />
        <circle :cx="ellipsis.x" :cy="ellipsis.y + 6" r="1.6" fill="#94a3b8" />
      </g>

      <!-- Param count badge -->
      <g v-if="showCount">
        <rect :x="cx - 75" :y="height - 28" width="150" height="22" rx="11"
              fill="#4f46e5" fill-opacity="0.25" stroke="#a78bfa" stroke-width="1.2" />
        <text :x="cx" :y="height - 13" text-anchor="middle" fill="#e0e7ff" font-size="11" font-weight="600">
          {{ paramCount }} parâmetros
        </text>
      </g>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  layers: number[]
  labels?: string[]
  showWeights?: boolean
  showCount?: boolean
  animate?: boolean
}>(), {
  showWeights: false,
  showCount: false,
  animate: false,
})

// Compact layout — fits comfortably in slide height
const radius = 14
const layerSpacing = 110
const neuronSpacing = 38
const padX = 60
const padY = 28

// Cap on visible neurons per layer. If sz > MAX, render TOP + ellipsis + BOTTOM
const MAX_VISIBLE = 7
const TOP = 4
const BOTTOM = 3

function visibleCount(sz: number) {
  return sz > MAX_VISIBLE ? MAX_VISIBLE : sz
}

const width = computed(() => padX * 2 + (props.layers.length - 1) * layerSpacing)
const height = computed(() => {
  const maxVisible = Math.max(...props.layers.map(visibleCount))
  return padY * 2 + (maxVisible - 1) * neuronSpacing + 28
})

const cx = computed(() => width.value / 2)
const viewBox = computed(() => `0 0 ${width.value} ${height.value}`)

function layerX(i: number) {
  return padX + i * layerSpacing
}

function neuronY(visN: number, idx: number) {
  const totalH = (visN - 1) * neuronSpacing
  const startY = (height.value - totalH) / 2 + 8
  return startY + idx * neuronSpacing
}

// Map (layer, displayedIdx) -> y position. For collapsed layers,
// displayedIdx is in [0, MAX_VISIBLE-1] and there's a "gap" between TOP and BOTTOM.
function neuronYDisplayed(sz: number, displayedIdx: number) {
  const visN = visibleCount(sz)
  return neuronY(visN, displayedIdx)
}

const neurons = computed(() => {
  const out: { x: number; y: number; fill: string; stroke: string; layer: number; idx: number }[] = []
  props.layers.forEach((sz, li) => {
    let fill = 'rgba(14,116,144,0.45)'
    let stroke = '#22d3ee'
    if (li === props.layers.length - 1) { fill = 'rgba(6,95,70,0.45)'; stroke = '#10b981' }
    else if (li > 0) { fill = 'rgba(245,158,11,0.35)'; stroke = '#f59e0b' }

    if (sz <= MAX_VISIBLE) {
      for (let i = 0; i < sz; i++) {
        out.push({ x: layerX(li), y: neuronYDisplayed(sz, i), fill, stroke, layer: li, idx: i })
      }
    } else {
      // Top neurons
      for (let i = 0; i < TOP; i++) {
        out.push({ x: layerX(li), y: neuronYDisplayed(sz, i), fill, stroke, layer: li, idx: i })
      }
      // Bottom neurons (offset displayed index past the ellipsis slot)
      for (let i = 0; i < BOTTOM; i++) {
        const displayed = TOP + 1 + i  // skip 1 slot for ellipsis
        if (displayed >= MAX_VISIBLE) break
        out.push({ x: layerX(li), y: neuronYDisplayed(sz, displayed), fill, stroke, layer: li, idx: sz - BOTTOM + i })
      }
    }
  })
  return out
})

// Position of "..." dots for layers that overflow
const ellipses = computed(() => {
  const out: { layer: number; x: number; y: number }[] = []
  props.layers.forEach((sz, li) => {
    if (sz > MAX_VISIBLE) {
      // Place between the TOP block and BOTTOM block
      const yTop = neuronYDisplayed(sz, TOP - 1)
      const yBot = neuronYDisplayed(sz, TOP + 1)
      out.push({ layer: li, x: layerX(li), y: (yTop + yBot) / 2 })
    }
  })
  return out
})

// Connections are drawn between every visible pair (shows the dense pattern visually)
const connections = computed(() => {
  const out: { x1: number; y1: number; x2: number; y2: number }[] = []
  for (let li = 0; li < props.layers.length - 1; li++) {
    const sz1 = props.layers[li]
    const sz2 = props.layers[li + 1]
    const vis1 = visibleCount(sz1)
    const vis2 = visibleCount(sz2)
    for (let i = 0; i < vis1; i++) {
      // Skip the ellipsis slot if present
      if (sz1 > MAX_VISIBLE && i === TOP) continue
      for (let j = 0; j < vis2; j++) {
        if (sz2 > MAX_VISIBLE && j === TOP) continue
        out.push({
          x1: layerX(li) + radius,
          y1: neuronYDisplayed(sz1, i),
          x2: layerX(li + 1) - radius,
          y2: neuronYDisplayed(sz2, j),
        })
      }
    }
  }
  return out
})

const paramCount = computed(() => {
  let n = 0
  for (let li = 0; li < props.layers.length - 1; li++) {
    n += props.layers[li] * props.layers[li + 1]
    n += props.layers[li + 1]
  }
  return n
})
</script>

<style scoped>
.nn-svg {
  display: block;
}
.animated .neuron {
  animation: pulse 2.5s ease-in-out infinite;
}
.animated .connections line {
  stroke-dasharray: 4 5;
  animation: dash 4s linear infinite;
}
@keyframes pulse {
  0%, 100% { filter: drop-shadow(0 0 1px currentColor); }
  50%      { filter: drop-shadow(0 0 6px currentColor); }
}
@keyframes dash {
  to { stroke-dashoffset: -200; }
}
</style>
