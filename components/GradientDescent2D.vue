<template>
  <div class="gd-2d">
    <svg viewBox="0 0 480 320" class="w-full max-w-3xl mx-auto">
      <!-- Contour ellipses (concentric, since g = w1^2 + w2^2) -->
      <g stroke-opacity="0.6" fill="none">
        <ellipse v-for="r in radii" :key="r" :cx="cx" :cy="cy" :rx="r * 1.2" :ry="r * 0.9"
                 :stroke="color(r)" stroke-width="1.2" />
      </g>

      <!-- Origin -->
      <circle :cx="cx" :cy="cy" r="4" fill="#10b981" />
      <text :x="cx + 8" :y="cy + 14" fill="#6ee7b7" font-size="11">mínimo</text>

      <!-- Path -->
      <g>
        <line
          v-for="(seg, i) in segs" :key="'s' + i"
          :x1="seg.x1" :y1="seg.y1" :x2="seg.x2" :y2="seg.y2"
          stroke="#fbbf24" stroke-width="2" stroke-opacity="0.8"
        />
        <circle
          v-for="(p, i) in pts" :key="'p' + i"
          :cx="p.x" :cy="p.y" r="3.5" fill="#fbbf24" stroke="#92400e" stroke-width="1"
        />
      </g>

      <!-- Gradient arrow at current point -->
      <g v-if="current">
        <line
          :x1="current.x" :y1="current.y"
          :x2="current.x + arrow.dx" :y2="current.y + arrow.dy"
          stroke="#ef4444" stroke-width="2" marker-end="url(#arr)"
        />
        <text :x="current.x + arrow.dx + 6" :y="current.y + arrow.dy" fill="#fca5a5" font-size="10">
          −α∇g
        </text>
      </g>

      <!-- Axes labels -->
      <text x="20"  y="cy" fill="#94a3b8" font-size="11">w₂</text>
      <text :y="cy + 4" x="20" fill="#94a3b8" font-size="11">w₂</text>
      <text :x="cx" y="312" text-anchor="middle" fill="#94a3b8" font-size="11">w₁</text>

      <defs>
        <marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#ef4444" />
        </marker>
      </defs>
    </svg>
    <div class="text-center text-xs opacity-70 mt-1">
      A trajetória cruza as curvas de nível em direção ao centro.
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue'

const cx = 240
const cy = 160

const radii = [25, 55, 90, 125, 160]
function color(r: number) {
  const t = (r - 25) / 135
  const hue = 230 - t * 90
  return `hsl(${hue} 70% 60% / 0.6)`
}

// State: position in (w1, w2) space mapped to pixels.
// 1 unit = 30px. minimum at origin.
const w1 = ref(-3.5)
const w2 = ref(2.6)

const pts = ref<{ x: number; y: number }[]>([])
const segs = ref<{ x1: number; y1: number; x2: number; y2: number }[]>([])

function toX(v: number) { return cx + v * 30 }
function toY(v: number) { return cy - v * 30 }

const current = computed(() => ({ x: toX(w1.value), y: toY(w2.value) }))
const arrow = computed(() => {
  // gradient = (2w1, 2w2). step = -α * grad. show as arrow scaled.
  const a = 0.25
  const dx = -a * 2 * w1.value * 30
  const dy = a * 2 * w2.value * 30
  return { dx, dy }
})

const ALPHA = 0.18
let timer: any
function tick() {
  const px = toX(w1.value), py = toY(w2.value)
  if (Math.abs(w1.value) + Math.abs(w2.value) < 0.05) {
    // restart
    w1.value = (Math.random() - 0.5) * 7
    w2.value = (Math.random() - 0.5) * 5
    pts.value = []
    segs.value = []
    return
  }
  w1.value = w1.value - ALPHA * 2 * w1.value
  w2.value = w2.value - ALPHA * 2 * w2.value
  const nx = toX(w1.value), ny = toY(w2.value)
  pts.value.push({ x: nx, y: ny })
  segs.value.push({ x1: px, y1: py, x2: nx, y2: ny })
  if (pts.value.length > 25) {
    pts.value.shift()
    segs.value.shift()
  }
}

onMounted(() => { pts.value = [{ x: toX(w1.value), y: toY(w2.value) }]; timer = setInterval(tick, 800) })
onUnmounted(() => clearInterval(timer))
</script>
