<template>
  <div class="gd-1d">
    <svg viewBox="0 0 360 240" class="w-full">
      <!-- Curve -->
      <path :d="curvePath" fill="none" stroke="#a78bfa" stroke-width="2.5" />

      <!-- Axes -->
      <line x1="20"  y1="210" x2="340" y2="210" stroke="#94a3b8" stroke-width="1" />
      <text x="335" y="225" fill="#94a3b8" font-size="10">w</text>

      <!-- Trail -->
      <g>
        <line
          v-for="(seg, i) in trail" :key="'t' + i"
          :x1="seg.x1" :y1="seg.y1" :x2="seg.x2" :y2="seg.y2"
          stroke="#fbbf24" stroke-width="1.4" stroke-opacity="0.55"
        />
      </g>

      <!-- Ball -->
      <circle :cx="bx" :cy="by" r="7" fill="#fbbf24" stroke="#92400e" stroke-width="1.5" />

      <!-- Iteration label -->
      <text x="180" y="20" text-anchor="middle" fill="#cbd5e1" font-size="11" font-weight="600">
        passo {{ step }} · w ≈ {{ w.toFixed(2) }}
      </text>
    </svg>
    <div class="text-center text-xs opacity-70 mt-1">
      A "bola" rola contra o gradiente até estabilizar no mínimo.
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue'

// g(w) = 0.18 * (w - 1.5)^2 + 0.5
function g(w: number) { return 0.18 * (w - 1.5) * (w - 1.5) + 0.5 }
function gp(w: number) { return 0.36 * (w - 1.5) }

function toPx(w: number) { return 180 + w * 40 }
function toPy(v: number) { return 210 - v * 60 }

const w = ref(-2.0)
const step = ref(0)
const trail = ref<{ x1: number; y1: number; x2: number; y2: number }[]>([])

const bx = computed(() => toPx(w.value))
const by = computed(() => toPy(g(w.value)))

const curvePath = computed(() => {
  const pts: string[] = []
  for (let i = 0; i <= 80; i++) {
    const ww = -3 + (i * 6) / 80
    const x = toPx(ww)
    const y = toPy(g(ww))
    if (y > 20 && y < 230 && x > 20 && x < 340)
      pts.push((pts.length === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1))
  }
  return pts.join(' ')
})

const ALPHA = 0.45
let timer: any

function tick() {
  const x1 = bx.value, y1 = by.value
  const grad = gp(w.value)
  if (Math.abs(grad) < 0.005) {
    // restart
    w.value = -2.0 + (Math.random() - 0.5) * 0.6
    step.value = 0
    trail.value = []
    return
  }
  w.value = w.value - ALPHA * grad
  step.value++
  trail.value.push({ x1, y1, x2: bx.value, y2: by.value })
  if (trail.value.length > 30) trail.value.shift()
}

onMounted(() => { timer = setInterval(tick, 800) })
onUnmounted(() => clearInterval(timer))
</script>
