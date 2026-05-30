<template>
  <div class="loss-landscape">
    <svg viewBox="0 0 540 280" class="w-full max-w-4xl mx-auto">
      <!-- Pseudo-3D contour landscape with bumps -->
      <g>
        <path v-for="(p, i) in contours" :key="i"
              :d="p.d" fill="none" :stroke="p.color" stroke-width="1.3" stroke-opacity="0.7" />
      </g>

      <!-- Several minima markers -->
      <g>
        <circle v-for="(m, i) in minima" :key="'m' + i"
                :cx="m.x" :cy="m.y" :r="m.r" :fill="m.color"
                :stroke="m.stroke" stroke-width="1.5" />
      </g>

      <!-- Path of SGD (zigzag descent) -->
      <polyline :points="pathPoints" fill="none" stroke="#fbbf24" stroke-width="2" stroke-opacity="0.85" />
      <circle v-for="(pt, i) in pathDots" :key="'pd' + i"
              :cx="pt.x" :cy="pt.y" r="3" fill="#fbbf24" />

      <!-- Title -->
      <text x="270" y="20" text-anchor="middle" fill="#cbd5e1" font-size="12" font-weight="600">
        Loss landscape de uma rede profunda (esquemático)
      </text>
    </svg>
    <div class="text-center text-xs opacity-70 mt-1">
      Não convexa, com vários "vales bons" — o ruído do SGD ajuda a explorar.
    </div>
  </div>
</template>

<script setup lang="ts">
// Procedurally generate concentric blob-like contours via parametric noise
const cx = 270, cy = 145

function blobPath(scale: number, seed: number) {
  const N = 64
  const pts: string[] = []
  for (let i = 0; i <= N; i++) {
    const t = (i / N) * Math.PI * 2
    const noise =
      Math.sin(t * 3 + seed) * 0.15 +
      Math.cos(t * 5 + seed * 2) * 0.1 +
      Math.sin(t * 2 + seed * 0.7) * 0.05
    const r = scale * (1 + noise)
    const x = cx + r * 1.5 * Math.cos(t)
    const y = cy + r * Math.sin(t)
    pts.push((i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1))
  }
  return pts.join(' ') + ' Z'
}

const contours = [
  { d: blobPath(120, 0.3), color: '#1e3a8a' },
  { d: blobPath(95,  0.5), color: '#3730a3' },
  { d: blobPath(75,  0.7), color: '#4f46e5' },
  { d: blobPath(55,  0.9), color: '#6366f1' },
  { d: blobPath(35,  1.1), color: '#a78bfa' },
  { d: blobPath(20,  1.3), color: '#c4b5fd' },
]

const minima = [
  { x: cx,       y: cy,       r: 5, color: '#10b981', stroke: '#065f46' }, // global
  { x: cx - 80,  y: cy + 40,  r: 4, color: '#f59e0b', stroke: '#92400e' }, // local
  { x: cx + 90,  y: cy - 30,  r: 4, color: '#f59e0b', stroke: '#92400e' }, // local
  { x: cx + 50,  y: cy + 70,  r: 3, color: '#22d3ee', stroke: '#155e75' }, // saddle
]

// SGD path zigzagging towards global min
const pathDots = [
  { x: 60,  y: 60  },
  { x: 90,  y: 95  },
  { x: 130, y: 80  },
  { x: 165, y: 115 },
  { x: 200, y: 125 },
  { x: 230, y: 145 },
  { x: cx,  y: cy  },
]
const pathPoints = pathDots.map(p => `${p.x},${p.y}`).join(' ')
</script>
