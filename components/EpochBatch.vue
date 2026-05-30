<template>
  <div class="epoch-batch">
    <svg viewBox="0 0 400 220" class="w-full">
      <!-- Training set frame -->
      <rect x="20" y="40" width="360" height="40" rx="6" fill="none" stroke="#94a3b8" stroke-width="1.5" />
      <text x="200" y="32" text-anchor="middle" fill="#cbd5e1" font-size="11" font-weight="600">conjunto de treinamento</text>

      <!-- Batches -->
      <g v-for="(b, i) in batches" :key="i">
        <rect :x="25 + i * 51" :y="44" width="46" height="32" rx="4"
              :fill="b.active ? 'rgba(251,191,36,0.5)' : 'rgba(34,211,238,0.2)'"
              :stroke="b.active ? '#fbbf24' : '#22d3ee'" stroke-width="1.4" />
        <text :x="48 + i * 51" :y="64" text-anchor="middle"
              :fill="b.active ? '#fde68a' : '#67e8f9'" font-size="10">B{{ i + 1 }}</text>
      </g>

      <!-- Arrow + label -->
      <line x1="200" y1="100" x2="200" y2="135" stroke="#cbd5e1" stroke-width="1.5" marker-end="url(#a)" />
      <text x="200" y="155" text-anchor="middle" fill="#cbd5e1" font-size="11">um <tspan font-weight="700" fill="#fbbf24">batch</tspan> por vez</text>

      <!-- Update rule -->
      <text x="200" y="195" text-anchor="middle" fill="#a78bfa" font-size="13" font-style="italic">w ← w − α · ∇L</text>

      <defs>
        <marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#cbd5e1" />
        </marker>
      </defs>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue'

const N = 7
const active = ref(0)

const batches = computed(() =>
  Array.from({ length: N }).map((_, i) => ({ active: i === active.value }))
)

let timer: any
onMounted(() => { timer = setInterval(() => { active.value = (active.value + 1) % N }, 700) })
onUnmounted(() => clearInterval(timer))
</script>
