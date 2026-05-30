<template>
  <div class="timeline">
    <svg viewBox="0 0 460 370" class="w-full">
      <!-- Vertical axis line -->
      <line x1="80" y1="20" x2="80" y2="350" stroke="#64748b" stroke-width="2" />

      <!-- Events -->
      <g v-for="(ev, i) in events" :key="i">
        <circle
          :cx="80" :cy="40 + i * 50"
          r="7"
          :fill="ev.color"
          stroke="#0f172a" stroke-width="2"
          class="dot"
        />
        <text
          :x="100" :y="40 + i * 50 + 5"
          fill="#e2e8f0"
          class="event-text"
        >
          <tspan font-weight="700">{{ ev.year }}</tspan>
          <tspan dx="10">{{ ev.label }}</tspan>
        </text>
      </g>
    </svg>
  </div>
</template>

<script setup lang="ts">
const events = [
  { year: '1956', label: 'Dartmouth · nasce a IA',         color: '#3b82f6' },
  { year: '1958', label: 'Perceptron (Rosenblatt)',         color: '#22d3ee' },
  { year: '1986', label: 'Retropropagação popularizada',     color: '#a78bfa' },
  { year: '1998', label: 'LeNet (CNN)',                      color: '#a78bfa' },
  { year: '2012', label: 'AlexNet · ImageNet',               color: '#f59e0b' },
  { year: '2017', label: 'Transformer',                       color: '#f59e0b' },
  { year: '2022', label: 'ChatGPT · IA generativa',           color: '#ec4899' },
]
</script>

<style scoped>
.event-text {
  font-size: 13px;
  font-family: 'Inter', sans-serif;
}
.dot {
  filter: drop-shadow(0 0 4px currentColor);
}
</style>
