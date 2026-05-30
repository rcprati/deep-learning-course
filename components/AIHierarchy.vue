<template>
  <div class="ai-hierarchy">
    <svg viewBox="0 0 600 380" class="w-full max-w-2xl mx-auto">
      <!-- AI - largest circle -->
      <ellipse
        cx="300" cy="190" rx="285" ry="170"
        fill="#1e3a8a" fill-opacity="0.15"
        stroke="#3b82f6" stroke-width="2"
      />
      <text x="300" y="40" text-anchor="middle" class="layer-label" fill="#93c5fd">
        Inteligência Artificial
      </text>

      <!-- ML -->
      <ellipse
        cx="300" cy="200" rx="230" ry="135"
        fill="#0e7490" fill-opacity="0.2"
        stroke="#22d3ee" stroke-width="2"
      />
      <text x="300" y="92" text-anchor="middle" class="layer-label" fill="#67e8f9">
        Machine Learning
      </text>

      <!-- DL -->
      <ellipse
        :cx="300" :cy="210" rx="170" ry="100"
        fill="#4f46e5" fill-opacity="0.25"
        stroke="#a78bfa" stroke-width="2"
        :class="{ pulse: showDL }"
      />
      <text x="300" y="142" text-anchor="middle" class="layer-label" fill="#c4b5fd">
        Deep Learning
      </text>

      <!-- Gen AI -->
      <ellipse
        v-if="showGenAI"
        cx="300" cy="225" rx="100" ry="60"
        fill="#db2777" fill-opacity="0.3"
        stroke="#f9a8d4" stroke-width="2"
        class="pulse"
      />
      <text v-if="showGenAI" x="300" y="232" text-anchor="middle" class="genai-label" fill="#fce7f3">
        IA Generativa
      </text>
    </svg>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  showDL?: boolean
  showGenAI?: boolean
}>()
</script>

<style scoped>
.layer-label {
  font-size: 18px;
  font-weight: 600;
  font-family: 'Source Serif Pro', serif;
}
.genai-label {
  font-size: 16px;
  font-weight: 700;
}
.pulse {
  animation: pulse 2.5s ease-in-out infinite;
  transform-origin: center;
  transform-box: fill-box;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}
</style>
