<template>
  <div class="cv-tasks grid grid-cols-2 md:grid-cols-4 gap-3 max-w-5xl mx-auto">
    <div v-for="t in tasks" :key="t.title" class="text-center">
      <div class="font-bold text-indigo-300 text-sm mb-2">{{ t.title }}</div>
      <svg viewBox="0 0 120 90" class="w-full">
        <!-- Background "scene" -->
        <rect x="2" y="2" width="116" height="86" fill="#1e293b" stroke="#475569" stroke-width="1" rx="3" />

        <!-- Three sheep silhouettes (positions vary by task) -->
        <g v-if="t.kind === 'classify'">
          <circle cx="60" cy="50" r="22" fill="#94a3b8" />
          <circle cx="60" cy="32" r="11" fill="#94a3b8" />
          <text x="60" y="80" text-anchor="middle" fill="#cbd5e1" font-size="9" font-weight="700">"ovelha"</text>
        </g>

        <g v-if="t.kind === 'localize'">
          <circle cx="60" cy="50" r="22" fill="#94a3b8" />
          <circle cx="60" cy="32" r="11" fill="#94a3b8" />
          <rect x="35" y="20" width="50" height="50" fill="none" stroke="#fbbf24" stroke-width="2" />
          <text x="60" y="14" text-anchor="middle" fill="#fde68a" font-size="8">ovelha</text>
        </g>

        <g v-if="t.kind === 'detect'">
          <circle cx="35" cy="55" r="13" fill="#94a3b8" /><circle cx="35" cy="44" r="7" fill="#94a3b8" />
          <circle cx="65" cy="60" r="14" fill="#94a3b8" /><circle cx="65" cy="48" r="7" fill="#94a3b8" />
          <circle cx="92" cy="55" r="11" fill="#94a3b8" /><circle cx="92" cy="46" r="6" fill="#94a3b8" />
          <rect x="20" y="36" width="30" height="32" fill="none" stroke="#fbbf24" stroke-width="1.5" />
          <rect x="49" y="40" width="32" height="34" fill="none" stroke="#fbbf24" stroke-width="1.5" />
          <rect x="80" y="38" width="24" height="28" fill="none" stroke="#fbbf24" stroke-width="1.5" />
        </g>

        <g v-if="t.kind === 'semantic'">
          <circle cx="35" cy="55" r="13" fill="#a78bfa" /><circle cx="35" cy="44" r="7" fill="#a78bfa" />
          <circle cx="65" cy="60" r="14" fill="#a78bfa" /><circle cx="65" cy="48" r="7" fill="#a78bfa" />
          <circle cx="92" cy="55" r="11" fill="#a78bfa" /><circle cx="92" cy="46" r="6" fill="#a78bfa" />
          <text x="60" y="84" text-anchor="middle" fill="#c4b5fd" font-size="8">cada pixel: classe</text>
        </g>
      </svg>
      <div class="text-xs opacity-75 mt-1">{{ t.hint }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
const tasks = [
  { title: 'Classificação',         kind: 'classify', hint: 'qual é a classe?' },
  { title: 'Localização',           kind: 'localize', hint: 'classe + caixa' },
  { title: 'Detecção de objetos',   kind: 'detect',   hint: 'várias classes/caixas' },
  { title: 'Segmentação semântica', kind: 'semantic', hint: 'classe por pixel' },
]
</script>
