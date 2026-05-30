<template>
  <div class="fashion-mnist grid grid-cols-5 gap-2 max-w-md mx-auto">
    <div v-for="(item, i) in items" :key="i" class="text-center">
      <div class="w-full aspect-square rounded bg-slate-900/60 border border-slate-700 flex items-center justify-center text-3xl">
        {{ item.icon }}
      </div>
      <div class="text-[10px] opacity-70 mt-1">{{ item.label }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
const items = [
  { icon: '👕', label: 'camiseta' },
  { icon: '👖', label: 'calça' },
  { icon: '🧥', label: 'pulôver' },
  { icon: '👗', label: 'vestido' },
  { icon: '🧣', label: 'casaco' },
  { icon: '👡', label: 'sandália' },
  { icon: '🎽', label: 'camisa' },
  { icon: '👟', label: 'tênis' },
  { icon: '👜', label: 'bolsa' },
  { icon: '🥾', label: 'bota' },
]
</script>
