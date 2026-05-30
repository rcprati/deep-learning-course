<template>
  <div class="keras-stack flex flex-col items-center gap-2">
    <div class="layer keras">Keras (API amigável)</div>
    <div class="connector">▼</div>
    <div class="layer tf">TensorFlow (núcleo)</div>
    <div class="connector">▼</div>
    <div class="grid grid-cols-3 gap-2 w-full">
      <div class="layer hw">CPU</div>
      <div class="layer hw">GPU</div>
      <div class="layer hw">TPU</div>
    </div>
  </div>
</template>

<style scoped>
.layer {
  width: 100%;
  text-align: center;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  border-width: 2px;
  border-style: solid;
  font-weight: 600;
  font-size: 0.85rem;
}
.keras {
  background: rgba(245, 158, 11, 0.25);
  border-color: #f59e0b;
  color: #fef3c7;
}
.tf {
  background: rgba(79, 70, 229, 0.30);
  border-color: #a78bfa;
  color: #e0e7ff;
}
.hw {
  background: rgba(14, 116, 144, 0.30);
  border-color: #22d3ee;
  color: #cffafe;
  font-size: 0.8rem;
  padding: 0.4rem 0.5rem;
}
.connector {
  color: #94a3b8;
  font-size: 0.7rem;
}
</style>
