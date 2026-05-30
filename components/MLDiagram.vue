<template>
  <div class="ml-diagram">
    <div class="flex items-center justify-center gap-4 max-w-4xl mx-auto px-4">
      <!-- Input box -->
      <div class="box input">
        <div class="title">Entrada</div>
        <div class="sub"><em>x₁, x₂, …, xₙ</em></div>
      </div>

      <!-- Arrow -->
      <div class="arrow">→</div>

      <!-- Model box -->
      <div class="box model">
        <div class="title-lg">Modelo de</div>
        <div class="title-lg">Machine Learning</div>
        <div class="sub italic">aprende de exemplos</div>
      </div>

      <!-- Arrow -->
      <div class="arrow">→</div>

      <!-- Output box -->
      <div class="box output">
        <div class="title">Saída</div>
        <div class="sub"><em>ŷ</em></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.box {
  border-radius: 12px;
  padding: 1.1rem 1.4rem;
  border-width: 2px;
  border-style: solid;
  text-align: center;
  min-width: 9rem;
}
.input {
  background: rgba(14, 116, 144, 0.25);
  border-color: #22d3ee;
  color: #cffafe;
}
.model {
  background: rgba(79, 70, 229, 0.30);
  border-color: #a78bfa;
  color: #e0e7ff;
  padding: 1.4rem 1.6rem;
  animation: glow 3s ease-in-out infinite;
}
.output {
  background: rgba(6, 95, 70, 0.30);
  border-color: #10b981;
  color: #d1fae5;
}
.title {
  font-size: 1.05rem;
  font-weight: 600;
  margin-bottom: 0.25rem;
}
.title-lg {
  font-size: 1.15rem;
  font-weight: 700;
  line-height: 1.2;
}
.sub {
  font-size: 0.9rem;
  opacity: 0.9;
  margin-top: 0.4rem;
}
.italic { font-style: italic; }

.arrow {
  font-size: 1.8rem;
  color: #94a3b8;
  font-weight: 300;
}

@keyframes glow {
  0%, 100% { box-shadow: 0 0 6px rgba(167, 139, 250, 0.30); }
  50%      { box-shadow: 0 0 16px rgba(167, 139, 250, 0.65); }
}
</style>
