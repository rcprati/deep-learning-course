<template>
  <div class="xy-matrix flex justify-center">
    <table class="xy-table">
      <thead>
        <tr>
          <th class="corner">X \ Y</th>
          <th v-for="m in modalities" :key="'h-' + m.key">
            <div class="text-2xl">{{ m.icon }}</div>
            <div class="text-xs opacity-70 mt-1">{{ m.label }}</div>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in modalities" :key="row.key">
          <th class="row-header">
            <div class="text-2xl">{{ row.icon }}</div>
            <div class="text-xs opacity-70 mt-1">{{ row.label }}</div>
          </th>
          <td v-for="col in modalities" :key="row.key + '-' + col.key" class="cell">
            <div class="cell-content">{{ exampleFor(row.key, col.key) }}</div>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
const modalities = [
  { key: 'text',  icon: '📝',  label: 'Texto'   },
  { key: 'image', icon: '🖼️',  label: 'Imagem'  },
  { key: 'audio', icon: '🔊',  label: 'Áudio'   },
  { key: 'code',  icon: '💻', label: 'Código'  },
]

const examples: Record<string, string> = {
  'text-text':   'tradução, resumo, chat',
  'text-image':  'geração de imagens',
  'text-audio':  'TTS, música',
  'text-code':   'geração de código',
  'image-text':  'legendas, OCR',
  'image-image': 'estilização, edição',
  'image-audio': 'sonificação',
  'image-code':  'sketch → app',
  'audio-text':  'transcrição',
  'audio-image': 'visualização',
  'audio-audio': 'remoção de ruído',
  'audio-code':  '—',
  'code-text':   'documentação',
  'code-image':  'diagramas',
  'code-audio':  'síntese sonora',
  'code-code':   'tradução de linguagens',
}

function exampleFor(x: string, y: string) {
  return examples[`${x}-${y}`] || ''
}
</script>

<style scoped>
.xy-table {
  border-collapse: separate;
  border-spacing: 4px;
  font-size: 12px;
}
.xy-table th, .xy-table td {
  border-radius: 8px;
  padding: 8px;
  vertical-align: middle;
  text-align: center;
}
.xy-table thead th {
  background: rgba(99, 102, 241, 0.15);
  color: #c4b5fd;
  min-width: 110px;
}
.row-header {
  background: rgba(99, 102, 241, 0.15);
  color: #c4b5fd;
}
.cell {
  background: rgba(30, 41, 59, 0.55);
  color: #e2e8f0;
  font-size: 11px;
  min-height: 60px;
  width: 110px;
}
.cell-content {
  line-height: 1.3;
}
.corner {
  background: transparent !important;
  color: #94a3b8;
}
</style>
