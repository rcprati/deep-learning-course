#!/usr/bin/env bash
# Workaround: Node 24 ESM + Vite 7 falham em caminhos com espaços.

set -e

SRC="$(cd "$(dirname "$0")" && pwd)"
TMP="/private/tmp/dl-build"
CLI_JS="$TMP/node_modules/@slidev/cli/dist/cli.js"

# ── 1. Copiar fontes ──────────────────────────────────────────────────────────
echo "→ Sincronizando fontes para $TMP ..."
rsync -a --delete \
  --exclude=dist/ \
  --exclude=.git/ \
  --exclude=node_modules/ \
  "$SRC/" "$TMP/"

# ── 2. Garantir node_modules REAL (não symlink) ───────────────────────────────
if [ -L "$TMP/node_modules" ]; then
  echo "→ Removendo symlink antigo de node_modules ..."
  rm "$TMP/node_modules"
fi

# ── 3. Instalar se cli.js não existe ou package.json é mais novo ──────────────
if [ ! -f "$CLI_JS" ] || [ "$SRC/package.json" -nt "$CLI_JS" ]; then
  echo "→ Instalando dependências em $TMP ..."
  cd "$TMP" && npm install
  # Verificar que cli.js foi realmente instalado
  ls "$CLI_JS" > /dev/null || { echo "ERRO: $CLI_JS não encontrado após npm install!"; exit 1; }
  echo "  cli.js OK ✓"
fi

# ── 4. Build ──────────────────────────────────────────────────────────────────
cd "$TMP"

LECTURES=(lec01 lec02 lec03 lec04 lec05 lec06 lec07 lec01-en lec02-en lec03-en lec04-en lec05-en lec06-en lec07-en)

for lec in "${LECTURES[@]}"; do
  if [ ! -f "${lec}.md" ]; then
    echo "⚠️  ${lec}.md não encontrado, pulando."
    continue
  fi
  echo "→ Buildando ${lec} ..."
  node "$TMP/node_modules/@slidev/cli/bin/slidev.mjs" build "${lec}.md" --out "dist/${lec}" --base "/${lec}/"
done

# ── 5. Copiar dist de volta ───────────────────────────────────────────────────
echo "→ Copiando dist para $SRC/dist ..."
rsync -a --delete "$TMP/dist/" "$SRC/dist/"
echo "✓ Pronto! Arquivos em: $SRC/dist/"
