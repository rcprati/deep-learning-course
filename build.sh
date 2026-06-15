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
echo "→ Buildando lec01 ..."
node "$TMP/node_modules/@slidev/cli/bin/slidev.mjs" build lec01.md --out dist/lec01 --base /lec01/
echo "→ Buildando lec02 ..."
node "$TMP/node_modules/@slidev/cli/bin/slidev.mjs" build lec02.md --out dist/lec02 --base /lec02/
echo "→ Buildando lec03 ..."
node "$TMP/node_modules/@slidev/cli/bin/slidev.mjs" build lec03.md --out dist/lec03 --base /lec03/
echo "→ Buildando lec01-en ..."
node "$TMP/node_modules/@slidev/cli/bin/slidev.mjs" build lec01-en.md --out dist/lec01-en --base /lec01-en/
echo "→ Buildando lec02-en ..."
node "$TMP/node_modules/@slidev/cli/bin/slidev.mjs" build lec02-en.md --out dist/lec02-en --base /lec02-en/
echo "→ Buildando lec03-en ..."
node "$TMP/node_modules/@slidev/cli/bin/slidev.mjs" build lec03-en.md --out dist/lec03-en --base /lec03-en/

# ── 5. Copiar dist de volta ───────────────────────────────────────────────────
echo "→ Copiando dist para $SRC/dist ..."
rsync -a --delete "$TMP/dist/" "$SRC/dist/"
echo "✓ Pronto! Arquivos em: $SRC/dist/"
