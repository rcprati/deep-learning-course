#!/usr/bin/env bash
# Inicia o servidor de desenvolvimento do Slidev para uma aula.
# Uso: bash dev.sh lec02.md

set -e

SRC="$(cd "$(dirname "$0")" && pwd)"
TMP="/private/tmp/dl-build"
CLI_JS="$TMP/node_modules/@slidev/cli/dist/cli.js"
FILE="${1:-lec02.md}"

# Sincronizar fontes (sem node_modules)
rsync -a \
  --exclude=dist/ \
  --exclude=.git/ \
  --exclude=node_modules/ \
  "$SRC/" "$TMP/"

# Instalar deps se necessário
if [ ! -f "$CLI_JS" ] || [ "$SRC/package.json" -nt "$CLI_JS" ]; then
  echo "→ Instalando dependências em $TMP ..."
  cd "$TMP" && npm install
fi

echo "→ Iniciando servidor de desenvolvimento para $FILE ..."
echo "   Acesse: http://localhost:3030"
cd "$TMP"
node "$TMP/node_modules/@slidev/cli/bin/slidev.mjs" dev "$FILE"
