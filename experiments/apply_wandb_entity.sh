#! /bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENTITY_FILE="$ROOT_DIR/wandb_entity.txt"

if [[ ! -f "$ENTITY_FILE" ]]; then
  echo "Missing file: $ENTITY_FILE"
  exit 1
fi

ENTITY="$(grep -v '^[[:space:]]*#' "$ENTITY_FILE" | grep -v '^[[:space:]]*$' | head -n 1 || true)"
if [[ -z "${ENTITY}" ]]; then
  echo "No entity found in $ENTITY_FILE"
  exit 1
fi

for f in "$ROOT_DIR"/experiment_*.sh; do
  ENTITY="$ENTITY" perl -pi -e 'my $e=$ENV{ENTITY}; s/--runner\.wandb_entity=(["\047]).*?\1/--runner.wandb_entity="$e"/g' "$f"
done

echo "Updated --runner.wandb_entity in experiment_*.sh to: $ENTITY"
