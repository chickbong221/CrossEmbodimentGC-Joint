#!/bin/bash
# Fetch a file (or a tar of several, when given several remote paths relative to a directory) from Hessian through hx.sh as
# base64 between markers, with an md5 check and retries.   Usage: fetch.sh REMOTE_DIR LOCAL_FILE NAME [NAME...]
HX="/c/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo-AppData-Roaming-Claude-scratch-workspaces-01d12a43-350f-44d6-be61-f854bdd47ef5-8404b0a0-4694-46bf-a0a5-722e1c8e0322-scratch-2026-09-29-af8bb3/4fd1fedc-8187-45f5-a299-25da41c82518/scratchpad/hx.sh"
DIR=$1; OUT=$2; shift 2
for i in 1 2 3 4; do
  raw=$(timeout 280 bash "$HX" "cd $DIR && tar -cf /tmp/fetch.tar $* && echo MD5=\$(md5sum /tmp/fetch.tar | cut -c1-32) && echo BEGIN64 && base64 -w0 /tmp/fetch.tar && echo && echo END64" < /dev/null 2>/dev/null)
  md5=$(echo "$raw" | grep -o '^MD5=[0-9a-f]*' | head -1 | cut -c5-)
  echo "$raw" | sed -n '/^BEGIN64$/,/^END64$/p' | sed '1d;$d' | tr -d '\n' | base64 -d > "$OUT" 2>/dev/null
  [ -n "$md5" ] && [ "$(md5sum "$OUT" | cut -c1-32)" = "$md5" ] && { echo "fetched $OUT ($(stat -c %s "$OUT") bytes)"; exit 0; }
  echo "retry $i"; sleep 5
done
echo "FETCH FAILED"; exit 1
