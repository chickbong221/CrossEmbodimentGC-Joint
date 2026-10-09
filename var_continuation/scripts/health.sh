#!/bin/bash
# One-shot health line of the VAR runs: update counts (one 'Step:' line per update), markers, tracebacks. Usage: health.sh [wait_seconds]
HX="/c/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo-AppData-Roaming-Claude-scratch-workspaces-01d12a43-350f-44d6-be61-f854bdd47ef5-8404b0a0-4694-46bf-a0a5-722e1c8e0322-scratch-2026-09-29-af8bb3/4fd1fedc-8187-45f5-a299-25da41c82518/scratchpad/hx.sh"
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
[ -n "$1" ] && sleep $1
timeout 120 bash "$HX" "cd $R/hq; echo \$(date -u +%H:%M:%S); for id in VARb-S0 VARb-S1 VARb-S2 VARb-S3; do m=run; [ -e state/done/\$id ] && m=DONE; [ -e state/failed/\$id ] && m=FAILED; echo \$id \$m \$((75000+\$(grep -c 'Step:' runlogs/\$id.log))) tb=\$(grep -c Traceback runlogs/\$id.log); done" < /dev/null 2>/dev/null | tr '\n' ' '; echo
