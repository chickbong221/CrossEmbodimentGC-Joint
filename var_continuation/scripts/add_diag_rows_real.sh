#!/bin/bash
# Variance-diagnostics rows of the four VARb branches: every saved checkpoint, plus the source checkpoint for the branches
# whose parameters it fits (S0, S1, S3; S2's frozen head + offsets need the branch's own parameter tree).
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/letuan_r2dreamergraph_2/morse
SRC=$R/repo/experiments/runs/morse/V7-CH/V7-CH-s0/models/at_75000
B=$R/repo/experiments/runs/morse/VARb
mkdir -p /tmp/var_runs/analysis
list() { echo "$B/VARb-$1/models/at_76000,$B/VARb-$1/models/at_80000,$B/VARb-$1/models/at_85000,$B/VARb-$1/models/at_90000,$B/VARb-$1/models/at_95000"; }
bash $R/analysis/var/add_diag_row.sh var-diag-S0 original "$SRC,$(list S0)" 4
bash $R/analysis/var/add_diag_row.sh var-diag-S1 detach "$SRC,$(list S1)" 4
bash $R/analysis/var/add_diag_row.sh var-diag-S2 private "$(list S2)" 4
bash $R/analysis/var/add_diag_row.sh var-diag-S3 protected "$SRC,$(list S3)" 4
printf 'var-diag-S0\nvar-diag-S1\nvar-diag-S2\nvar-diag-S3\n' > $R/tools/queue/queue_h_vardiag.txt
