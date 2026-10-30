#!/usr/bin/env bash
# Move (or delete) everything in ./results except the files/patterns below.
#   bash clean_results.sh           dry run: only lists what would happen
#   bash clean_results.sh --move    moves files into ./results_to_delete
#   bash clean_results.sh --delete  permanently deletes files
set -euo pipefail

DIR="./results"
TRASH="./results_to_delete"

KEEP=(
  'Temp_Quench_T1_20__T20_g0.5_N140000_N240000_tth0.01_L4e-05_J0_D-4e+06_VP4_a_s100N_grid512_id002.hdf5'
  'Temp_Quench_T1_20__T20_g0.5_N140000_N240000_tth0.01_L4e-05_J0_D-4e+06_VP4_a_s100N_grid512_id003.hdf5'
  'Temp_Quench_T1_10__T10_g0.5_N140000_N240000_tth0.01_L4e-05_J0_D-4e+06_VP4_a_s100N_grid512_id000.hdf5'
  'Temp_Quench_T1_20__T0_g0.1_N120000_N220000_tth0.01_L2e-05_J0_D-1e+07_VP13_a_s100N_grid256_id00*.hdf5'
  'Temp_Quench_T1_20__T20_g0.5_N160000_N260000_tth0.01_L1e-05_J0_D-4e+06_VP2.5_a_s100N_grid128_id000.hdf5'
  'Temp_Quench_T1_20__T20_g0.5_N160000_N260000_tth0.01_L4e-05_J0_D-4e+06_VP4_a_s100N_grid512_id002.hdf5'
  'Temp_Quench_T1_20__T20_g0.5_N160000_N260000_tth0.01_L4e-05_J0_D-4e+06_VP4_a_s100N_grid512_id003.hdf5'
  'Temp_Quench_T1_20__T20_g0.5_N180000_N280000_tth0.01_L2e-05_J0_D-7e+06_VP12_a_s100N_grid256_id001.hdf5'
  'Temp_Quench_T1_20__T20_g0.5_N180000_N280000_tth0.01_L2e-05_J0_D-7e+06_VP12_a_s100N_grid256_id002.hdf5'
  'Temp_Quench_T1_20__T20_g0.5_N180000_N280000_tth0.01_L1.2e-05_J0_D-6e+06_VP10_a_s100N_grid128_id000.hdf5'
  'Temp_Quench_T1_20__T20_g0.5_N180000_N280000_tth0.01_L1.6e-05_J0_D-6e+06_VP10_a_s100N_grid128_id001.hdf5'
  'Temp_Quench_T1_5__T5_g0.1_N140000_N240000_tth0.0102_L4e-05_J0_D-1e+07_VP10_a_s100N_grid512_id004.hdf5'
  'Temp_Quench_T1_5__T5_g0.1_N140000_N240000_tth0.0102_L4e-05_J0_D-1e+07_VP10_a_s100N_grid512_id006.hdf5'
  'Temp_Quench_T1_30__T*_g0.01_N120000_N220000_tth0.05_L4e-05_J0_D-1e+07_VP0_a_s100N_grid512_id00*.hdf5'
  'Temp_Quench_T1_30__T0_g0.01_N120000_N220000_tth0.05_L4e-05_J0_D-1e+07_VP0_a_s100N_grid512_id004.hdf5'
  'Temp_Quench_T1_200__T1_g0.1_N140000_N240000_tth0.01_L2e-05_J0_D-1e+07_VP0_a_s100N_grid256_id001.hdf5'
  'Temp_Quench_T1_*__T*_g0.01_N120000_N220000_tth0.05_L4e-05_J0_D-1e+07_VP0_a_s100N_grid512_id000.hdf5'
)

MODE="dry"
case "${1:-}" in
  --move)   MODE="move"; mkdir -p "$TRASH" ;;
  --delete) MODE="delete" ;;
  "")       ;;
  *) echo "Unknown option: $1 (use --move or --delete)"; exit 1 ;;
esac

kept=0; removed=0
while IFS= read -r -d '' f; do
  name=$(basename "$f")
  keep=0
  for p in "${KEEP[@]}"; do
    # $p unquoted on purpose so * acts as a wildcard
    if [[ "$name" == $p ]]; then keep=1; break; fi
  done
  if (( keep )); then
    echo "KEEP    $name"; kept=$((kept+1))
  else
    case "$MODE" in
      move)   mv -n -- "$f" "$TRASH/"; echo "MOVED   $name" ;;
      delete) rm -- "$f";              echo "DELETED $name" ;;
      *)                               echo "REMOVE  $name" ;;
    esac
    removed=$((removed+1))
  fi
done < <(find "$DIR" -maxdepth 1 -type f -print0)

echo "----"
case "$MODE" in
  move)   echo "kept: $kept   moved to $TRASH: $removed" ;;
  delete) echo "kept: $kept   deleted: $removed" ;;
  *)      echo "kept: $kept   would remove: $removed"
          echo "Dry run only. Re-run with --move (safe) or --delete." ;;
esac
