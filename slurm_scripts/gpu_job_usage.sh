#!/usr/bin/env bash
# gpu_job_usage.sh
# Aggregate Slurm accounting into a single per-job CSV row and append/update $JOB_USAGE_CSV.
# Usage:
#   gpu_job_usage.sh <jobid>[,<jobid2>,...] [--wait <seconds>] [--retries <n>]
#
# Optional env:
#   JOB_USAGE_CSV      : path to CSV (default: $HOME/job_usage.csv)
#   OUTPUT_SIZE_GIB    : storage footprint in GiB to record for this run (optional)
#
# Notes:
# - Aggregates across job steps (batch/extern) to capture MaxRSS and I/O that Slurm
#   often records only at the step level.
# - Parses GPU allocation from AllocTRES (supports gres/gpu=4 or gres/gpu:4).
# - Computes gpu_hours = (elapsed_sec / 3600) * gpus.
# - If CSV exists, the row for the same job_id is replaced.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <jobid>[,<jobid2>,...] [--wait <seconds>] [--retries <n>]" >&2
  exit 1
fi

IDS="$1"; shift || true
WAIT_SECS=2
RETRIES=10

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait) WAIT_SECS="${2:-2}"; shift 2 ;;
    --retries) RETRIES="${2:-10}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

CSV="${JOB_USAGE_CSV:-$HOME/job_usage.csv}"
TMP="$(mktemp)"

# Header for our CSV
HEADER="job_id,name,partition,state,elapsed_sec,gpus,gpu_hours,maxrss_gib,maxvmsize_gib,disk_read_gib,disk_write_gib,storage_gib"

# Function: run sacct and aggregate
run_sacct() {
  sacct -j "$IDS" -n -P \
    --format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,MaxDiskRead,MaxDiskWrite \
  | awk -F'|' -v storage_gib="${OUTPUT_SIZE_GIB:-}" '
  function toGiB(x,  n,u){ if(x==""||x=="0"||x=="0K")return 0; n=substr(x,1,length(x)-1)+0; u=toupper(substr(x,length(x),1));
    return (u=="K")?n/1048576:(u=="M")?n/1024:(u=="G")?n:(u=="T")?n*1024:0 }
  function base(j){ sub(/\..*$/,"",j); return j }
  function parse_gpus(t, a){
    # supports gres/gpu=2, gres/gpu:2, gpu=2, gpu:2
    if (match(t,/(gres\/gpu[:=]|gpu[:=])([0-9]+)/,a)) return a[2]+0
    return 0
  }
  {
    b=base($1)
    # prefer non-batch/extern name
    if(!(b in name) || ($2!="batch" && $2!="extern")) { name[b]=$2 }
    if(!(b in part) && $3!="") { part[b]=$3 }
    state[b]=$4
    if(($5+0) > elap[b]) elap[b]=$5+0
    g=parse_gpus($6); if(g>gpus[b]) gpus[b]=g
    mr=toGiB($7); if(mr>maxrss[b]) maxrss[b]=mr
    mv=toGiB($8); if(mv>maxvm[b]) maxvm[b]=mv
    rd[b]+=toGiB($11)
    wr[b]+=toGiB($12)
  }
  END{
    for (b in elap){
      gh=(elap[b]/3600.0)*gpus[b]
      printf "%s,%s,%s,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n",
        b,name[b],part[b],state[b],elap[b],gpus[b],gh,maxrss[b],maxvm[b],rd[b],wr[b],
        (storage_gib=="" ? "" : storage_gib)
    }
  }'
}

# Retry loop because sacct can lag a bit after job finish
i=0
out=""
while [[ $i -lt $RETRIES ]]; do
  if out="$(run_sacct)"; then
    if [[ -n "$out" ]]; then
      break
    fi
  fi
  sleep "$WAIT_SECS"
  i=$((i+1))
done

if [[ -z "$out" ]]; then
  echo "No sacct data retrieved; try increasing --wait/--retries." >&2
  exit 3
fi

# Ensure CSV with header exists
if [[ ! -f "$CSV" ]]; then
  echo "$HEADER" > "$CSV"
fi

# Update-or-append: remove existing rows with same job_id(s), then add new
ID_REGEX="$(echo "$IDS" | sed 's/,/|/g')"

awk -F, -v OFS=, -v re="^("ID_REGEX")," '
  NR==1 { print; next }
  $1 ~ re { next }
  { print }
' "$CSV" > "$TMP"

printf "%s\n" "$out" >> "$TMP"

{ head -n1 "$TMP"; tail -n +2 "$TMP" | sort -t, -k1,1n; } > "$CSV"

rm -f "$TMP"
echo "Updated: $CSV"
