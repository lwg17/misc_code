#!/usr/bin/env bash
# slurm_queue_logger.sh (memory-aware)
# Appends a CSV row per job with queue wait time, runtime, resources, and memory usage.
# Usage: source this inside a SLURM batch job. Set LOG_ON_EXIT=1 to auto-log on EXIT.
# Environment:
#   SLURM_QUEUE_LOG : path to CSV (default: $HOME/slurm_wait_times.csv)
#   LOG_ON_EXIT     : if 1, installs EXIT trap to log automatically (default: 0)

SLURM_QUEUE_LOG="${SLURM_QUEUE_LOG:-$HOME/slurm_wait_times.csv}"

# If existing CSV lacks memory columns, write to a sibling file with .mem.csv
choose_logfile() {
  local f="$1"
  if [[ -f "$f" ]]; then
    local first_line
    first_line="$(head -n 1 "$f" | tr -d '\r')"
    if [[ -n "$first_line" && "$first_line" != *"max_rss_kb"* ]]; then
      echo "${f%.csv}.mem.csv"
      return
    fi
  fi
  echo "$f"
}
LOGFILE="$(choose_logfile "$SLURM_QUEUE_LOG")"

# Convert memory tokens to KiB (e.g., 2G -> 2097152)
mem_to_kib() {
  local v="${1:-}"
  [[ -z "$v" ]] && { echo ""; return; }
  v="$(echo -n "$v" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
  v="${v%B}"  # drop trailing B if present (MB -> M)
  if [[ "$v" =~ ^([0-9]+)([KMGTP]?)$ ]]; then
    local n="${BASH_REMATCH[1]}"
    local u="${BASH_REMATCH[2]}"
    case "$u" in
      ""|K) echo "$n" ;;
      M)    echo $(( n * 1024 )) ;;
      G)    echo $(( n * 1024 * 1024 )) ;;
      T)    echo $(( n * 1024 * 1024 * 1024 )) ;;
      P)    echo $(( n * 1024 * 1024 * 1024 * 1024 )) ;;
      *)    echo "" ;;
    esac
  else
    [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo ""
  fi
}

# Parse ReqMem like 4000Mc (per-CPU) or 16Gn (per-node) to total KiB
reqmem_to_total_kib() {
  local req="${1:-}" alloc_cpus="${2:-}" nnodes="${3:-}"
  [[ -z "$req" ]] && { echo ""; return; }
  req="$(echo -n "$req" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
  if [[ "$req" =~ ^([0-9]+)([KMGTP]?)([CN])$ ]]; then
    local n="${BASH_REMATCH[1]}" u="${BASH_REMATCH[2]}" scope="${BASH_REMATCH[3]}"
    local base_kib; base_kib="$(mem_to_kib "${n}${u}")"
    [[ -z "$base_kib" ]] && { echo ""; return; }
    if [[ "$scope" == "C" && "$alloc_cpus" =~ ^[0-9]+$ ]]; then
      echo $(( base_kib * alloc_cpus ))
    elif [[ "$scope" == "N" && "$nnodes" =~ ^[0-9]+$ ]]; then
      echo $(( base_kib * nnodes ))
    else
      echo "$base_kib"
    fi
  else
    mem_to_kib "$req"
  fi
}

csv_escape() {
  local s="$1"
  if [[ "$s" =~ [,\"\\n] ]]; then
    s="${s//\"/\"\"}"
    printf '"%s"' "$s"
  else
    printf '%s' "$s"
  fi
}

slurm_log_queue_time() {
  [[ -z "$SLURM_JOB_ID" ]] && return 0

  if [[ ! -s "$LOGFILE" ]]; then
    printf "job_id,job_name,partition,submit,start,end,wait_s,elapsed_s,elapsed_hms,nnodes,alloc_cpus,ncpus,nodelist,cpu_models,gpu_type,gpu_count,exit_code,max_rss_kb,ave_rss_kb,max_vmsize_kb,req_mem_kb,alloc_mem_kb\n" >> "$LOGFILE"
  fi

  # Gather accounting data
  set +e
  SACCT_ROWS="$(sacct -j "$SLURM_JOB_ID" -X -n -P --format=JobID,JobName,Partition,Submit,Start,End,Elapsed,ElapsedRaw,NNodes,AllocCPUS,NCPUS,NodeList,ExitCode,ReqGRES,AllocTRES,ReqTRES,MaxRSS,AveRSS,MaxVMSize,ReqMem 2>/dev/null)"
  set -e

  # Prefer the .batch row
  PICKED_ROW="$(printf '%s\n' "$SACCT_ROWS" | awk -F'|' -v id="$SLURM_JOB_ID" '$1==id".batch"{print; exit}')"
  [[ -z "$PICKED_ROW" ]] && PICKED_ROW="$(printf '%s\n' "$SACCT_ROWS" | awk -F'|' -v id="$SLURM_JOB_ID" '$1==id{print; exit}')"
  [[ -z "$PICKED_ROW" ]] && PICKED_ROW="$(printf '%s\n' "$SACCT_ROWS" | sed -n '1p')"

  IFS='|' read -r JOBID JOBNAME PART SUBMIT START END ELAPSED ELAPSED_RAW NNODES ALLOC_CPUS NCPUS NODELIST EXITCODE REQ_GRES ALLOC_TRES REQ_TRES MAXRSS_AVG AVERS_AVG MAXVMSIZE REQMEM <<<"$PICKED_ROW"

  # Fallbacks via scontrol for missing fields
  if [[ -z "$SUBMIT" || -z "$START" || -z "$END" || -z "$ELAPSED" || -z "$ELAPSED_RAW" || -z "$NNODES" || -z "$NODELIST" || -z "$ALLOC_CPUS" || -z "$NCPUS" ]]; then
    SJ="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)"
    [[ -z "$SUBMIT"     ]] && SUBMIT="$(awk -F= '/SubmitTime=/{print $2}' <<<"$SJ" | awk '{print $1}')"
    [[ -z "$START"      ]] && START="$(awk  -F= '/StartTime=/{print $2}'  <<<"$SJ" | awk '{print $1}')"
    [[ -z "$END"        ]] && END="$(awk    -F= '/EndTime=/{print $2}'    <<<"$SJ" | awk '{print $1}')"
    [[ -z "$ELAPSED"    ]] && ELAPSED="$(awk -F= '/RunTime=/{print $2}'   <<<"$SJ" | awk '{print $1}')"
    [[ -z "$NNODES"     ]] && NNODES="$(awk  -F= '/NumNodes=/{print $2}'  <<<"$SJ" | awk '{print $1}')"
    [[ -z "$NODELIST"   ]] && NODELIST="$(awk -F= '/NodeList=/{print $2}' <<<"$SJ" | awk '{print $1}')"
    [[ -z "$ALLOC_CPUS" ]] && ALLOC_CPUS="$(awk -F= '/AllocCPUS=/{print $2}' <<<"$SJ" | awk '{print $1}')"
    [[ -z "$NCPUS"      ]] && NCPUS="$(awk -F= '/NumCPUs=/{print $2}' <<<"$SJ" | awk '{print $1}')"
    [[ -z "$ALLOC_CPUS" && -n "$NCPUS" ]] && ALLOC_CPUS="$NCPUS"
    [[ -z "$NCPUS" && -n "$ALLOC_CPUS" ]] && NCPUS="$ALLOC_CPUS"
  fi

  # Normalize node list
  NODELIST="${NODELIST//$'\n'/;}"; NODELIST="${NODELIST//$'\r'/}"
  NODELIST="${NODELIST//(null)/}"; NODELIST="${NODELIST//;;/;}"
  NODELIST="${NODELIST#;}"; NODELIST="${NODELIST%;}"

  # Time math
  to_epoch() { date -d "$1" +%s 2>/dev/null; }
  se="$(to_epoch "$SUBMIT")"; st="$(to_epoch "$START")"; ee="$(to_epoch "$END")"
  WAIT_S=""; [[ -n "$se" && -n "$st" ]] && WAIT_S=$(( st - se ))

  elapsed_to_seconds() {
    local t="$1"
    if [[ "$t" =~ ^([0-9]+)-([0-9]{2}):([0-9]{2}):([0-9]{2})$ ]]; then
      echo $(( ${BASH_REMATCH[1]}*86400 + ${BASH_REMATCH[2]}*3600 + ${BASH_REMATCH[3]}*60 + ${BASH_REMATCH[4]}  ))
    elif [[ "$t" =~ ^([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
      echo $(( ${BASH_REMATCH[1]}*3600 + ${BASH_REMATCH[2]}*60 + ${BASH_REMATCH[3]}  ))
    else
      echo ""
    fi
  }
  ELAPSED_S=""
  [[ -n "$ELAPSED_RAW" && "$ELAPSED_RAW" != "0" ]] && ELAPSED_S="$ELAPSED_RAW"
  [[ -z "$ELAPSED_S" ]] && ELAPSED_S="$(elapsed_to_seconds "$ELAPSED")"
  [[ -z "$ELAPSED_S" && -n "$st" && -n "$ee" ]] && ELAPSED_S=$(( ee - st ))

  # GPUs
  GPU_COUNT="${SLURM_GPUS:-}"
  [[ -z "$GPU_COUNT" && -n "$SLURM_JOB_GRES" ]] && GPU_COUNT="$(sed -n 's/.*gpu:[^:]*:\([0-9]\+\).*/\1/p' <<<"$SLURM_JOB_GRES")"
  if [[ -z "$GPU_COUNT" ]]; then
    GPU_COUNT="$(sed -n 's/.*gpu[:=]\([0-9]\+\).*/\1/p' <<<"$REQ_GRES")"
    [[ -z "$GPU_COUNT" ]] && GPU_COUNT="$(sed -n 's/.*gres\/gpu[:=]\([0-9]\+\).*/\1/p' <<<"$ALLOC_TRES")"
    [[ -z "$GPU_COUNT" ]] && GPU_COUNT="$(sed -n 's/.*gres\/gpu[:=]\([0-9]\+\).*/\1/p' <<<"$REQ_TRES")"
  fi
  GPU_TYPE=""
  if [[ -n "$SLURM_JOB_GRES" ]]; then
    GPU_TYPE="$(sed -n 's/.*gpu:\([^:]\+\).*/\1/p' <<<"$SLURM_JOB_GRES")"
  fi
  if [[ -z "$GPU_TYPE" && -n "$REQ_GRES" ]]; then
    GPU_TYPE="$(sed -n 's/.*gpu:\([^:]\+\).*/\1/p' <<<"$REQ_GRES")"
  fi

  # CPU models (unique)
  cpu_models_collect() {
    local compact="$1"
    local out=""
    local n
    while read -r n; do
      [[ -z "$n" ]] && continue
      local NINFO M
      NINFO="$(scontrol show node -o "$n" 2>/dev/null)"
      M="$(awk -F= '{for(i=1;i<=NF;i++) if($i~/CpuModelName/){split($i,a,"=");print a[2]}}' <<<"$NINFO")"
      [[ -z "$M" ]] && M="$(awk -F= '{for(i=1;i<=NF;i++) if($i~/AvailableFeatures/){split($i,a,"=");print a[2]}}' <<<"$NINFO")"
      [[ -z "$M" ]] && M="$(awk -F= '{for(i=1;i<=NF;i++) if($i~/Architecture/){split($i,a,"=");print a[2]}}' <<<"$NINFO")"
      [[ -z "$M" ]] && M="unknown"
      out+="$M"$'\n'
    done < <(scontrol show hostnames "$compact" 2>/dev/null)
    echo "$out" | awk 'NF' | sort -u | paste -sd';' -
  }
  CPU_MODELS=""; [[ -n "$NODELIST" ]] && CPU_MODELS="$(cpu_models_collect "$NODELIST")"

  # Memory fields (KiB)
  MAX_RSS_KB="$(mem_to_kib "$MAXRSS_AVG")"
  AVE_RSS_KB="$(mem_to_kib "$AVERS_AVG")"
  MAX_VMSIZE_KB="$(mem_to_kib "$MAXVMSIZE")"

  # Requested and allocated memory (total KiB)
  REQ_MEM_KB="$(reqmem_to_total_kib "$REQMEM" "$ALLOC_CPUS" "$NNODES")"
  ALLOC_MEM_KB=""
  if [[ -n "$ALLOC_TRES" ]]; then
    mem_token="$(sed -n 's/.*mem=\([0-9]\+[KMGTP]\?\).*/\1/p' <<<"$ALLOC_TRES")"
    ALLOC_MEM_KB="$(mem_to_kib "$mem_token")"
  fi

  # Output CSV row
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$(csv_escape "${JOBID:-$SLURM_JOB_ID}")" \
    "$(csv_escape "${JOBNAME:-${SLURM_JOB_NAME:-}}")" \
    "$(csv_escape "${PART:-$SLURM_JOB_PARTITION}")" \
    "$(csv_escape "$SUBMIT")" \
    "$(csv_escape "$START")" \
    "$(csv_escape "$END")" \
    "$(csv_escape "$WAIT_S")" \
    "$(csv_escape "$ELAPSED_S")" \
    "$(csv_escape "$ELAPSED")" \
    "$(csv_escape "$NNODES")" \
    "$(csv_escape "$ALLOC_CPUS")" \
    "$(csv_escape "$NCPUS")" \
    "$(csv_escape "$NODELIST")" \
    "$(csv_escape "$CPU_MODELS")" \
    "$(csv_escape "$GPU_TYPE")" \
    "$(csv_escape "$GPU_COUNT")" \
    "$(csv_escape "$EXITCODE")" \
    "$(csv_escape "$MAX_RSS_KB")" \
    "$(csv_escape "$AVE_RSS_KB")" \
    "$(csv_escape "$MAX_VMSIZE_KB")" \
    "$(csv_escape "$REQ_MEM_KB")" \
    "$(csv_escape "$ALLOC_MEM_KB")" \
    >> "$LOGFILE"
}

# Auto-install EXIT trap when sourced with LOG_ON_EXIT=1
if [[ "${LOG_ON_EXIT:-0}" -eq 1 ]]; then
  trap slurm_log_queue_time EXIT
fi
