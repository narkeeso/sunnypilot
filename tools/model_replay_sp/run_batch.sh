#!/usr/bin/env bash
# Replay each stop window in JOBS_FILE through every model, two windows at a time (runs on device, offroad).
# usage: run_batch.sh JOBS_FILE MODELS(comma list of catalog short names) OUT_DIR
# JOBS_FILE lines: TAG ROUTE SEGS START_S   (window = START_S..START_S+35 s from the first road frame of SEGS)
# Get windows from list_stops.py: stop at t >= 30 s -> SEGS=s, START_S=t-30; else SEGS=s-1,s, START_S=t+30.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_FILE="$1"; MODELS="$2"; OUT="$3"; mkdir -p "$OUT"
(setsid bash "$DIR/onroad_guard.sh" > /dev/null 2>&1 < /dev/null &)  # subshell: not counted in jobs below; stops everything if onroad

run_one() {  # tag route segs start
  cd /tmp
  REPLAY_TAG=$1 REPLAY_START_S=$4 REPLAY_MAX_S=35 REPLAY_COPYFRAMES=1 PATH=/usr/local/venv/bin:$PATH \
    PYTHONPATH=$DIR/replayshim:/data/openpilot setsid python -u "$DIR/model_replay_sp.py" "$2" "$3" "$MODELS" "$OUT/$1" \
    > "$OUT/$1.log" 2>&1 < /dev/null &
  local pid=$!
  wait $pid
  pkill -g $pid 2>/dev/null  # replayed modeld left in that session
  rm -f /dev/shm/replay_$1_*.bin
  echo "$(date +%T) done $1: $(grep -c 'frames in' "$OUT/$1.log") models ok, $(grep -c ': FAILED' "$OUT/$1.log") failed, $(grep -c Traceback "$OUT/$1.log") tracebacks"
}

while read -r tag route segs start; do
  [ -z "$tag" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 5; done
  run_one "$tag" "$route" "$segs" "$start" &
done < "$JOBS_FILE"
wait
echo ALL DONE
