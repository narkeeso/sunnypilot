# Stops all replay work the moment the car goes onroad, so it never competes with the live driving stack.
# Replays run as session leaders (setsid), so killing each session takes its replayed modeld with it.
while pgrep -f "run_batch.sh" >/dev/null; do
  if [ "$(cat /data/params/d/IsOnroad 2>/dev/null)" = "1" ]; then
    pkill -f "run_batch.sh"
    for sid in $(pgrep -f "model_replay_sp.py"); do pkill -s "$sid"; done
    for p in $(pgrep -f modeld_tinygrad); do [ "$(ps -o ppid= -p "$p" | tr -d ' ')" = "1" ] && kill "$p"; done  # orphans only; live modeld's parent is manager
    echo "$(date +%T) onroad: stopped replay batch" >> /data/tmp/model_replay_onroad_guard.log
    exit 0
  fi
  sleep 2
done
