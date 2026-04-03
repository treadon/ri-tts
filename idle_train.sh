#!/bin/bash
# ri-tts idle training: runs when your Mac is idle, survives crashes/reboots
# Usage:
#   ./idle_train.sh start    # start training (or resume)
#   ./idle_train.sh stop     # graceful stop (saves checkpoint)
#   ./idle_train.sh status   # check if running
#   ./idle_train.sh log      # tail the training log

set -e
cd /Users/ritesh/Dev/model-training/ri-tts

PIDFILE=".train.pid"
LOGFILE="training.log"

case "${1:-start}" in
  start)
    # Kill existing if running
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      echo "Training already running (PID $(cat $PIDFILE)). Use './idle_train.sh stop' first."
      exit 1
    fi

    echo "Starting training (2cb, Qwen3-0.6B, MPS)..."
    echo "  Log: $LOGFILE"
    echo "  Stop: ./idle_train.sh stop"
    echo "  Status: ./idle_train.sh status"

    source venv/bin/activate
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

    # Run in background, auto-resume from checkpoint
    nohup python train.py \
      --codebooks 2 \
      --epochs 10 \
      --batch-size 1 \
      >> "$LOGFILE" 2>&1 &

    echo $! > "$PIDFILE"
    echo "Started (PID $!)"
    ;;

  stop)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      PID=$(cat $PIDFILE)
      echo "Stopping training (PID $PID)..."
      # SIGINT triggers graceful shutdown in HF Trainer (saves checkpoint)
      kill -INT "$PID"
      # Wait up to 60s for graceful shutdown
      for i in $(seq 1 60); do
        if ! kill -0 "$PID" 2>/dev/null; then
          echo "Stopped gracefully."
          rm -f "$PIDFILE"
          exit 0
        fi
        sleep 1
      done
      echo "Force killing..."
      kill -9 "$PID" 2>/dev/null
      rm -f "$PIDFILE"
      echo "Killed."
    else
      echo "Training not running."
      rm -f "$PIDFILE" 2>/dev/null
    fi
    ;;

  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      PID=$(cat $PIDFILE)
      echo "Training running (PID $PID)"
      # Show last loss
      grep "'loss'" "$LOGFILE" 2>/dev/null | tail -1
      # Show progress
      grep "s/it" "$LOGFILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+'
    else
      echo "Training not running."
      rm -f "$PIDFILE" 2>/dev/null
      # Show where we left off
      ls checkpoints/qwen3-0.6b-2cb/ 2>/dev/null | tail -3
    fi
    ;;

  log)
    tail -30 "$LOGFILE"
    ;;

  watch)
    # Watchdog: restart if crashed. Run this in a separate terminal or cron.
    echo "Watching training... (Ctrl+C to stop watching)"
    while true; do
      if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
        sleep 60
      else
        echo "$(date): Training not running, restarting..."
        rm -f "$PIDFILE" 2>/dev/null
        source venv/bin/activate
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        nohup python train.py \
          --codebooks 2 \
          --epochs 10 \
          --batch-size 1 \
          >> "$LOGFILE" 2>&1 &
        echo $! > "$PIDFILE"
        echo "$(date): Restarted (PID $!)"
        sleep 120  # Wait 2 min before checking again
      fi
    done
    ;;

  *)
    echo "Usage: ./idle_train.sh {start|stop|status|log|watch}"
    ;;
esac
