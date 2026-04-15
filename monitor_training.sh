#!/bin/bash
# Monitor NAAC training progress

LOG_FILE="logs/naac_training_full.log"

echo "NAAC Training Monitor"
echo "===================="
echo ""

# Check if process is running
PID=$(ps aux | grep "train_naac.py" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "Training process not running!"
    exit 1
fi

echo "Process PID: $PID"
echo "Log file: $LOG_FILE"
echo ""

# Show latest progress
echo "Latest progress:"
tail -5 "$LOG_FILE" | grep "Episode"

echo ""
echo "Training statistics:"
grep "Episode" "$LOG_FILE" | tail -10

echo ""
echo "Estimated completion:"
EPISODES=$(grep "Episode" "$LOG_FILE" | tail -1 | awk '{print $2}')
if [ ! -z "$EPISODES" ]; then
    REMAINING=$((50000 - EPISODES))
    EPS_PER_SEC=$(grep "Episode" "$LOG_FILE" | tail -1 | awk '{print $NF}' | sed 's/eps\/s//')
    if [ ! -z "$EPS_PER_SEC" ]; then
        SECONDS_LEFT=$(echo "$REMAINING / $EPS_PER_SEC" | bc)
        MINUTES_LEFT=$(echo "$SECONDS_LEFT / 60" | bc)
        echo "  Episodes completed: $EPISODES / 50000"
        echo "  Episodes remaining: $REMAINING"
        echo "  Speed: $EPS_PER_SEC eps/s"
        echo "  Estimated time remaining: ~$MINUTES_LEFT minutes"
    fi
fi
