#!/bin/bash

# 監視するPID
MONITORED_PID=$1


echo "Monitoring PID: $MONITORED_PID"

# 無限ループでPIDを監視
while true; do
  # プロセスが存在するか確認
  if ps -p $MONITORED_PID > /dev/null; then
    echo "Process $MONITORED_PID is running."
  else
    echo "Process $MONITORED_PID is not running. Starting new process..."
    # 新しいプロセスを起動
    bash for_topk_ab.sh 3
    exit 0
  fi
  # 5秒待機
  sleep 5
done
