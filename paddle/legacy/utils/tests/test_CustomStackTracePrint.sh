#!/bin/bash
echo "Test Custom Stack Trace print correct result when fail"
./test_CustomStackTracePrint >customStackTraceLog 2>&1
if [ $? -eq 0 ]; then
  exit 1
else
  set -e
  TEXT=""
  for ((i=0; i<=998; i++))
  do
    TEXT="layer_$i, "$TEXT
  done
  TEXT="Forwarding "$TEXT
  grep -q "$TEXT" customStackTraceLog
fi
