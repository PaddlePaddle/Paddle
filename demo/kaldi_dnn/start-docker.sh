#!/usr/bin/env bash
set -e
work_dir=$(dirname $(dirname $(realpath $0)))

sudo docker run -it --rm \
  -v $work_dir:/paddle/demo \
  -v /work/local/paddle/data:/paddle/data \
  paddledev/paddle:cpu-latest /bin/bash
