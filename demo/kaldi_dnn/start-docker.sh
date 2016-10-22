#!/usr/bin/env bash
set -e

sudo docker run -it --rm \
  -v /work/local/lipeng/github/unisound-ail/paddle/demo:/paddle/demo \
  -v /work/local/paddle/data:/paddle/data \
  paddledev/paddle:cpu-latest /bin/bash
