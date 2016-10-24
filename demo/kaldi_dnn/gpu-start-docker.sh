#!/usr/bin/env bash
set -e
work_dir=$(dirname $(dirname $(realpath $0)))

sudo docker run -it --rm \
  --privileged \
  -v $work_dir:/paddle/demo \
  -v /home/core/paddle/data:/paddle/data \
  -v /var/lib/nvidia:/usr/local/nvidia/lib64 \
  harbor.ail.unisound.com/liuqs_public/paddle:gpu-latest /bin/bash
