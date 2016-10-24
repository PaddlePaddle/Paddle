#!/usr/bin/env bash
set -e

sudo docker run -it --rm \
  --privileged \
  -v /home/core/lipeng/unisound-ail/paddle/demo:/paddle/demo \
  -v /home/core/paddle/data:/paddle/data \
  -v /var/lib/nvidia:/usr/local/nvidia/lib64 \
  harbor.ail.unisound.com/liuqs_public/paddle:gpu-latest /bin/bash
