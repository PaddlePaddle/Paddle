#!/bin/bash
set -e
mkdir -p /src/
cd /src
wget https://github.com/NVIDIA/nccl/archive/v1.3.4-1.tar.gz -O nccl.tar.gz
tar xzf nccl.tar.gz
mv nccl-* nccl
rm nccl.tar.gz
cd nccl
make -j 4
make install PREFIX=/usr/local/cuda/targets/x86_64-linux
cd /
rm -rf /src/nccl
