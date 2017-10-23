#!/bin/bash
set -e
mkdir -p /src/
cd /src
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j 4
make install
