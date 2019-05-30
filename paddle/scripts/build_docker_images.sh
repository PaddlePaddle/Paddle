#!/bin/bash
set -xe

REPO="${REPO:-paddlepaddle}"

sed 's/<baseimg>/9.0-cudnn7-devel-ubuntu16.04/g' Dockerfile.tmp | 
sed 's/<tensor_rt_version>/5.1_ga_cuda9_cudnnv7.5/g' |
sed 's/<nccl_version>/2.4.7-1+cuda9.0/g' > Dockerfile.cuda9.0-cudnn7
# docker build -t ${REPO}/paddle:cuda9.0-cudnn7-devel-ubuntu16.04 -f Dockerfile.cuda9.0-cudnn7 .

sed 's/<baseimg>/10.0-cudnn7-devel-ubuntu16.04/g' Dockerfile.tmp | 
sed 's/<tensor_rt_version>/5.1_ga_cuda10_cudnnv7.5/g' |
sed 's/<nccl_version>/2.4.7-1+cuda10.0/g' > Dockerfile.cuda10.0-cudnn7
# docker build -t ${REPO}/paddle:cuda10.0-cudnn7-devel-ubuntu16.04 -f Dockerfile.cuda10.0-cudnn7 .
