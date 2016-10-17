#!/bin/bash
set -e
cd ~
mkdir -p ~/dist/gpu
mkdir -p ~/dist/cpu
mkdir -p ~/dist/cpu-noavx
mkdir -p ~/dist/gpu-noavx
yum -y install cmake3
git clone https://github.com/baidu/Paddle.git paddle
cd paddle
mkdir build
cd build
cmake3 .. -DWITH_GPU=OFF -DWITH_SWIG_PY=ON -DWITH_AVX=ON
make -j `nproc`
cpack3 -G RPM
mv *.rpm ~/dist/cpu

rm -rf *
cmake3 .. -DWITH_GPU=ON -DWITH_SWIG_PY=ON -DWITH_AVX=ON
make -j `nproc`
cpack3 -G RPM
mv *.rpm ~/dist/gpu


rm -rf *
cmake3 .. -DWITH_GPU=OFF -DWITH_SWIG_PY=ON -DWITH_AVX=OFF
make -j `nproc`
cpack3 -G RPM
mv *.rpm ~/dist/cpu-noavx

rm -rf *
cmake3 .. -DWITH_GPU=ON -DWITH_SWIG_PY=ON -DWITH_AVX=OFF
make -j `nproc`
cpack3 -G RPM
mv *.rpm ~/dist/gpu-noavx


