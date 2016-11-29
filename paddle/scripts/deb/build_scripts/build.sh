#!/bin/bash
set -e
apt-get update
apt-get install -y dh-make
cd ~
mkdir -p ~/dist/gpu
mkdir -p ~/dist/cpu
mkdir -p ~/dist/cpu-noavx
mkdir -p ~/dist/gpu-noavx
cd paddle
mkdir build
cd build
cmake .. -DWITH_GPU=OFF -DWITH_SWIG_PY=ON -DWITH_AVX=ON
make -j `nproc`
cpack -D CPACK_GENERATOR='DEB' ..
mv *.deb ~/dist/cpu

rm -rf *
cmake .. -DWITH_GPU=ON -DWITH_SWIG_PY=ON -DWITH_AVX=ON -DCUDNN_ROOT=/usr/
make -j `nproc`
cpack -D CPACK_GENERATOR='DEB' ..
mv *.deb ~/dist/gpu


rm -rf *
cmake .. -DWITH_GPU=OFF -DWITH_SWIG_PY=ON -DWITH_AVX=OFF
make -j `nproc`
cpack -D CPACK_GENERATOR='DEB' ..
mv *.deb ~/dist/cpu-noavx

rm -rf *
cmake .. -DWITH_GPU=ON -DWITH_SWIG_PY=ON -DWITH_AVX=OFF -DCUDNN_ROOT=/usr/
make -j `nproc`
cpack -D CPACK_GENERATOR='DEB' ..
mv *.deb ~/dist/gpu-noavx
