#!/bin/bash
set -e
ARCH=$(uname -i)
apt-get update
apt-get install -y dh-make
cd ~
mkdir -p ~/dist/gpu
mkdir -p ~/dist/cpu
mkdir -p ~/dist/cpu-noavx
mkdir -p ~/dist/gpu-noavx
cd paddle

# clean build dir and third_party dir cache
rm -rf build third_party
mkdir -p build
cd build
cmake .. -DWITH_GPU=OFF -DWITH_SWIG_PY=ON -DWITH_AVX=ON -DWITH_SWIG_PY=ON -DWITH_STYLE_CHECK=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j `nproc`
# FIXME(typhoonzero): add ARCH gpu noavx flag to CPACK_SYSTEM_NAME. Why -D not affect anything?
cpack -D CPACK_GENERATOR='DEB' ..
mv *.deb ~/dist/cpu

rm -rf *
ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/libcudnn.so
cmake .. -DWITH_GPU=ON -DWITH_SWIG_PY=ON -DWITH_AVX=ON -DCUDNN_ROOT=/usr/ -DWITH_SWIG_PY=ON -DWITH_STYLE_CHECK=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j `nproc`
cpack -D CPACK_GENERATOR='DEB' ..
mv *.deb ~/dist/gpu


rm -rf *
rm -f /usr/lib/libcudnn.so
cmake .. -DWITH_GPU=OFF -DWITH_SWIG_PY=ON -DWITH_AVX=OFF -DWITH_SWIG_PY=ON -DWITH_STYLE_CHECK=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j `nproc`
cpack -D CPACK_GENERATOR='DEB' ..
mv *.deb ~/dist/cpu-noavx

rm -rf *
ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/libcudnn.so
cmake .. -DWITH_GPU=ON -DWITH_SWIG_PY=ON -DWITH_AVX=OFF -DCUDNN_ROOT=/usr/ -DWITH_SWIG_PY=ON -DWITH_STYLE_CHECK=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j `nproc`
cpack -D CPACK_GENERATOR='DEB' ..
mv *.deb ~/dist/gpu-noavx
