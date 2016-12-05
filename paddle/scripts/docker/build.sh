#!/bin/bash

function abort(){
    echo "An error occurred. Exiting..." 1>&2
    exit 1
}

trap 'abort' 0
set -e

if [ ${WITH_GPU} == 'ON' ]; then
  ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/libcudnn.so
fi

mkdir -p /paddle/build # -p means no error if exists
cd /paddle/build
cmake .. \
      -DWITH_DOC=ON \
      -DWITH_GPU=${WITH_GPU} \
      -DWITH_AVX=${WITH_AVX} \
      -DWITH_SWIG_PY=ON \
      -DCUDNN_ROOT=/usr/ \
      -DWITH_STYLE_CHECK=OFF
make -j `nproc`
make install

trap : 0
