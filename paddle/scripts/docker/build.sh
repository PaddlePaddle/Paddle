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
      -DWITH_STYLE_CHECK=OFF \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j `nproc`
make install

# Install woboq_codebrowser.
git clone https://github.com/woboq/woboq_codebrowser /woboq
cd /woboq
cmake -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-3.8 \
      -DCMAKE_BUILD_TYPE=Release \
      .
make

export WOBOQ_OUT=/usr/share/nginx/html/paddle
export BUILD_DIR=/paddle/build
mkdir -p $WOBOQ_OUT
cp -rv /woboq/data $WOBOQ_OUT/../data
/woboq/generator/codebrowser_generator \
    -b /paddle/build \
    -a \
    -o $WOBOQ_OUT \
    -p paddle:/paddle
/woboq/indexgenerator/codebrowser_indexgenerator $WOBOQ_OUT

trap : 0
