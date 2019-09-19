#!/bin/bash

set -x

PADDLE_ROOT=$1


# download models
function download() {
    echo "downloading is currently not supported"
}

download

# build demo trainer
fluid_install_dir=${PADDLE_ROOT}/build/fluid_install_dir

mkdir -p build
cd build
rm -rf *
cmake .. -DPADDLE_LIB=$fluid_install_dir \
         -DWITH_MKLDNN=OFF \
         -DWITH_MKL=OFF
make

cd ..

build/demo_trainer
