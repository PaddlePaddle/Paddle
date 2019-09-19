#!/bin/bash

set -x

PADDLE_ROOT=$1


# download models
function download() {
    #wget yq01-sys-hic-v100-box-a225-0169.yq01.baidu.com:8123/main_program
    #wget yq01-sys-hic-v100-box-a225-0169.yq01.baidu.com:8123/startup_program
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
