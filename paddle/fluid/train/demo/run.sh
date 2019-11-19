#!/bin/bash

set -x

PADDLE_ROOT=$1
TURN_ON_MKL=$2 # use MKL or Openblas

# download models
function download() {
    wget -q http://paddle-tar.bj.bcebos.com/train_demo/LR-1-7/main_program
    wget -q http://paddle-tar.bj.bcebos.com/train_demo/LR-1-7/startup_program
}

download

# build demo trainer
fluid_install_dir=${PADDLE_ROOT}/build/fluid_install_dir

mkdir -p build
cd build
rm -rf *
cmake .. -DPADDLE_LIB=$fluid_install_dir \
         -DWITH_MKLDNN=$TURN_ON_MKL \
         -DWITH_MKL=$TURN_ON_MKL
make

cd ..

# run demo trainer
build/demo_trainer
