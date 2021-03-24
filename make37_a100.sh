#!/bin/bash
(($#!=1)) && echo "usage $0 all/fast" && exit -1
export http_proxy="http://172.19.57.45:3128"
export https_proxy="http://172.19.57.45:3128"

CWD=$PWD/..
#export PATH=/opt/compiler/gcc-8.2/bin:$CWD/py37/bin/:/ssd1/luyuxiang/software/cuda-11.0/bin:$PATH
#export LD_LIBRARY_PATH=/ssd1/luyuxiang/software/cuda-11.0/lib64:/ssd1/luyuxiang/software/cuda/lib64:/ssd1/luyuxiang/software/cuda/include:/ssd1/luyuxiang/software/nccl_2.7.8-1+cuda11.0_x86_64/lib:$LD_LIBRARY_PATH

if [ $1 != "fast" ];then
    rm -rf build/ && mkdir build && cd build
    cmake .. \
	-DWITH_MKL=OFF \
	-DWITH_FLUID_ONLY=ON \
	-DWITH_GLOO=ON \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DWITH_DISTRIBUTE=ON \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m/  \
        -DPYTHON_LIBRARY=/usr/lib/python3.7/\
        -DPY_VERSION=3.7 \
        -DWITH_GPU=ON \
        -DWITH_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \

else 
    cd build
fi

make -j 40
#python3.6 -m pip install -U python/dist/paddlepaddle_gpu-0.0.0-cp36-cp36m-linux_x86_64.whl

