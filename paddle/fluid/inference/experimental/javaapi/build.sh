#!/bin/bash
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#

mkdir build && cd build
export library_path=$1
export jni_path=$2
export jni_sub_path=$3
mkldnn_lib=$library_path"/third_party/install/mkldnn/lib"
mklml_lib=$library_path"/third_party/install/mklml/lib"
export paddle_inference_lib=$library_path"/paddle/lib"
export paddle_path=$library_path"/paddle/include"
export LD_LIBRARY_PATH=mkldnn_lib:mklml_lib:paddle_inference_lib
cmake .. && make
#g++ -fPIC -D_REENTRANT -I $jni_path -I $jni_sub_path -I $paddle_path -L $paddle_inference_lib -c com_baidu_paddle_inference_Predictor.cpp com_baidu_paddle_inference_Config.cpp com_baidu_paddle_inference_Tensor.cpp
#g++ -shared -I $paddle_path -L $paddle_inference_lib com_baidu_paddle_inference_Config.o com_baidu_paddle_inference_Predictor.o com_baidu_paddle_inference_Tensor.o -o libpaddle_inference.so -lpaddle_inference_c

cd ../src/main/java/com/baidu/paddle/inference
javac Config.java Predictor.java Tensor.java
cd ../../../../../../../
cp ./native/libpaddle_inference.so libpaddle_inference.so
pwd
jar cvf JavaInference.jar -C src/main/java/ .
