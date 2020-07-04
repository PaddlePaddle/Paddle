#!/bin/bash
VERSION=$(nvcc --version | grep release | grep -oEi "release ([0-9]+)\.([0-9])"| sed "s/release //")

if [[ "$version" == "cuda10.1cudnn7" ]];then
  wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT6-cuda10.1-cudnn7.tar.gz --no-check-certificate
  tar -zxf TensorRT6-cuda10.1-cudnn7.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT6-cuda10.1-cudnn7/include/* /usr/include/ && cp -rf /usr/local/TensorRT6-cuda10.1-cudnn7/lib/* /usr/lib/
elif [[ "$version" == "cuda10.0cudnn7" ]];then
  wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT6-cuda10.0-cudnn7.tar.gz --no-check-certificate
  tar -zxf TensorRT6-cuda10.0-cudnn7.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT6-cuda10.0-cudnn7/include/* /usr/include/ && cp -rf /usr/local/TensorRT6-cuda10.0-cudnn7/lib/* /usr/lib/
elif [[ "$version" == "cuda9.0cudnn7" ]];then
  wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT6-cuda9.0-cudnn7.tar.gz --no-check-certificate
  tar -zxf TensorRT6-cuda9.0-cudnn7.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT6-cuda9.0-cudnn7/include/* /usr/include/ && cp -rf /usr/local/TensorRT6-cuda9.0-cudnn7/lib/* /usr/lib/
fi
