#!/bin/bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

docker_name=$1

function ref_whl(){
  ref_gpu=gpu-cuda${ref_CUDA_MAJOR}-cudnn${CUDNN_MAJOR}

  if [[ ${ref_CUDA_MAJOR} == "11.0" ]];then
      ref_version=.post110
  elif [[ ${ref_CUDA_MAJOR} == "11.2" ]];then
      ref_version=.post112
  elif [[ ${ref_CUDA_MAJOR} == "10.1" ]];then
      ref_version=.post101
  elif [[ ${ref_CUDA_MAJOR} == "10.2" ]];then
      ref_version=""
  fi

  ref_web="https://paddle-wheel.bj.bcebos.com/${PADDLE_BRANCH}/linux/linux-${ref_gpu}-mkl-gcc8.2-avx"
  ref_paddle37_whl=paddlepaddle_gpu-${PADDLE_VERSION/-/}${ref_version}-cp37-cp37m-linux_x86_64.whl
}


function install_whl(){
  dockerfile_line=`wc -l Dockerfile.tmp|awk '{print $1}'`
  sed -i "${dockerfile_line}i RUN wget -q ${ref_web}/${ref_paddle37_whl} && pip3.7 install ${ref_paddle37_whl} && rm -f ${ref_paddle37_whl}" Dockerfile.tmp
}

function set_cuda_env(){
  if [[ ${ref_CUDA_MAJOR} == "11.2" ]];then
      sed -i 's#<set_cuda_env>#RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so \&\& \\ \
    ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.11 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so \&\& \\ \
    ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusparse.so.11 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusparse.so \&\& \\ \
    ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcublas.so.11 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcublas.so \
ENV LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/cuda-11.2/targets/x86_64-linux/lib/:${LD_LIBRARY_PATH} #g' Dockerfile.tmp
  elif [[ ${ref_CUDA_MAJOR} == "10.2" ]];then
      sed -i 's#<set_cuda_env>#RUN rm -f /usr/lib/x86_64-linux-gnu/libcublas.so \&\& \\ \
    ln -s /usr/lib/x86_64-linux-gnu/libcublas.so.10 /usr/lib/x86_64-linux-gnu/libcublas.so \
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/lib/x86_64-linux-gnu/libcudnn.so \&\& \\ \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so \&\& \\ \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusolver.so \
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.2/targets/x86_64-linux/lib/:${LD_LIBRARY_PATH} #g' Dockerfile.tmp
  fi
}

# wrong code, bash file
function install_nccl2(){
  if [[ ${ref_CUDA_MAJOR} == "11.2" ]];then
      sed -i 's#<install_nccl2>#COPY tools/dockerfile/performance_docker/install_nccl2.sh / \
RUN bash /install_nccl2.sh \
RUN rm -rf /install_nccl2.sh #g' Dockerfile.tmp
  else
      sed -i 's#<install_nccl2># #g' Dockerfile.tmp
  fi
}


function install_dali(){
  if [[ ${ref_CUDA_MAJOR} == "11.2" ]];then
      sed -i 's#<install_dali>#RUN python3.7 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110 #g' Dockerfile.tmp
  elif [[ ${ref_CUDA_MAJOR} == "10.2" ]];then
      sed -i 's#<install_dali>#RUN python3.7 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100 #g' Dockerfile.tmp
  fi
}


function make_dockerfile(){
  sed "s#<baseimg>#${docker_name} #g" tools/dockerfile/performance_docker/Dockerfile.perf >Dockerfile.tmp
}


function main(){
  ref_whl
  make_dockerfile
  install_dali
  install_nccl2
  set_cuda_env
  install_whl
}

main $@
