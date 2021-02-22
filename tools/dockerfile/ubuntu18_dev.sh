#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
  if [[ ${WITH_GPU} == "ON" ]]; then
      ref_gpu=gpu-cuda${ref_CUDA_MAJOR}-cudnn${CUDNN_MAJOR}
      install_gpu="_gpu"
  else
      ref_gpu="cpu"
      install_gpu=""
  fi
  
  if [[ ${WITH_MKL} == "ON" ]]; then
      ref_mkl=mkl
  else
      ref_mkl=openblas
  fi

  if [[ ${WITH_GPU} != "ON" ]]; then
    ref_gcc = ""
  elif [[ ${gcc_version} == "8.2.0" ]];then
    ref_gcc=_gcc8.2
  fi

  if [[ ${ref_CUDA_MAJOR} == "11.0" ]];then
      ref_version=.post110
  elif [[ ${ref_CUDA_MAJOR} == "10" ]];then
      ref_version=.post100
  elif [[ ${ref_CUDA_MAJOR} == "10.1" ]];then
      ref_version=.post101
  elif [[ ${ref_CUDA_MAJOR} == "10.2" ]];then
      ref_version=""
  elif [[ ${ref_CUDA_MAJOR} == "9" ]];then
      ref_version=.post90
  fi
  
  ref_web="https://paddle-wheel.bj.bcebos.com/${PADDLE_BRANCH}-${ref_gpu}-${ref_mkl}${ref_gcc}"
  
  if [[ ${PADDLE_VERSION} == "0.0.0" && ${WITH_GPU} == "ON" ]]; then
    ref_paddle_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
    ref_paddle3_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp35-cp35m-linux_x86_64.whl
    ref_paddle36_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp36-cp36m-linux_x86_64.whl
    ref_paddle37_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp37-cp37m-linux_x86_64.whl
    ref_paddle38_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp38-cp38-linux_x86_64.whl
  else
    ref_paddle_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
    ref_paddle3_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp35-cp35m-linux_x86_64.whl
    ref_paddle36_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp36-cp36m-linux_x86_64.whl
    ref_paddle37_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp37-cp37m-linux_x86_64.whl
    ref_paddle38_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp38-cp38-linux_x86_64.whl
  fi
  
  if [[ ${PADDLE_VERSION} != "0.0.0" && ${WITH_GPU} == "ON" ]]; then
    ref_paddle_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}${ref_version}-cp27-cp27mu-linux_x86_64.whl
    ref_paddle3_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}${ref_version}-cp35-cp35m-linux_x86_64.whl
    ref_paddle36_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}${ref_version}-cp36-cp36m-linux_x86_64.whl
    ref_paddle37_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}${ref_version}-cp37-cp37m-linux_x86_64.whl
    ref_paddle38_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}${ref_version}-cp38-cp38-linux_x86_64.whl
  else
    ref_paddle_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
    ref_paddle3_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp35-cp35m-linux_x86_64.whl
    ref_paddle36_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp36-cp36m-linux_x86_64.whl
    ref_paddle37_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp37-cp37m-linux_x86_64.whl
    ref_paddle38_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp38-cp38-linux_x86_64.whl
  fi
}


function install_whl(){
  dockerfile_line=`wc -l Dockerfile.tmp|awk '{print $1}'`
  sed -i "${dockerfile_line}i RUN wget -q ${ref_web}/${ref_paddle_whl} && pip install ${ref_paddle_whl} && rm -f  ${ref_paddle_whl}" Dockerfile.tmp
  sed -i "${dockerfile_line}i RUN wget -q ${ref_web}/${ref_paddle3_whl} && pip3.5 install ${ref_paddle3_whl} && rm  -f ${ref_paddle3_whl}" Dockerfile.tmp
  sed -i "${dockerfile_line}i RUN wget -q ${ref_web}/${ref_paddle36_whl} && pip3.6 install ${ref_paddle36_whl} && rm -f ${ref_paddle36_whl}" Dockerfile.tmp
  sed -i "${dockerfile_line}i RUN wget -q ${ref_web}/${ref_paddle37_whl} && pip3.7 install ${ref_paddle37_whl} && rm -f ${ref_paddle37_whl}" Dockerfile.tmp
  sed -i "${dockerfile_line}i RUN wget -q ${ref_web}/${ref_paddle38_whl} && pip3.8 install ${ref_paddle38_whl} && rm -f ${ref_paddle38_whl}" Dockerfile.tmp
}


function install_gcc(){
  if [ "${gcc_version}" == "8.2.0" ];then
    sed -i 's#<install_gcc>#WORKDIR /usr/bin \
      COPY tools/dockerfile/build_scripts /build_scripts \
      RUN bash /build_scripts/install_trt.sh \
      RUN bash /build_scripts/install_gcc.sh gcc82 \&\& rm -rf /build_scripts \
      RUN cp gcc gcc.bak \&\& cp g++ g++.bak \&\& rm gcc \&\& rm g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++ \
      ENV PATH=/usr/local/gcc-8.2/bin:$PATH #g' Dockerfile.tmp
  else
    sed -i 's#<install_gcc>#RUN apt-get update \
      WORKDIR /usr/bin \
      RUN apt install -y gcc-4.8 g++-4.8 \&\& cp gcc gcc.bak \&\& cp g++ g++.bak \&\& rm gcc \&\& rm g++ \&\& ln -s gcc-4.8 gcc \&\& ln -s g++-4.8 g++ #g' Dockerfile.tmp
  fi
}



function make_dockerfile(){
  sed "s/<baseimg>/${docker_name}/g" tools/dockerfile/Dockerfile.ubuntu18 >Dockerfile.tmp
}


function main(){
  make_dockerfile
  install_gcc
  ref_whl
  install_whl
}

main $@
