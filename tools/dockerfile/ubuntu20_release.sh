#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

function base_image(){
  if [[ ${ref_CUDA_MAJOR} == "12.8" ]]; then
    dockerfile_name="Dockerfile-128"
    sed "s#<baseimg>#nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04#g" ./Dockerfile.release.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "12.9" ]]; then
    dockerfile_name="Dockerfile-129"
    sed "s#<baseimg>#nvidia/cuda:12.9.0-cudnn-devel-ubuntu20.04#g" ./Dockerfile.release.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "0" ]];then
    dockerfile_name="Dockerfile-cpu"
    sed "s#<baseimg>#ubuntu:20.04#g" ./Dockerfile.release.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64#g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>#RUN apt-get install -y gcc g++ make#g' ${dockerfile_name}
    sed -i 's#ENV WITH_GPU=${WITH_GPU:-ON}#ENV WITH_GPU=${WITH_GPU:-OFF}#g' ${dockerfile_name}
  else
    echo "Dockerfile ERROR!!!"
    exit 1
  fi

}


export ref_CUDA_MAJOR=0
base_image
export ref_CUDA_MAJOR=12.8
base_image
export ref_CUDA_MAJOR=12.9
base_image
