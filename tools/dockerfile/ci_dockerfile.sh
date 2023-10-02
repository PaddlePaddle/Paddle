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


function make_ubuntu_trt7_dockerfile(){
  dockerfile_name="Dockerfile.cuda102_cudnn8_gcc82_ubuntu16"
  sed "s/<baseimg>/10.2-cudnn8-devel-ubuntu16.04/g" ./Dockerfile.ubuntu >${dockerfile_name}
  sed -i "s#liblzma-dev#liblzma-dev openmpi-bin openmpi-doc libopenmpi-dev#g" ${dockerfile_name} 
  dockerfile_line=$(wc -l ${dockerfile_name}|awk '{print $1}')
  sed -i "${dockerfile_line}i RUN apt remove -y libcudnn* --allow-change-held-packages \&\& \
      apt-get install -y --allow-unauthenticated libsndfile1 libcudnn8=8.1.0.77-1+cuda10.2 libcudnn8-dev=8.1.0.77-1+cuda10.2 --allow-change-held-packages" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q  \
      https://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/2/cuda_10.2.2_linux.run \&\& \
      bash cuda_10.2.2_linux.run --silent --toolkit \&\& ldconfig" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q  \
      https://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/1/cuda_10.2.1_linux.run \&\& \
      bash cuda_10.2.1_linux.run --silent --toolkit \&\& ldconfig" ${dockerfile_name}
  sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh ubuntu1604-7234#g' ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz \&\& \
     tar -xzf     hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN apt remove git -y \&\& apt install -y libcurl4-openssl-dev gettext zstd ninja-build  \&\& wget -q https://paddle-ci.gz.bcebos.com/git-2.17.1.tar.gz \&\& \
    tar -xvf git-2.17.1.tar.gz \&\& \
    cd git-2.17.1 \&\& \
    ./configure --with-openssl --with-curl --prefix=/usr/local \&\& \
    make -j8 \&\& make install " ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN pip install wheel \&\& pip3 install PyGithub wheel \&\& pip3.7 install PyGithub distro \&\& pip3.8 install PyGithub distro" ${dockerfile_name}
  sed -i "s#<install_gcc>#WORKDIR /usr/bin \\
    COPY tools/dockerfile/build_scripts /build_scripts \\
    RUN bash /build_scripts/install_gcc.sh gcc82 \&\& rm -rf /build_scripts \\
    RUN cp gcc  gcc.bak \&\& cp g++  g++.bak \&\& rm gcc \&\& rm g++ \\
    RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc \\
    RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ \\
    RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc \\
    RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++ \\
    ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
  sed -i "s#bash /build_scripts/install_nccl2.sh#wget -q --no-proxy https://nccl2-deb.gz.bcebos.com/nccl-local-repo-ubuntu1604-2.8.4-cuda10.2_1.0-1_amd64.deb \\
    RUN apt remove -y libnccl* --allow-change-held-packages \&\&  apt-get install -y --allow-unauthenticated libsndfile1 libnccl2=2.8.4-1+cuda10.2 libnccl-dev=2.8.4-1+cuda10.2 zstd pigz --allow-change-held-packages #g" ${dockerfile_name}
}

function make_ubuntu_trt7_dockerfile_temp_ues(){
  dockerfile_name="Dockerfile.cuda102_cudnn8_gcc82_ubuntu16"
  echo "FROM registry.baidubce.com/paddlepaddle/paddleqa:coverage-ci-temp-use" >> ${dockerfile_name}
  echo "RUN wget https://www.openssl.org/source/openssl-1.1.1v.tar.gz && tar -xvf openssl-1.1.1v.tar.gz && cd openssl-1.1.1v && ./config -fPIC --prefix=/usr/local/ssl > /dev/null && make > /dev/null && make install > /dev/null && cd ../ && rm -rf openssl-1.1.1v*" >> ${dockerfile_name}
  echo "ENV OPENSSL_ROOT_DIR=/usr/local/ssl" >> ${dockerfile_name}
  echo "ENV LD_LIBRARY_PATH=/usr/local/ssl/lib:\g$LD_LIBRARY_PATH" >> ${dockerfile_name}
}

function make_cpu_dockerfile(){
  dockerfile_name="Dockerfile.cuda9_cudnn7_gcc48_py35_centos6"
  sed "s#<baseimg>#ubuntu:20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
  sed -i "s#<setcuda>##g" ${dockerfile_name}
  sed -i "s#WITH_GPU:-ON#WITH_GPU:-OFF#g" ${dockerfile_name}
  sed -i "s#RUN apt-key del 7fa2af80##g" ${dockerfile_name}
  sed -i 's#RUN rm /etc/apt/sources.list.d/\*##g' ${dockerfile_name}
  sed -i "s#RUN apt-key adv --fetch-keys https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub##g" ${dockerfile_name}
  dockerfile_line=$(wc -l ${dockerfile_name}|awk '{print $1}')
  sed -i 's#RUN bash /build_scripts/install_trt.sh##g' ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz \&\& \
     tar -xzf     hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN apt remove git -y \&\& apt update \&\& apt install -y libsndfile1 zstd pigz ninja-build" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN pip install wheel PyGithub distro" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN apt remove git -y \&\& apt update \&\& apt install -y libcurl4-openssl-dev gettext pigz \&\& wget -q https://paddle-ci.gz.bcebos.com/git-2.17.1.tar.gz \&\& \
    tar -xvf git-2.17.1.tar.gz \&\& \
    cd git-2.17.1 \&\& \
    ./configure --with-openssl --with-curl --prefix=/usr/local \&\& \
    make -j8 \&\& make install " ${dockerfile_name}
  sed -i 's#<install_cpu_package>#RUN apt-get install -y gcc g++ make#g' ${dockerfile_name}
}


function make_ce_framework_dockcerfile(){
  dockerfile_name="Dockerfile.cuda11.2_cudnn8_gcc82_trt8"
  sed "s#<baseimg>#nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
  dockerfile_line=$(wc -l ${dockerfile_name}|awk '{print $1}')
  sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
  sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
  sed -i "7i RUN chmod 777 /tmp" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz \&\& \
     tar -xzf  hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name} 
  sed -i "${dockerfile_line}i RUN apt install -y zstd pigz libcurl4-openssl-dev gettext ninja-build" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN pip install wheel distro" ${dockerfile_name}
  sed -i "s#<install_gcc>#WORKDIR /usr/bin \\
    ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
  sed -i "s#gcc121#gcc82#g" ${dockerfile_name}
  sed -i "s#gcc-12.1#gcc-8.2#g" ${dockerfile_name}
  sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8531#g' ${dockerfile_name}
}


function make_unbuntu20_cu12_dockerfile(){
  dockerfile_name="Dockerfile.cuda117_cudnn8_gcc82_ubuntu18_coverage"
  sed "s#<baseimg>#nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
  sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.0/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
  sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
  sed -i "7i ENV TZ=Asia/Beijing" ${dockerfile_name}
  sed -i "8i RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone" ${dockerfile_name}
  sed -i "27i RUN apt-get update && apt-get install -y liblzma-dev openmpi-bin openmpi-doc libopenmpi-dev libsndfile1" ${dockerfile_name}
  dockerfile_line=$(wc -l ${dockerfile_name}|awk '{print $1}')
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz \&\& \
     tar -xzf  hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN apt remove git -y \&\& apt update \&\& apt install -y libcurl4-openssl-dev gettext pigz zstd ninja-build \&\& wget -q https://paddle-ci.gz.bcebos.com/git-2.17.1.tar.gz \&\& \
    tar -xvf git-2.17.1.tar.gz \&\& \
    cd git-2.17.1 \&\& \
    ./configure --with-openssl --with-curl --prefix=/usr/local \&\& \
    make -j8 \&\& make install " ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN pip install wheel \&\& pip3 install PyGithub wheel distro \&\& pip3.7 install PyGithub \&\& pip3.8 install distro" ${dockerfile_name}
  sed -i 's# && rm /etc/apt/sources.list.d/nvidia-ml.list##g' ${dockerfile_name}
  sed -i 's#Run bash /build_scripts/install_cudnn.sh cudnn841#RUN bash /build_scripts/install_cudnn.sh cudnn891#g' ${dockerfile_name}
}


function make_ubuntu20_cu112_dockerfile(){
  dockerfile_name="Dockerfile.cuda11.2_cudnn8.1_trt8.4_gcc8.2_ubuntu18"
  sed "s#<baseimg>#nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
  sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.2/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
  dockerfile_line=$(wc -l ${dockerfile_name}|awk '{print $1}')
  sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8431#g' ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz \&\& \
     tar -xzf     hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN apt remove git -y \&\& apt install -y libsndfile1 zstd pigz libcurl4-openssl-dev gettext zstd ninja-build \&\& wget -q https://paddle-ci.gz.bcebos.com/git-2.17.1.tar.gz \&\& \
    tar -xvf git-2.17.1.tar.gz \&\& \
    cd git-2.17.1 \&\& \
    ./configure --with-openssl --with-curl --prefix=/usr/local \&\& \
    make -j8 \&\& make install " ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN pip install wheel \&\& pip3 install PyGithub wheel \&\& pip3.7 install PyGithub distro \&\& pip3.8 install PyGithub distro" ${dockerfile_name}
  sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
  sed -i "s#<install_gcc>#WORKDIR /usr/bin \\
    ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
  sed -i "s#gcc121#gcc82#g" ${dockerfile_name}
  sed -i "s#gcc-12.1#gcc-8.2#g" ${dockerfile_name}
  sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8531#g' ${dockerfile_name}
}

function main() {
  make_ubuntu_trt7_dockerfile_temp_ues
  make_cpu_dockerfile
  make_ce_framework_dockcerfile
  make_unbuntu20_cu12_dockerfile
  make_ubuntu20_cu112_dockerfile
}

main "$@"
