#!/bin/bash
function make_ubuntu_dockerfile(){
  dockerfile_name="Dockerfile.cuda10_cudnn7_gcc82_ubuntu16"
  sed 's/<baseimg>/10.1-cudnn7-devel-ubuntu16.04/g' ./Dockerfile.ubuntu >${dockerfile_name}
  sed -i 's#liblzma-dev#liblzma-dev openmpi-bin openmpi-doc libopenmpi-dev#g' ${dockerfile_name} 
  dockerfile_line=`wc -l ${dockerfile_name}|awk '{print $1}'`
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz && \
     tar -xzf  hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name} 
  sed -i 's#<install_gcc>#WORKDIR /usr/bin \
      COPY tools/dockerfile/build_scripts /build_scripts \
      RUN bash /build_scripts/install_gcc.sh gcc82 \&\& rm -rf /build_scripts \
      RUN cp gcc gcc.bak \&\& cp g++ g++.bak \&\& rm gcc \&\& rm g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++ \
      ENV PATH=/usr/local/gcc-8.2/bin:$PATH #g' ${dockerfile_name}
  sed -i 's#bash /build_scripts/install_nccl2.sh#wget --no-proxy https://nccl2-deb.cdn.bcebos.com/nccl-repo-ubuntu1604-2.7.8-ga-cuda10.1_1-1_amd64.deb && \
      dpkg -i nccl-repo-ubuntu1604-2.7.8-ga-cuda10.1_1-1_amd64.deb && \
      apt-get install -y libnccl2=2.7.8-1+cuda10.1 libnccl-dev=2.7.8-1+cuda10.1 --allow-change-held-packages #g' ${dockerfile_name}
}


function make_centos_dockerfile(){
  dockerfile_name="Dockerfile.cuda9_cudnn7_gcc48_py35_centos6"
  sed 's/<baseimg>/10.2-cudnn7-devel-centos6/g' Dockerfile.centos >${dockerfile_name}
  sed -i 's#COPY build_scripts /build_scripts#COPY tools/dockerfile/build_scripts ./build_scripts#g' ${dockerfile_name} 
  dockerfile_line=`wc -l ${dockerfile_name}|awk '{print $1}'`
  sed -i "${dockerfile_line}i RUN ln -s /usr/lib64/libz.so /usr/local/lib/libz.so && \
     ln -s /usr/local/lib/libnccl.so /usr/local/cuda/lib64/ && \
     rm -rf /usr/include/NvInfer*" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz && \
     tar -xzf  hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name}
  sed -i 's#RUN bash build_scripts/install_nccl2.sh##g' ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-2.7.8-1+cuda10.2.x86_64.rpm && \
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-devel-2.7.8-1+cuda10.2.x86_64.rpm && \
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-static-2.7.8-1+cuda10.2.x86_64.rpm && \
    rpm -ivh libnccl-2.7.8-1+cuda10.2.x86_64.rpm && \
    rpm -ivh libnccl-devel-2.7.8-1+cuda10.2.x86_64.rpm && \
    rpm -ivh libnccl-static-2.7.8-1+cuda10.2.x86_64.rpm && rm -f /usr/local/include/nccl.h " ${dockerfile_name}
  sed -i 's#<install_gcc>#WORKDIR /usr/bin \
      COPY tools/dockerfile/build_scripts /build_scripts \
      RUN bash /build_scripts/install_gcc.sh gcc82 \&\& rm -rf /build_scripts \
      RUN cp gcc gcc.bak \&\& cp g++ g++.bak \&\& rm gcc \&\& rm g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++ \
      ENV PATH=/usr/local/gcc-8.2/bin:$PATH #g' ${dockerfile_name}
}


function main() {
  make_ubuntu_dockerfile
  make_centos_dockerfile
}

main $@
