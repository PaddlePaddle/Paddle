#!/bin/bash
function make_ubuntu_dockerfile(){
  sed 's/<baseimg>/10.1-cudnn7-devel-ubuntu16.04/g' ./Dockerfile.ubuntu >Dockerfile.cuda10_cudnn7_gcc48_ubuntu16
  dockerfile_line=`wc -l Dockerfile.cuda10_cudnn7_gcc48_ubuntu16|awk '{print $1}'`
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate https://pslib.bj.bcebos.com/openmpi-1.4.5.tar.gz && tar -xzf openmpi-1.4.5.tar.gz && \
      cd openmpi-1.4.5 && ./configure --prefix=/usr/local && make all -j8 && make install -j8 && \
      export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH && export PATH=/usr/local/bin:$PATH && cd .. && \
      rm -rf openmpi-1.4.5.tar.gz && pip --no-cache-dir install mpi4py && ln -fs /bin/bash /bin/sh && \
      apt-get install libprotobuf-dev -y" Dockerfile.cuda10_cudnn7_gcc48_ubuntu16
  sed -i 's#<install_gcc>#RUN apt-get update \
      WORKDIR /usr/bin \
      RUN apt install -y gcc-4.8 g++-4.8 \&\& cp gcc gcc.bak \&\& cp g++ g++.bak \&\& rm gcc \&\& rm g++ \&\& ln -s gcc-4.8 gcc \&\& ln -s g++-4.8 g++ #g' Dockerfile.cuda10_cudnn7_gcc48_ubuntu16
}


function make_centos_dockerfile(){
  sed 's/<baseimg>/9.0-cudnn7-devel-centos6/g' Dockerfile.centos >Dockerfile.cuda9_cudnn7_gcc48_py35_centos6
  sed -i 's#COPY build_scripts /build_scripts#COPY tools/manylinux1/build_scripts ./build_scripts#g' Dockerfile.cuda9_cudnn7_gcc48_py35_centos6
}


function main() {
  make_ubuntu_dockerfile
  make_centos_dockerfile
}

main $@
