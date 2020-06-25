#!/bin/bash
function make_ubuntu_dockerfile(){
  sed 's/<baseimg>/10.1-cudnn7-devel-ubuntu16.04/g' ./Dockerfile.ubuntu >Dockerfile.cuda10_cudnn7_gcc82_ubuntu16
  dockerfile_line=`wc -l Dockerfile.cuda10_cudnn7_gcc48_ubuntu16|awk '{print $1}'`

  sed -i 's#liblzma-dev#liblzma-dev openmpi-bin openmpi-doc libopenmpi-dev#g' Dockerfile.cuda10_cudnn7_gcc82_ubuntu16

  sed -i 's#<install_gcc>#WORKDIR /usr/bin \
      COPY tools/dockerfile/build_scripts /build_scripts \
      RUN bash /build_scripts/install_gcc.sh gcc82 \&\& rm -rf /build_scripts \
      RUN cp gcc gcc.bak \&\& cp g++ g++.bak \&\& rm gcc \&\& rm g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++ \
      ENV PATH=/usr/local/gcc-8.2/bin:$PATH #g' Dockerfile.cuda10_cudnn7_gcc82_ubuntu16

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
