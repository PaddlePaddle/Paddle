#!/bin/bash

sed 's/<baseimg>/9.0-cudnn7-devel-ubuntu16.04/g' ./Dockerfile.ubuntu.gcc48 >Dockerfile.ci_ubuntu_gcc8
dockerfile_line=`wc -l Dockerfile.ci_ubuntu_gcc8|awk '{print $1}'`
sed -i "${dockerfile_line}i RUN wget --no-check-certificate https://pslib.bj.bcebos.com/openmpi-1.4.5.tar.gz && tar -xzf openmpi-1.4.5.tar.gz && \
    cd openmpi-1.4.5 && ./configure --prefix=/usr/local && make all -j8 && make install -j8 && \
    export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH && export PATH=/usr/local/bin:$PATH && cd .. && \
    rm -rf openmpi-1.4.5.tar.gz && pip --no-cache-dir install mpi4py && ln -fs /bin/bash /bin/sh && \
    apt-get install libprotobuf-dev -y" Dockerfile.ci_ubuntu_gcc8
