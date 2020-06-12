#!/bin/bash


function centos() {
  # centos6
  sed 's#<baseimg>#8.0-cudnn7-devel-centos6#g'  Dockerfile.centos >test/centos_6_cpu_runtime.dockerfile 
  sed 's#<baseimg>#9.0-cudnn7-devel-centos6#g'  Dockerfile.centos >test/centos_6_gpu_cuda9.0_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#9.1-cudnn7-devel-centos6#g'  Dockerfile.centos >test/centos_6_gpu_cuda9.1_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#9.2-cudnn7-devel-centos6#g'  Dockerfile.centos >test/centos_6_gpu_cuda9.2_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#10.0-cudnn7-devel-centos6#g' Dockerfile.centos >test/centos_6_gpu_cuda10.0_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#10.1-cudnn7-devel-centos6#g' Dockerfile.centos >test/centos_6_gpu_cuda10.1_cudnn7_single_gpu_runtime.dockerfile
  
  # centos7
  sed 's#<baseimg>#8.0-cudnn7-devel-centos7#g'  Dockerfile.centos >test/centos_7_cpu_runtime.dockerfile 
  sed 's#<baseimg>#9.0-cudnn7-devel-centos7#g'  Dockerfile.centos >test/centos_7_gpu_cuda9.0_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#9.1-cudnn7-devel-centos7#g'  Dockerfile.centos >test/centos_7_gpu_cuda9.1_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#9.2-cudnn7-devel-centos7#g'  Dockerfile.centos >test/centos_7_gpu_cuda9.2_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#10.0-cudnn7-devel-centos7#g' Dockerfile.centos >test/centos_7_gpu_cuda10.0_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#10.1-cudnn7-devel-centos7#g' Dockerfile.centos >test/centos_7_gpu_cuda10.1_cudnn7_single_gpu_runtime.dockerfile
}


function ubuntu() {
  # ubuntu 14
  sed 's#<baseimg>#8.0-cudnn7-devel-ubuntu14.04#g'  Dockerfile.ubuntu >test/ubuntu_1404_cpu.dockerfile
  sed 's#<baseimg>#9.0-cudnn7-devel-ubuntu14.04#g'  Dockerfile.ubuntu >test/ubuntu_1404_gpu_cuda9.0_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#9.1-cudnn7-devel-ubuntu14.04#g'  Dockerfile.ubuntu >test/ubuntu_1404_gpu_cuda9.1_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#9.2-cudnn7-devel-ubuntu14.04#g'  Dockerfile.ubuntu >test/ubuntu_1404_gpu_cuda9.2_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#10.0-cudnn7-devel-ubuntu14.04#g' Dockerfile.ubuntu >test/ubuntu_1404_gpu_cuda10.0_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#10.1-cudnn7-devel-ubuntu14.04#g' Dockerfile.ubuntu >test/ubuntu_1404_gpu_cuda10.1_cudnn7_runtime.dockerfile
 
  # ubuntu 16
  sed 's#<baseimg>#8.0-cudnn7-devel-ubuntu16.04#g'  Dockerfile.ubuntu >test/ubuntu_1604_cpu.dockerfile
  sed 's#<baseimg>#9.0-cudnn7-devel-ubuntu16.04#g'  Dockerfile.ubuntu >test/ubuntu_1604_gpu_cuda9.0_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#9.1-cudnn7-devel-ubuntu16.04#g'  Dockerfile.ubuntu >test/ubuntu_1604_gpu_cuda9.1_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#9.2-cudnn7-devel-ubuntu16.04#g'  Dockerfile.ubuntu >test/ubuntu_1604_gpu_cuda9.2_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#10.0-cudnn7-devel-ubuntu16.04#g' Dockerfile.ubuntu >test/ubuntu_1604_gpu_cuda10.0_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#10.1-cudnn7-devel-ubuntu16.04#g' Dockerfile.ubuntu >test/ubuntu_1604_gpu_cuda10.1_cudnn7_runtime.dockerfile

  # ubuntu 18
  sed 's#<baseimg>#8.0-cudnn7-devel-ubuntu18.04#g'  Dockerfile.ubuntu >test/ubuntu_1804_cpu.dockerfile
  sed 's#<baseimg>#9.0-cudnn7-devel-ubuntu18.04#g'  Dockerfile.ubuntu >test/ubuntu_1804_gpu_cuda9.0_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#9.1-cudnn7-devel-ubuntu18.04#g'  Dockerfile.ubuntu >test/ubuntu_1804_gpu_cuda9.1_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#9.2-cudnn7-devel-ubuntu18.04#g'  Dockerfile.ubuntu >test/ubuntu_1804_gpu_cuda9.2_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#10.0-cudnn7-devel-ubuntu18.04#g' Dockerfile.ubuntu >test/ubuntu_1804_gpu_cuda10.0_cudnn7_runtime.dockerfile
  sed 's#<baseimg>#10.1-cudnn7-devel-ubuntu18.04#g' Dockerfile.ubuntu >test/ubuntu_1804_gpu_cuda10.1_cudnn7_runtime.dockerfile
}


function main() {
  if [ ! -d "test" ];then
    mkdir test
  fi

  centos
  ubuntu
}


main
