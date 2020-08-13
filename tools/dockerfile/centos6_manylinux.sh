#!/bin/bash
set -xe

REPO="${REPO:-paddledocker}"

function make_cuda9cudnn7(){
  sed 's/<baseimg>/9.0-cudnn7-devel-centos6/g' Dockerfile.centos >Dockerfile.tmp
}


function make_cuda10cudnn7() {
  sed 's/<baseimg>/10.0-cudnn7-devel-centos6/g' Dockerfile.centos >Dockerfile.tmp
}


function make_cuda101cudnn7() {
  sed 's/<baseimg>/10.1-cudnn7-devel-centos6/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#COPY build_scripts /build_scripts#COPY build_scripts /build_scripts \nRUN bash build_scripts/install_gcc.sh gcc82 \nENV PATH=/usr/local/gcc-8.2/bin:\$PATH#g" Dockerfile.tmp
}


function main() {
  local CMD=$1 
  case $CMD in
    cuda9cudnn7)
      make_cuda9cudnn7
      ;;
    cuda10cudnn7)
      make_cuda10cudnn7
      ;;
    cuda101cudnn7)
      make_cuda101cudnn7
      ;;
    *)
      echo "Make dockerfile error, Without this paramet."
      exit 1
      ;;
  esac
}

main $@
