#!/bin/bash

BUILD_DIR=$PWD/build
DEB_DIST_DIR=$PWD/dist
VERSION=latest

function build_in_docker() {
  if [ ! -d $BUILD_DIR ]; then
    mkdir -p $BUILD_DIR
  fi
  if [ ! -d $DEB_DIST_DIR ]; then
    mkdir -p $DEB_DIST_DIR
  fi
  docker build . -t paddle-build-env -f paddle/scripts/docker/paddle-dev/Dockerfile
  # FIXME: need to wait a signal not sleeping
  BUILDER=$(docker run -d -v ${PWD}:/root/paddle -v ${DEB_DIST_DIR}:/root/dist paddle-build-env sleep 3600)
  # NOTICE: build deb files for real paddle image
  docker exec $BUILDER /bin/bash -c "/root/paddle/paddle/scripts/deb/build_scripts/build.sh"

  docker stop $BUILDER && docker rm $BUILDER
}

function build_paddle_core() {
  docker build . -t paddle:$VERSION -f paddle/scripts/docker/paddle-core/Dockerfile
  docker build . -t paddle:gpu-$VERSION -f paddle/scripts/docker/paddle-core/Dockerfile.gpu
  docker build . -t paddle:cpu-noavx-$VERSION -f paddle/scripts/docker/paddle-core/Dockerfile.noavx
  docker build . -t paddle:gpu-noavx-$VERSION -f paddle/scripts/docker/paddle-core/Dockerfile.gpunoavx
}

build_in_docker
build_paddle_core
