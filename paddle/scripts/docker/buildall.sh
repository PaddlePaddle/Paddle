#!/bin/bash

BINARIES_DIR=paddle/scripts/docker/buildimage/binaries
BUILD_DIR=$PWD/build

function build_in_docker() {
  if [ ! -d $BUILD_DIR ]; then
    mkdir -p $BUILD_DIR
  fi
  docker build . -t paddle-build-env -f paddle/scripts/docker/paddle-dev/Dockerfile
  # FIXME: need to wait a signal not sleeping
  BUILDER=$(docker run -d -v ${PWD}:/paddle  paddle-build-env sleep 3600)
  # TODO(typhoonzero):
  docker exec $BUILDER /bin/bash -c "export BUILD_AND_INSTALL=ON && /paddle/paddle/scripts/docker/build.sh"
  mkdir -p $BINARIES_DIR
  # docker cp $BUILDER:/usr/local/opt/paddle/bin/paddle_pserver_main $BINARIES_DIR
  # docker cp $BUILDER:/usr/local/opt/paddle/bin/paddle_trainer $BINARIES_DIR
  # docker cp $BUILDER:/usr/local/opt/paddle/bin/paddle_merge_model $BINARIES_DIR
  # docker cp $BUILDER:/usr/local/bin/paddle $BINARIES_DIR
  # docker cp $BUILDER:/usr/local/opt/paddle/bin/paddle_usage $BINARIES_DIR
  #
  # docker cp $BUILDER:/usr/local/opt/paddle/share/wheels $BINARIES_DIR

  docker stop $BUILDER && docker rm $BUILDER
}

function build_paddle_core() {
  docker build . -t paddle-core -f paddle/scripts/docker/paddle-core/Dockerfile
}

build_in_docker
#build_paddle_core
