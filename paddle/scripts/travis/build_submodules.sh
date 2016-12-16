#!/bin/bash
set -e
WORK_DIR=$PWD
PROJ_ROOT=$(git rev-parse --show-cdup)
SUBMODULES=$(grep path ${PROJ_ROOT}.gitmodules | sed 's/^.*path = //')

for module in $SUBMODULES
do
  case $module in
    "warp-ctc")
      if [ -d ${PROJ_ROOT}warp-ctc/build ]; then
        rm -rf ${PROJ_ROOT}warp-ctc/build
      fi
      mkdir ${PROJ_ROOT}warp-ctc/build
      cd ${PROJ_ROOT}warp-ctc/build
      cmake ..; make
    ;;
  esac
done
cd $WORK_DIR
