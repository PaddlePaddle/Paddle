#!/bin/bash
function abort(){
    echo "Your commit not fit PaddlePaddle code style" 1>&2
    echo "Please use pre-commit scripts to auto-format your code" 1>&2
    exit 1
}

trap 'abort' 0
set -e
source common.sh
cd ..
export PATH=/usr/bin:$PATH
pre-commit install
clang-format --version

if ! pre-commit run -a ; then
  git diff  --exit-code
fi

trap : 0
