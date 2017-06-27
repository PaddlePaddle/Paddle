#!/bin/bash
function abort(){
    echo "Your change doesn't follow PaddlePaddle's code style." 1>&2
    echo "Please use pre-commit to reformat your code and git push again." 1>&2
    exit 1
}

trap 'abort' 0
set -e

cd $TRAVIS_BUILD_DIR
export PATH=/usr/bin:$PATH
pre-commit install
clang-format --version

if ! pre-commit run -a ; then
  git diff  --exit-code
fi

trap : 0
