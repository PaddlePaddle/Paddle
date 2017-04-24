#!/bin/bash
set -e
docker run --rm -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "WITH_DOC=ON" paddledev/paddle:dev
