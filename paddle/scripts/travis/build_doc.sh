#!/bin/bash
set -e

# Create the build directory for CMake.
mkdir -p $TRAVIS_BUILD_DIR/build
cd $TRAVIS_BUILD_DIR/build

# Compile Documentation only.
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_DOC=ON

make -j `nproc` paddle_docs paddle_apis

# check websites for broken links
linkchecker doc/v2/en/html/index.html
linkchecker doc/v2/cn/html/index.html
linkchecker doc/v2/api/en/html/index.html
