#!/bin/bash
set -e

# Create the build directory for CMake.
mkdir -p $TRAVIS_BUILD_DIR/build
cd $TRAVIS_BUILD_DIR/build

# Compile Documentation only.
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_DOC=ON -DWITH_STYLE_CHECK=OFF
make -j `nproc` gen_proto_py framework_py_proto
make -j `nproc` copy_paddle_pybind
make -j `nproc` paddle_docs paddle_docs_cn paddle_api_docs

# check websites for broken links
linkchecker doc/v2/en/html/index.html
linkchecker doc/v2/cn/html/index.html
linkchecker doc/v2/api/en/html/index.html
