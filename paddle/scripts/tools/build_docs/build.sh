#!/bin/bash
set -ex

mkdir -p /build
cd /build
cmake /paddle -DWITH_DOC=ON
make paddle_docs paddle_docs_cn -j `nproc`
mkdir -p /output/doc
mkdir -p /output/doc_cn
cp -r doc/html/* /output/doc/
cp -r doc_cn/html/* /output/doc_cn/
cd /
rm -rf /paddle/build
