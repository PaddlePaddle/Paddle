#!/bin/bash
set -e

cd /tmp
wget https://github.com/google/protobuf/archive/v3.0.2.tar.gz -O protobuf.tar.gz
tar xf protobuf.tar.gz
cd protobuf*
./autogen.sh
./configure
make -j 2 install
cd ..
rm -rf protobuf*

pushd /usr/src/gtest
cmake .
make
sudo cp *.a /usr/lib
popd
