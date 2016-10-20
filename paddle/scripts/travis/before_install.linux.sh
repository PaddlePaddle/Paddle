#!/bin/bash
set -e
pushd /usr/src/gtest
cmake .
make
sudo cp *.a /usr/lib
popd
