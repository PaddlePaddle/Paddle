#!/bin/bash
brew update
brew tap homebrew/science
brew install python
sudo pip install --upgrade protobuf==2.6.0
brew install homebrew/versions/protobuf260 --without-python
brew install cmake python glog gflags openblas wget md5sha1sum

wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz -O gtest.tar.gz
tar xf gtest.tar.gz
cd googletest-release-1.8.0/
cmake .
make install
