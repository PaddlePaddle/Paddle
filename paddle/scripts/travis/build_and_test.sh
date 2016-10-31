#!/bin/bash
source ./common.sh
cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_GPU=OFF -DWITH_DOC=OFF -DWITH_TESTING=ON -DON_TRAVIS=ON -DON_COVERALLS=ON
make -j `nproc`
make coveralls
sudo make install
sudo paddle version
