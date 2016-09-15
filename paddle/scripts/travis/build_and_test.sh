#!/bin/bash
source ./common.sh
cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_GPU=OFF -DWITH_DOC=OFF -DWITH_TESTING=ON -DON_TRAVIS=ON
make -j `nproc`
env CTEST_OUTPUT_ON_FAILURE=1 make test ARGS="-j `nproc`"
sudo make install
sudo paddle version
