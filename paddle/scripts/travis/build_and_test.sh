#!/bin/bash
source ./common.sh
CMAKE_EXTRA=""
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  CMAKE_EXTRA="-DPYTHON_LIBRARY=/usr/local/Cellar/python/2.7.12_1/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config/libpython2.7.dylib"
else
  CMAKE_EXTRA="-DWITH_SWIG_PY=ON"
fi


cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_GPU=OFF -DWITH_DOC=OFF -DWITH_TESTING=ON -DON_TRAVIS=ON -DON_COVERALLS=ON ${CMAKE_EXTRA}

NPROC=1
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  NRPOC=`nproc`
  make -j $NPROC
  make coveralls
elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  NPROC=`sysctl -n hw.ncpu`
  make -j $NPROC
  env CTEST_OUTPUT_ON_FAILURE=1 make test ARGS="-j $NPROC"
fi


sudo make install
sudo paddle version
