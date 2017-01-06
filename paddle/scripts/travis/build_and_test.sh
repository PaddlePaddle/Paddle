#!/bin/bash
source ./common.sh

python -c 'import pip; print(pip.pep425tags.get_supported())'

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  CMAKE_EXTRA="-DWITH_SWIG_PY=OFF"
else
  CMAKE_EXTRA="-DWITH_SWIG_PY=ON"
fi

cmake .. -DWITH_GPU=OFF -DWITH_DOC=OFF -DWITH_TESTING=ON -DON_TRAVIS=ON -DON_COVERALLS=ON ${CMAKE_EXTRA}

NPROC=1
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  NRPOC=`nproc`
  make -j $NPROC
  make coveralls
  sudo make install
elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  NPROC=`sysctl -n hw.ncpu`
  make -j $NPROC
  env CTEST_OUTPUT_ON_FAILURE=1 make test ARGS="-j $NPROC"
  sudo make install
  sudo paddle version
fi
