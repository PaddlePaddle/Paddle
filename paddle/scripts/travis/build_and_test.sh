#!/bin/bash
source ./common.sh

NPROC=1
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  export PYTHONPATH=/opt/python/2.7.12/lib/python2.7/site-packages
  export PYTHONHOME=/opt/python/2.7.12
  export PATH=/opt/python/2.7.12/bin:${PATH}
  cmake .. -DON_TRAVIS=ON -DWITH_COVERAGE=ON -DCOVERALLS_UPLOAD=ON
  NRPOC=`nproc`
  make -j $NPROC
  make coveralls
  sudo make install
elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  export PYTHONPATH=/usr/local/lib/python2.7/site-packages
  cmake .. -DON_TRAVIS=ON
  NPROC=`sysctl -n hw.ncpu`
  make -j $NPROC
fi
