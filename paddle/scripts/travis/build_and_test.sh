#!/bin/bash
source ./common.sh

NPROC=1
export PYTHONPATH=/opt/python/2.7.12/lib/python2.7/site-packages
export PYTHONHOME=/opt/python/2.7.12
export PATH=/opt/python/2.7.12/bin:${PATH}
cmake .. -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran-4.8 -DON_TRAVIS=ON -DWITH_COVERAGE=ON -DCOVERALLS_UPLOAD=ON ${EXTRA_CMAKE_OPTS}
NRPOC=`nproc`
make -j $NPROC
make coveralls
sudo make install
