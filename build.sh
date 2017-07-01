#!/bin/bash

set -xe

mkdir -p ./build
cd ./build

PADDLE_INSTALL_DIR='/home/xingzhaolong/pr/paddle'
cmake .. \
      -DCUDNN_ROOT=/home/work/cudnn/cudnn_v5/cuda/ \
      -DCMAKE_INSTALL_PREFIX=/home/xingzhaolong/.jumbo/
     # -DPYTHON_LIBRARY=$PADDLE_INSTALL_DIR/python27/lib/libpython2.7.a \
     # -DCMAKE_EXE_LINKER_FLAGS="-lutil" \
     # -DPYTHON_INCLUDE_DIR=$PADDLE_INSTALL_DIR/python27/include/python2.7 \
     # -DPYTHON_EXECUTABLE=$PADDLE_INSTALL_DIR/python27/bin/python \
     # -DPY_SITE_PACKAGES_PATH=$PADDLE_INSTALL_DIR/python27/lib/python2.7/site-packages \

cat <<EOF
========================================
Building in /paddle/build ...
   Build unit tests: ${WITH_TESTING:-OFF}
========================================
EOF
make -j `nproc`
if [ ${WITH_TESTING:-OFF} == "ON" ] && [ ${RUN_TEST:-OFF} == "ON" ] ; then
    pip uninstall -y py-paddle paddle || true
    ctest -V
fi


cat <<EOF
========================================
Installing ...
========================================
EOF
make install
pip install -U /home/xingzhaolong/.jumbo/opt/paddle/share/wheels/*.whl
paddle version
