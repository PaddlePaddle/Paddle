#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -ex

if [ -f "/etc/redhat-release" ];then
  lib_so_5=/usr/lib64/libgfortran.so.5
  lib_so_6=/usr/lib64/libstdc++.so.6
  lib_path=/usr/lib64
else
  lib_so_5=/usr/lib/x86_64-linux-gnu/libstdc++.so.5
  lib_so_6=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  lib_path=/usr/lib/x86_64-linux-gnu
fi

if [ "$1" == "gcc82" ]; then
  wget https://paddle-ci.gz.bcebos.com/gcc-8.2.0.tar.xz 
  tar -xvf gcc-8.2.0.tar.xz && \
  cd gcc-8.2.0 && \
  unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
  ./contrib/download_prerequisites && \
  cd .. && mkdir temp_gcc82 && cd temp_gcc82 && \
  ../gcc-8.2.0/configure --prefix=/usr/local/gcc-8.2 --enable-threads=posix --disable-checking --disable-multilib && \
  make -j8 && make install
  cd .. && rm -rf temp_gcc82
  cp ${lib_so_6} ${lib_so_6}.bak  && rm -f ${lib_so_6} && 
  ln -s /usr/local/gcc-8.2/lib64/libgfortran.so.5 ${lib_so_5} && \
  ln -s /usr/local/gcc-8.2/lib64/libstdc++.so.6 ${lib_so_6} && \
  cp /usr/local/gcc-8.2/lib64/libstdc++.so.6.0.25 ${lib_path}
fi
