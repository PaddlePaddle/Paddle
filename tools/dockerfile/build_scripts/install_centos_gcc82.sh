#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -ex

if [ "$1" == "gcc82" ]; then
cd /usr/bin && wget -q http://mirror.linux-ia64.org/gnu/gcc/releases/gcc-8.2.0/gcc-8.2.0.tar.xz && \
tar -xvf gcc-8.2.0.tar.xz && \
cd gcc-8.2.0 && \
unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
./contrib/download_prerequisites && \
cd .. && mkdir temp_gcc82 && cd temp_gcc82 && \
../gcc-8.2.0/configure --prefix=/usr/local/gcc-8.2 --enable-threads=posix --disable-checking --disable-multilib && \
make -j40 && make install && \
cd /usr/bin && cp gcc gcc.bak && cp g++ g++.bak && rm gcc && rm g++ && \
ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc && ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ && ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc && ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++ && cd /usr/bin && rm -rf /usr/bin/temp_gcc82 && \
cp /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so.6.bak && rm -f /usr/lib64/libstdc++.so.6 && ln -s /usr/local/gcc-8.2/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so.6 && cp /usr/local/gcc-8.2/lib64/libstdc++.so.6.0.25 /usr/lib64/
fi
