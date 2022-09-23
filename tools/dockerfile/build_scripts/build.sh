#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -ex

# Python versions to be installed in /opt/$VERSION_NO
# NOTE Only need python 2.7.11 for nupic.core/nupic.bindings at this time, so
# remove others to expedite build and reduce docker image size. The original
# manylinux docker image project builds many python versions.
# NOTE We added back 3.5.1, since auditwheel requires python 3.3+
CPYTHON_VERSIONS="3.9.0 3.8.0 3.7.0 3.6.0"

# openssl version to build, with expected sha256 hash of .tar.gz
# archive
OPENSSL_ROOT=openssl-1.0.2g
OPENSSL_HASH=b784b1b3907ce39abf4098702dade6365522a253ad1552e267a9a0e89594aa33
PATCHELF_HASH=f2aa40a6148cb3b0ca807a1bf836b081793e55ec9e5540a5356d800132be7e0a
AUTOCONF_ROOT=autoconf-2.69
AUTOCONF_HASH=954bd69b391edc12d6a4a51a2dd1476543da5c6bbf05a95b59dc0dd6fd4c2969

# Dependencies for compiling Python that we want to remove from
# the final image after compiling Python
PYTHON_COMPILE_DEPS="zlib-devel bzip2-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel libffi-devel"

# Libraries that are allowed as part of the manylinux1 profile
MANYLINUX1_DEPS="glibc-devel libstdc++-devel glib2-devel libX11-devel libXext-devel libXrender-devel  mesa-libGL-devel libICE-devel libSM-devel ncurses-devel freetype-devel libpng-devel"

# Get build utilities
MY_DIR=$(dirname "${BASH_SOURCE[0]}")
source $MY_DIR/build_utils.sh

# EPEL support
yum -y install wget curl epel-release

# Development tools and libraries
yum -y install bzip2 make git patch unzip bison yasm diffutils \
    automake which file \
    kernel-devel-`uname -r` \
    devtoolset-2-binutils devtoolset-2-gcc \
    devtoolset-2-gcc-c++ devtoolset-2-gcc-gfortran \
    ${PYTHON_COMPILE_DEPS}

# Install more recent version of cmake
# curl -O https://cmake.org/files/v3.8/cmake-3.8.1-Linux-x86_64.sh
# /bin/sh cmake-3.8.1-Linux-x86_64.sh --prefix=/usr/local --skip-license
# rm cmake-3.8.1-Linux-x86_64.sh

wget -q https://cmake.org/files/v3.16/cmake-3.16.0.tar.gz && tar xzf cmake-3.16.0.tar.gz && \
cd cmake-3.16.0 && ./bootstrap && \
make -j8 && make install && cd .. && rm cmake-3.16.0.tar.gz && rm -rf cmake-3.16.0

# Install newest autoconf
build_autoconf $AUTOCONF_ROOT $AUTOCONF_HASH
autoconf --version

# Compile the latest Python releases.
# (In order to have a proper SSL module, Python is compiled
# against a recent openssl [see env vars above], which is linked
# statically. We delete openssl afterwards.)
build_openssl $OPENSSL_ROOT $OPENSSL_HASH
mkdir -p /opt/python
build_cpythons $CPYTHON_VERSIONS

PY36_BIN=/opt/python/cp36-cp36m/bin
PY37_BIN=/opt/python/cp37-cp37m/bin
PY38_BIN=/opt/python/cp38-cp38m/bin
PY39_BIN=/opt/python/cp39-cp39m/bin
# NOTE Since our custom manylinux image builds pythons with shared
# libpython, we need to add libpython's dir to LD_LIBRARY_PATH before running
# python.
ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname ${PY36_BIN})/lib:$(dirname ${PY37_BIN})/lib:$(dirname ${PY38_BIN})/lib:$(dirname ${PY39_BIN})/lib"

# Our openssl doesn't know how to find the system CA trust store
#   (https://github.com/pypa/manylinux/issues/53)
# And it's not clear how up-to-date that is anyway
# So let's just use the same one pip and everyone uses
LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname ${PY37_BIN})/lib" $PY37_BIN/pip install certifi
ln -s $($PY37_BIN/python -c 'import certifi; print(certifi.where())') \
      /opt/_internal/certs.pem
# If you modify this line you also have to modify the versions in the
# Dockerfiles:
export SSL_CERT_FILE=/opt/_internal/certs.pem


# Install patchelf (latest with unreleased bug fixes)
# FIXME(typhoonzero): restore this when the link is fixed.
# curl -sLO http://nipy.bic.berkeley.edu/manylinux/patchelf-0.9njs2.tar.gz
# check_sha256sum patchelf-0.9njs2.tar.gz $PATCHELF_HASH
# tar -xzf patchelf-0.9njs2.tar.gz
# (cd patchelf-0.9njs2 && ./configure && make && make install)
# rm -rf patchelf-0.9njs2.tar.gz patchelf-0.9njs2
sh "$MY_DIR/install_patchelf.sh"

# Install latest pypi release of auditwheel
#LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname ${PY35_BIN})/lib" $PY35_BIN/pip install auditwheel
#ln -s $PY35_BIN/auditwheel /usr/local/bin/auditwheel

# Clean up development headers and other unnecessary stuff for
# final image
yum -y erase wireless-tools gtk2 libX11 hicolor-icon-theme \
    avahi freetype bitstream-vera-fonts \
    ${PYTHON_COMPILE_DEPS}  > /dev/null 2>&1 || true
yum -y install ${MANYLINUX1_DEPS} && yum -y clean all > /dev/null 2>&1 || true
yum list installed
# we don't need libpython*.a, and they're many megabytes
find /opt/_internal -name '*.a' -print0 | xargs -0 rm -f
# Strip what we can -- and ignore errors, because this just attempts to strip
# *everything*, including non-ELF files:
find /opt/_internal -type f -print0 \
    | xargs -0 -n1 strip --strip-unneeded 2>/dev/null || true
# We do not need the Python test suites, or indeed the precompiled .pyc and
# .pyo files. Partially cribbed from:
#    https://github.com/docker-library/python/blob/master/3.4/slim/Dockerfile
find /opt/_internal \
     \( -type d -a -name test -o -name tests \) \
  -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
  -print0 | xargs -0 rm -f

for PYTHON in /opt/python/*/bin/python; do
    # Add matching directory of libpython shared library to library lookup path
    LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname $(dirname ${PYTHON}))/lib"

    # Smoke test to make sure that our Pythons work, and do indeed detect as
    # being manylinux compatible:
    LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname $(dirname ${PYTHON}))/lib" $PYTHON $MY_DIR/manylinux1-check.py
    # Make sure that SSL cert checking works
    LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname $(dirname ${PYTHON}))/lib" $PYTHON $MY_DIR/ssl-check.py
done

# Restore LD_LIBRARY_PATH
LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}"

# According to ar issues: https://lists.gnu.org/archive/html/bug-binutils/2016-05/msg00211.html
# we should install new version ar with 64-bit supported here
wget --no-check-certificate https://ftp.gnu.org/gnu/binutils/binutils-2.27.tar.gz
tar xzf binutils-2.27.tar.gz && cd binutils-2.27
./configure --prefix=/opt/rh/devtoolset-2/root/usr/ --enable-64-bit-archive && make -j `nproc` && make install
cd .. && rm binutils-2.27.tar.gz && rm -rf binutils-2.27
