#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -ex

# Python versions to be installed in /opt/$VERSION_NO
# NOTE Only need python 2.7.11 for nupic.core/nupic.bindings at this time, so
# remove others to expedite build and reduce docker image size. The original
# manylinux docker image project builds many python versions.
# NOTE We added back 3.5.1, since auditwheel requires python 3.3+
CPYTHON_VERSIONS="3.7.0 3.6.0 3.5.1 2.7.11"

# openssl version to build, with expected sha256 hash of .tar.gz
# archive
OPENSSL_ROOT=openssl-1.1.0i
OPENSSL_HASH=ebbfc844a8c8cc0ea5dc10b86c9ce97f401837f3fa08c17b2cdadc118253cf99
EPEL_RPM_HASH=e5ed9ecf22d0c4279e92075a64c757ad2b38049bcf5c16c4f2b75d5f6860dc0d
DEVTOOLS_HASH=a8ebeb4bed624700f727179e6ef771dafe47651131a00a78b342251415646acc
CURL_ROOT=curl-7.49.1
CURL_HASH=eb63cec4bef692eab9db459033f409533e6d10e20942f4b060b32819e81885f1
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
yum -y install wget curl
curl -sLO https://dl.fedoraproject.org/pub/epel/6/x86_64/epel-release-6-8.noarch.rpm
check_sha256sum epel-release-6-8.noarch.rpm $EPEL_RPM_HASH

# Dev toolset (for LLVM and other projects requiring C++11 support)
curl -sLO http://people.centos.org/tru/devtools-2/devtools-2.repo
check_sha256sum devtools-2.repo $DEVTOOLS_HASH
mv devtools-2.repo /etc/yum.repos.d/devtools-2.repo
rpm -Uvh --replacepkgs epel-release-6*.rpm
rm -f epel-release-6*.rpm

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

wget -q https://cmake.org/files/v3.5/cmake-3.5.2.tar.gz && tar xzf cmake-3.5.2.tar.gz && \
cd cmake-3.5.2 && ./bootstrap && \
make -j8 && make install && cd .. && rm cmake-3.5.2.tar.gz


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

PY35_BIN=/opt/python/cp35-cp35m/bin
PY36_BIN=/opt/python/cp36-cp36m/bin
PY37_BIN=/opt/python/cp37-cp37m/bin
# NOTE Since our custom manylinux image builds pythons with shared
# libpython, we need to add libpython's dir to LD_LIBRARY_PATH before running
# python.
ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname ${PY35_BIN})/lib:$(dirname ${PY36_BIN})/lib:$(dirname ${PY37_BIN})/lib"

# Our openssl doesn't know how to find the system CA trust store
#   (https://github.com/pypa/manylinux/issues/53)
# And it's not clear how up-to-date that is anyway
# So let's just use the same one pip and everyone uses
LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname ${PY35_BIN})/lib" $PY35_BIN/pip install certifi
ln -s $($PY35_BIN/python -c 'import certifi; print(certifi.where())') \
      /opt/_internal/certs.pem
# If you modify this line you also have to modify the versions in the
# Dockerfiles:
export SSL_CERT_FILE=/opt/_internal/certs.pem

# Install newest curl
build_curl $CURL_ROOT $CURL_HASH
rm -rf /usr/local/include/curl /usr/local/lib/libcurl* /usr/local/lib/pkgconfig/libcurl.pc
hash -r
curl --version
curl-config --features

# Now we can delete our built SSL
rm -rf /usr/local/ssl

# Install patchelf (latest with unreleased bug fixes)
yum install -y patchelf

# Install latest pypi release of auditwheel
LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname ${PY35_BIN})/lib" $PY35_BIN/pip install auditwheel
ln -s $PY35_BIN/auditwheel /usr/local/bin/auditwheel

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
