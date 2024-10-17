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

# Helper utilities for build

PYTHON_DOWNLOAD_URL=https://www.python.org/ftp/python
# XXX: the official https server at www.openssl.org cannot be reached
# with the old versions of openssl and curl in Centos 5.11 hence the fallback
# to the ftp mirror:
# OPENSSL_DOWNLOAD_URL=ftp://ftp.openssl.org/source
OPENSSL_DOWNLOAD_URL=https://www.openssl.org/source
# Ditto the curl sources
CURL_DOWNLOAD_URL=http://curl.askapache.com/download

GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py

AUTOCONF_DOWNLOAD_URL=http://ftp.gnu.org/gnu/autoconf


function check_var {
    if [ -z "$1" ]; then
        echo "required variable not defined"
        exit 1
    fi
}


function lex_pyver {
    # Echoes Python version string padded with zeros
    # Thus:
    # 3.2.1 -> 003002001
    # 3     -> 003000000
    echo $1 | awk -F "." '{printf "%03d%03d%03d", $1, $2, $3}'
}


function do_cpython_build {
    local py_ver=$1
    check_var $py_ver
    local ucs_setting=$2
    check_var $ucs_setting
    tar -xzf Python-$py_ver.tgz
    pushd Python-$py_ver
    if [ "$ucs_setting" = "none" ]; then
        unicode_flags=""
        dir_suffix=""
    else
        local unicode_flags="--enable-unicode=$ucs_setting"
        local dir_suffix="-$ucs_setting"
    fi
    local prefix="/opt/_internal/cpython-${py_ver}${dir_suffix}"
    mkdir -p ${prefix}/lib

    # NOTE --enable-shared for generating libpython shared library needed for
    # linking of some of the nupic.core test executables.
    if [ $(lex_pyver $py_ver) -ge $(lex_pyver 3.7) ]; then
        # NOTE python 3.7 should be installed via make altinstall rather than
        # make install, and we should specify the location of ssl
        CFLAGS="-Wformat" ./configure --prefix=${prefix} --with-openssl=/usr/local/ssl --enable-shared $unicode_flags > /dev/null
        make -j8 > /dev/null
        make altinstall > /dev/null
    else
        LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} CFLAGS="-Wformat" ./configure --prefix=${prefix} --enable-shared $unicode_flags > /dev/null
        LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} make -j8 > /dev/null
        LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} make install > /dev/null
    fi
    popd
    echo "ZZZ looking for libpython"
    find / -name 'libpython*.so*'
    rm -rf Python-$py_ver
    # Some python's install as bin/python3. Make them available as
    # bin/python.
    if [ -e ${prefix}/bin/python3.8 ]; then
        ln -s python3.8 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.9 ]; then
        ln -s python3.9 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.10 ]; then
        ln -s python3.10 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.11 ]; then
        ln -s python3.11 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.12 ]; then
        ln -s python3.12 ${prefix}/bin/python
    fi
    # NOTE Make libpython shared library visible to python calls below
    if [ -e ${prefix}/bin/python3.10 ] || [ -e ${prefix}/bin/python3.11 ] || [ -e ${prefix}/bin/python3.12 ]; then
        LD_LIBRARY_PATH="/usr/local/ssl/lib:${prefix}/lib" ${prefix}/bin/python -m pip config set global.trusted-host mirrors.aliyun.com
        LD_LIBRARY_PATH="/usr/local/ssl/lib:${prefix}/lib" ${prefix}/bin/python -m pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
    fi
    LD_LIBRARY_PATH="/usr/local/ssl/lib:${prefix}/lib" ${prefix}/bin/python get-pip.py
    LD_LIBRARY_PATH="/usr/local/ssl/lib:${prefix}/lib" ${prefix}/bin/pip install wheel==0.40.0
    cd /
    ls ${MY_DIR}
    abi_version=$(LD_LIBRARY_PATH="${prefix}/lib" ${prefix}/bin/python -V|awk '{print $2}'|awk -F '.' '{print $1$2}')
    local abi_tag=$(echo cp$abi_version-cp$abi_version)
    ln -s ${prefix} /opt/python/${abi_tag}
}


function build_cpython {
    local py_ver=$1
    check_var $py_ver
    check_var $PYTHON_DOWNLOAD_URL
    wget -q $PYTHON_DOWNLOAD_URL/$py_ver/Python-$py_ver.tgz
    if [ $(lex_pyver $py_ver) -lt $(lex_pyver 3.3) ]; then
        # NOTE We only need wide unicode for nupic.bindings wheel
        do_cpython_build $py_ver ucs2
        do_cpython_build $py_ver ucs4
    else
        do_cpython_build $py_ver none
    fi
    rm -f Python-$py_ver.tgz
}


function build_cpythons {
    for py_ver in $@; do
        check_var $GET_PIP_URL
        curl -sLO $GET_PIP_URL
        build_cpython $py_ver
    done
    rm -f get-pip.py
    rm -f ez_setup.py
}


function do_openssl_build {
    ./config -fPIC --prefix=/usr/local/ssl > /dev/null
    make > /dev/null
    make install > /dev/null
    ln -sf /usr/lib64/libcrypto.so.1.1 /usr/local/ssl/lib/libcrypto.so.1.1
}


function check_sha256sum {
    local fname=$1
    check_var ${fname}
    local sha256=$2
    check_var ${sha256}

    echo "${sha256}  ${fname}" > ${fname}.sha256
    sha256sum -c ${fname}.sha256
    rm ${fname}.sha256
}


function build_openssl {
    local openssl_fname=$1
    check_var ${openssl_fname}
    local openssl_sha256=$2
    check_var ${openssl_sha256}
    check_var ${OPENSSL_DOWNLOAD_URL}
    curl -sLO ${OPENSSL_DOWNLOAD_URL}/${openssl_fname}.tar.gz
    check_sha256sum ${openssl_fname}.tar.gz ${openssl_sha256}
    tar -xzf ${openssl_fname}.tar.gz
    (cd ${openssl_fname} && do_openssl_build)
    rm -rf ${openssl_fname} ${openssl_fname}.tar.gz
}


function do_curl_build {
    LIBS=-ldl ./configure --with-ssl --disable-shared > /dev/null
    make > /dev/null
    make install > /dev/null
    ln -s /usr/local/ssl/lib/libcrypto.so /usr/lib/libcrypto.so
    ln -s /usr/local/ssl/lib/libssl.so /usr/lib/libssl.so
    ln -s /usr/local/ssl/bin/openssl /usr/local/bin/openssl
}


function build_curl {
    local curl_fname=$1
    check_var ${curl_fname}
    local curl_sha256=$2
    check_var ${curl_sha256}
    check_var ${CURL_DOWNLOAD_URL}
    curl -sLO ${CURL_DOWNLOAD_URL}/${curl_fname}.tar.bz2
    check_sha256sum ${curl_fname}.tar.bz2 ${curl_sha256}
    tar -jxf ${curl_fname}.tar.bz2
    (cd ${curl_fname} && do_curl_build)
    rm -rf ${curl_fname} ${curl_fname}.tar.bz2
}


function do_standard_install {
    ./configure > /dev/null
    make > /dev/null
    make install > /dev/null
}


function build_autoconf {
    local autoconf_fname=$1
    check_var ${autoconf_fname}
    local autoconf_sha256=$2
    check_var ${autoconf_sha256}
    check_var ${AUTOCONF_DOWNLOAD_URL}
    curl -sLO ${AUTOCONF_DOWNLOAD_URL}/${autoconf_fname}.tar.gz
    check_sha256sum ${autoconf_fname}.tar.gz ${autoconf_sha256}
    tar -zxf ${autoconf_fname}.tar.gz
    (cd ${autoconf_fname} && do_standard_install)
    rm -rf ${autoconf_fname} ${autoconf_fname}.tar.gz
}
