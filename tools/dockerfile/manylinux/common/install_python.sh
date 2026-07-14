# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

    #if [ $1 -eq '3.13.0t' ];then
    #    GIL='--disable-gil'
    #fi

    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} CFLAGS="-Wformat" LDFLAGS="-Wl,-rpath,${prefix}/lib" ./configure --prefix=${prefix} --enable-shared $unicode_flags > /dev/null
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} make -j8 > /dev/null
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} make install > /dev/null

    popd
    echo "ZZZ looking for libpython"
    find / -name 'libpython*.so*'
    rm -rf Python-$py_ver
    # Some python's install as bin/python3. Make them available as bin/python.
    if [ -e ${prefix}/bin/python3.10 ]; then
        ln -s python3.10 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.11 ]; then
        ln -s python3.11 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.12 ]; then
        ln -s python3.12 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.13 ]; then
        ln -s python3.13 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.13t ]; then
        ln -s python3.13t ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.14 ]; then
        ln -s python3.14 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.14t ]; then
        ln -s python3.14t ${prefix}/bin/python
    fi

    # NOTE Make libpython shared library visible to python calls below
    LD_LIBRARY_PATH="/usr/local/ssl/lib:${prefix}/lib" ${prefix}/bin/python -m pip config set global.trusted-host mirrors.aliyun.com
    LD_LIBRARY_PATH="/usr/local/ssl/lib:${prefix}/lib" ${prefix}/bin/python -m pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
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
    do_cpython_build $py_ver none
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

PYTHON_DOWNLOAD_URL=https://www.python.org/ftp/python
GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py
CPYTHON_VERSIONS="3.14.0 3.13.0 3.12.0 3.11.0 3.10.0"

mkdir -p /opt/python
build_cpythons $CPYTHON_VERSIONS
