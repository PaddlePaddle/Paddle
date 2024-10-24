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
if [ "$1" == "python313" ]; then
    set -ex
    wget https://github.com/python/cpython/archive/refs/tags/v3.13.0.tar.gz
    tar -xvf v3.13.0.tar.gz
    cd cpython-3.13.0/
    ./configure --with-pydebug --enable-optimizations --with-lto --disable-gil
    make -s -j
    make altinstall
    if [ -f "/usr/local/bin/python3.13t" ]; then
        rm /usr/local/bin/python3.13t
    fi
    ln -s /usr/local/bin/python3.13td /usr/local/bin/python3.13t
    ./configure --with-pydebug
    make -s -j
    make altinstall
fi
