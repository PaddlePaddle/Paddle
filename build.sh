# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

export PYTHONPATH=/opt/_internal/cpython-3.7.0/lib/python3.7/:${PATH}
cmake .. -DWITH_MKL=OFF \
     -DCMAKE_BUILD_TYPE=Release \
     -DWITH_PYTHON=ON \
     -DWITH_DISTRIBUTE=ON \
     -DWITH_TESTING=OFF \
     -DWITH_BRPC=ON \
     -DPY_VERSION=3.7 \
     -DPYTHON_INCLUDE_DIR=/opt/_internal/cpython-3.7.0/include/python3.7m \
     -DPYTHON_LIBRARY=/opt/_internal/cpython-3.7.0/lib/libpython3.7m.so \
     -DPYTHON_EXECUTABLE=/opt/_internal/cpython-3.7.0/bin/python \
     -DWITH_PSLIB=ON \
     -DWITH_HETERPS=ON \
     -DWITH_GLOO=ON \
     -DWITH_GPU=ON \
     -DWITH_FLUID_ONLY=ON \
     -DWITH_PSLIB_BRPC=OFF \
     -DWITH_PSCORE=OFF && make -j30
