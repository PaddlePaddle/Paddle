# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name="record_time_ops",
    ext_modules=CUDAExtension(
        sources=[
            "./record_time.cu",
        ]
    ),
)

#  python setup_cuda.py install 安装即可
#  from record_time_ops import record_time 即可使用
# a = record_time(a, "点云开始时候打点")
# a 是输入，这个函数只把输入拷贝给输出，不进行任何计算，同时打点此时的时刻，单位是ms，毫秒

