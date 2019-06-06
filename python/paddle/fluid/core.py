# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import sys
from x86cpu import info as cpuinfo

load_noavx = False
if cpuinfo.supports_avx:
    try:
        from .core_avx import *
    except ImportError as error:
        print(
            'WARNING: AVX is supported on local machine, you could build paddle '
            'WITH_AVX=ON to get better performance.\n' +
            error.__class__.__name__ + ": " + error.message)
        load_noavx = True
else:
    load_noavx = True

if load_noavx:
    try:
        from .core_avx import *
    except ImportError as error:
        sys.exit("Error: Can not load core_noavx.* \n" +
                 error.__class__.__name__ + ": " + error.message)
        load_noavx = True
