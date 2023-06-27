# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import os
cinndir = os.path.dirname(os.path.abspath(__file__))
runtime_include_dir = os.path.join(cinndir, "libs")
cuhfile = os.path.join(runtime_include_dir, "cinn_cuda_runtime_source.cuh")

if os.path.exists(cuhfile):
    os.environ.setdefault('runtime_include_dir', runtime_include_dir)

from .core_api.common import *
from .core_api.backends import *
from .core_api.poly import *
from .core_api.ir import *
from .core_api.lang import *
from .version import full_version as __version__
