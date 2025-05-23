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

import importlib
import inspect

from paddle.incubate.cc.py_code_gen.traits.pir_trait import PirTrait


def GetClasses(filepath):
    spec = importlib.util.spec_from_file_location(
        "pir_py_code_module", filepath
    )
    pir_py_code_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pir_py_code_module)
    yield from inspect.getmembers(pir_py_code_module, inspect.isclass)


def GetProgramClasses(filepath):
    for name, cls in GetClasses(filepath):
        yield type(name, (cls, PirTrait), {})
