# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

from paddle.fluid import core
from paddle.fluid.core import Load


class Layer:
    def __init__(self):
        self.cpp_layer = None
        # {name: Function}
        self.functions = {}

    def load(self, load_path, place):
        self.cpp_layer = Load(load_path, place)

        for name in self.cpp_layer.function_names():
            function = self.cpp_layer.function(name)
            info = self.cpp_layer.function_info(name)
            self.functions[name] = Function(function, info)
            setattr(self, name, self.functions[name])


class Function:
    def __init__(self, function, info):
        self.function = function
        self.info = FunctionInfo(info)

    def __call__(self, *args):
        return core.eager.jit_function_call(self.function, args)


class FunctionInfo:
    def __init__(self, info):
        self.info = info

    def name(self):
        return self.info.name()
