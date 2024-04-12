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
from paddle import cinn
from paddle.cinn import framework
from paddle.cinn.backends import Compiler


class Module:
    def __init__(self, llir_module, target, fn_name, arg_names):
        self.arg_names = arg_names
        self.fn_name = fn_name
        self.compiler = Compiler.create(target)
        self.compiler.build(llir_module)
        self._instruction = framework.Instruction(
            target, None, [], arg_names, fn_name
        )

    def __call__(self, *args):
        name2pod = {}
        for i, name in enumerate(self.arg_names):
            if isinstance(args[i], cinn.runtime.data_array.DataArray):
                name2pod[name] = cinn.runtime.cinn_pod_value_t(args[i].data)
            else:
                name2pod[name] = cinn.runtime.cinn_pod_value_t(args[i])

        self._instruction.run(self.compiler, self.fn_name, name2pod)
