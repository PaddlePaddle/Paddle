# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import functools


def print_args(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        function_name = func.__name__
        from paddle.jit.dy2static.utils import parse_arg_and_kwargs

        api_params, api_defaults = parse_arg_and_kwargs(func)
        import collections

        inputs = collections.OrderedDict()
        params = collections.OrderedDict()
        from paddle import Tensor

        # from paddle import Tensor
        for i in range(len(args)):
            if isinstance(args[i], Tensor):
                inputs[api_params[i]] = args[i]
            elif (
                isinstance(args[i], (list, tuple))
                and len(args[i]) > 0
                and isinstance(args[i][0], Tensor)
            ):
                inputs[api_params[i]] = args[i]
            else:
                params[api_params[i]] = args[i]
        for key, value in kwargs.items():
            if type(value) == Tensor:
                inputs[key] = value
            else:
                params[key] = value
        log_msg = f"{{function_name : {function_name}, "
        log_msg += "inputs: { "
        for name, value in inputs.items():
            shape = []
            if (
                isinstance(value, (list, tuple))
                and len(value) > 0
                and isinstance(value[0], Tensor)
            ):
                for v in value:
                    shape.append(v.shape)
                input_type = "List(Tensor)"
            else:
                shape = value.shape
                input_type = str(type(value))
            log_msg += f"{{ {name}, type: {input_type}, shape: {shape} }}, "
        log_msg += "}, "
        log_msg += "params: [ "
        for name, value in params.items():
            log_msg += f"{name}: {str(value)}, "
        log_msg += "]}"
        print(log_msg, flush=True)
        return func(*args, **kwargs)

    return inner