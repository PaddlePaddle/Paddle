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

import inspect

import paddle


def is_inplace_api(func):
    inplace_apis = {paddle.static.setitem}
    return func in inplace_apis


def get_tensor_methods():
    return [
        member_name
        for member_name, member in inspect.getmembers(paddle.static.Variable)
        if inspect.isfunction(member)
    ]


def get_paddle_api():
    modules = [
        paddle,
        paddle.nn.functional,
        paddle.incubate.nn.functional,
        paddle.linalg,
        paddle.signal,
        paddle.fft,
        paddle.vision.ops,
        paddle.metric,
    ]
    special_paddle_apis = [paddle.tensor.fill_constant]
    non_operator_related_apis = [
        paddle.in_dynamic_mode,
        paddle.save,
        paddle.load,
        paddle.get_cuda_rng_state,
        paddle.set_rng_state,
        paddle.set_cuda_rng_state,
        paddle.get_rng_state,
        paddle.set_default_dtype,
        paddle.check_shape,
        paddle.summary,
        paddle.finfo,
        paddle.iinfo,
        paddle.enable_static,
        paddle.disable_static,
        paddle.is_grad_enabled,
    ]
    # TODO: users should not call static_apis, but we need to use, so add static_apis here temporary
    static_apis = [paddle.static.setitem, paddle.static.accuracy]
    paddle_api_list = []
    for module in modules:
        for fn_name in getattr(module, "__all__", []):
            fn = getattr(module, fn_name)
            if inspect.isfunction(fn):
                paddle_api_list.append(fn)
    return list(
        set(special_paddle_apis)
        | set(static_apis)
        | set(paddle_api_list) - set(non_operator_related_apis)
    )


paddle_tensor_methods = get_tensor_methods()
paddle_api_list = get_paddle_api()

# TODO(Aurelius84): It seems that we use it to judge 'in_paddle_module()'.
# Bug what does 'is_paddle_module' really means? Is all paddle.xx sub module
# considered as paddle module？
paddle_api_module_prefix = {
    "paddle.nn.functional",
}

break_graph_set = set()


break_graph_tensor_method = {
    'register_hook',
    'numpy',
    'clear_gradient',
    # TODO: Browse all possible functions and make prior judgments.
}

not_supported_paddle_layer = {paddle.nn.RNN}


def is_not_supported_paddle_layer(layer_class):
    return layer_class in not_supported_paddle_layer


def is_break_graph_tensor_methods(method_name):
    return method_name in break_graph_tensor_method


def add_break_graph_apis(apis: list):
    break_graph_set.update(apis)
