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

import functools

from paddle.autograd.py_layer import PyLayerMeta
from paddle.common_ops_import import LayerHelper
from paddle.fluid.framework import Variable
from paddle.static.nn import static_pylayer

from .program_translator import convert_to_static, unwrap_decorators


def is_pylayer_func(func):
    """predict whether a function is from PyLayer."""
    func_self = getattr(func, '__self__', None)
    if func_self and isinstance(func_self, PyLayerMeta):
        return True
    return False


class StaticPyLayerContext:
    def __init__(self):
        self.saved_vars = []
    
    def save_for_backward(self, *tensors):
        for tensor in tensors:
            assert isinstance(tensor, Variable)
            self.saved_vars.append(tensor)

    def saved_tensor(self):
        helper = LayerHelper("StaticPyLayerContext")
        out_list = []
        for saved_var in self.saved_vars:
            out = helper.create_variable(
                name=saved_var.name,
                dtype=saved_var.dtype,
                shape=saved_var.shape
            )
            out_list.append(out)
            
        return out_list
        

    def mark_not_inplace(self, *args):
        # self.not_inplace_tensors = args
        raise NotImplementedError()
    def mark_non_differentiable(self, *args):
        # self.non_differentiable = args
        raise NotImplementedError()

    def set_materialize_grads(self, value: bool):
        # self.materialize_grads = value
        raise NotImplementedError()


class StaticPyLayer:
    def __init__(self, dyfunc_self):
        self.dyfunc_self = dyfunc_self
        _, self.orig_forward_fn = unwrap_decorators(dyfunc_self.forward)
        _, self.orig_backward_fn = unwrap_decorators(dyfunc_self.backward)
        self.static_pylayer_context = StaticPyLayerContext()

        self.forward_fn_with_ctx = functools.partial(
            convert_to_static(self.orig_forward_fn), self.static_pylayer_context
        )
        self.backward_fn_with_ctx = functools.partial(
            convert_to_static(self.orig_backward_fn),
            self.static_pylayer_context,
        )

    # NOTE: only support position args and Variables Now
    def apply(self, *args):
        return static_pylayer(
            forward_fn=self.forward_fn_with_ctx,
            inputs=list(args),
            backward_fn=self.backward_fn_with_ctx,
        )
