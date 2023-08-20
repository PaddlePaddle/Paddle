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
from paddle.static.nn import static_pylayer

from .program_translator import convert_to_static, unwrap_decorators


def is_pylayer_func(func):
    """predict whether a function is from PyLayer."""
    func_self = getattr(func, '__self__', None)
    if func_self and isinstance(func_self, PyLayerMeta):
        return True
    return False


class StaticPyLayerContext:
    def save_for_backward(self, *tensors):
        # insert one OP ?
        # self.container = tensors
        print("call save_for_backward")
        pass

    def saved_tensor(self):
        # insert one OP ?
        # return self.container
        print("call saved_tensor")
        pass

    def mark_not_inplace(self, *args):
        # insert one OP ?
        # self.not_inplace_tensors = args
        print("call mark_not_inplace")
        pass

    def mark_non_differentiable(self, *args):
        # insert one OP ?
        # self.non_differentiable = args
        print("call mark_non_differentiable")
        pass

    def set_materialize_grads(self, value: bool):
        # insert one OP ?
        # self.materialize_grads = value
        print("call set_materialize_grads")
        pass


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
    def apply(self, *args, **kwargs):
        return static_pylayer(
            forward_fn=self.forward_fn_with_ctx,
            inputs=list(args),
            backward_fn=self.backward_fn_with_ctx,
        )
