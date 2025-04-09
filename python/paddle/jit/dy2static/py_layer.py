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
import inspect
import textwrap

from paddle import pir
from paddle.base.framework import Variable, in_pir_mode
from paddle.base.libpaddle.pir import build_pipe_for_pylayer
from paddle.common_ops_import import LayerHelper
from paddle.static.nn import static_pylayer
from paddle.utils import flatten, pack_sequence_as

from .program_translator import convert_to_static, unwrap_decorators


class StaticPyLayerContext:
    def __init__(self):
        self.saved_vars = []
        self.saved_vars_structure = None

        if in_pir_mode():
            self.tuple_push_op_name = "cf.tuple_push"
            self.tuple_pop_op_name = "cf.tuple_pop"

    def __setattr__(self, attr: str, value: object):
        attr_allow_list = ["saved_vars", "saved_vars_structure"]
        if (
            in_pir_mode()
            and attr not in attr_allow_list
            and isinstance(value, pir.Value)
        ):
            raise AttributeError(
                textwrap.dedent(
                    f"""\
                `ctx.{attr} = tensor` is not allowed in static mode, please use `ctx.save_for_backward(tensor)` instead.

                For example:

                    >>> class ExamplePyLayer(PyLayer):
                    ...     @staticmethod
                    ...     def forward(ctx, x):
                    ...         # ctx.x = x  # This is not allowed in static mode, Replace it with `ctx.save_for_backward(x)`
                    ...         ctx.save_for_backward(x)
                    ...         x1 = paddle.tanh(x)
                    ...         return x1

                    ...     @staticmethod
                    ...     def backward(ctx, grad):
                    ...         # x = ctx.x  # Same as above, replace it with `x, = ctx.saved_tensor()`
                    ...         x, = ctx.saved_tensor()
                    ...         x_grad = grad * (1 - paddle.square(x))
                    ...         return x_grad
                """
                )
            )
        super().__setattr__(attr, value)

    def save_for_backward(self, *tensors):
        if in_pir_mode():
            self.saved_vars_structure = tensors
            flatten_tensors = flatten(tensors)
            tensor_elements = list(
                filter(lambda x: isinstance(x, pir.Value), flatten_tensors)
            )
            current_insert_point = pir.get_current_insertion_point()
            current_block = current_insert_point.block()
            build_pipe_for_pylayer(current_block, tensor_elements)
        else:
            for tensor in tensors:
                assert isinstance(tensor, Variable)
                self.saved_vars.append(tensor)

    def saved_tensor(self):
        if in_pir_mode():
            current_insert_point = pir.get_current_insertion_point()
            current_block = current_insert_point.block()
            out_list = []
            for op in current_block.ops:
                if op.name() == self.tuple_pop_op_name:
                    out_list = op.as_tuple_pop_op().pop_all_values()
            if self.saved_vars_structure is not None:
                flattened_structure = flatten(self.saved_vars_structure)
                value_cursor = 0
                for i, tensor in enumerate(flattened_structure):
                    if isinstance(tensor, pir.Value):
                        flattened_structure[i] = out_list[value_cursor]
                        value_cursor += 1
                out_list = pack_sequence_as(
                    self.saved_vars_structure, flattened_structure
                )
        else:
            helper = LayerHelper("StaticPyLayerContext")
            out_list = []
            for saved_var in self.saved_vars:
                out = helper.create_variable(
                    name=saved_var.name,
                    dtype=saved_var.dtype,
                    shape=saved_var.shape,
                    type=saved_var.type,
                )
                out_list.append(out)

        return out_list

    # TODO(MarioLulab): support not_inplace
    def mark_not_inplace(self, *args):
        raise NotImplementedError

    # TODO(MarioLulab): support non_differentiable
    def mark_non_differentiable(self, *args):
        raise NotImplementedError

    # TODO(MarioLulab): support materialize_grads
    def set_materialize_grads(self, value: bool):
        raise NotImplementedError


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
        # rearrange `position-args + keyword-args` into `position-args`
        dyfunc_sig = inspect.signature(self.dyfunc_self.forward)
        bound_args = dyfunc_sig.bind(self.dyfunc_self, *args, **kwargs)
        bound_args.apply_defaults()
        input_args = [
            item
            for i, item in enumerate(bound_args.arguments.values())
            if i > 0
        ]  # index 0 indicate `dyfunc_self` which shouldn't be put into `input_args`

        return static_pylayer(
            forward_fn=self.forward_fn_with_ctx,
            inputs=input_args,
            backward_fn=self.backward_fn_with_ctx,
        )
