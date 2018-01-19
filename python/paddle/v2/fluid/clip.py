#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import layers
import framework
from . import core

__all__ = [
    'GradientClipByValue',
    'ErrorClipByValue',
    'append_gradient_clip_ops',
    'error_clip_callback',
]


class BaseErrorClipAttr(object):
    def append_clip_op(self, block, grad_name):
        raise NotImplementedError()


class ErrorClipByValue(BaseErrorClipAttr):
    def __init__(self, max, min=None):
        max = float(max)
        if min is None:
            min = -max
        else:
            min = float(min)
        self.max = max
        self.min = min

    def append_clip_op(self, block, grad_name):
        clip_op_desc = block.desc.append_op()
        clip_op_desc.set_type("clip")
        clip_op_desc.set_input("X", [grad_name])
        clip_op_desc.set_output("Out", [grad_name])
        clip_op_desc.set_attr("min", self.min)
        clip_op_desc.set_attr("max", self.max)


def error_clip_callback(block, context):
    # the context is a grad_to_var map
    grad_to_var = context
    op_desc = block.desc.op(block.desc.op_size() - 1)
    for grad_n in filter(lambda n: grad_to_var.has_key(n),
                         op_desc.output_arg_names()):
        fwd_var = block.var_recursive(grad_to_var[grad_n])
        error_clip = getattr(fwd_var, "error_clip", None)
        if not (error_clip is None or isinstance(error_clip,
                                                 BaseErrorClipAttr)):
            raise TypeError(
                "Variable's error_clip should be an instance of BaseErrorClipAttr or None."
            )
        if error_clip is not None:
            error_clip.append_clip_op(block, grad_n)


class BaseGradientClipAttr(object):
    def process_context(self, context, param, grad):
        raise NotImplementedError()

    def create_operators(self, param, grad):
        raise NotImplementedError()


class NullGradientClipAttr(BaseGradientClipAttr):
    def process_context(self, context, param, grad):
        pass

    def create_operators(self, param, grad):
        return param, grad


class GradientClipByValue(BaseGradientClipAttr):
    def __init__(self, max, min=None):
        max = float(max)
        if min is None:
            min = -max
        else:
            min = float(min)
        self.max = max
        self.min = min

    def process_context(self, context, param, grad):
        pass

    def create_operators(self, param, grad):
        new_grad = layers.clip(x=grad, min=self.min, max=self.max)
        return param, new_grad


class GradientClipByNorm(BaseGradientClipAttr):
    def __init__(self, clip_norm):
        self.clip_norm = clip_norm

    def process_context(self, context, param, grad):
        pass

    def create_operators(self, param, grad):
        new_grad = layers.clip_by_norm(x=grad, max_norm=self.clip_norm)
        return param, new_grad


class GradientClipByGlobalNorm(BaseGradientClipAttr):
    global_norm_var = None
    local_norm_var = None
    clip_norm_var = None
    scale_var = None

    @classmethod
    def init(cls, clip_norm):
        if not (isinstance(clip_norm, int) or isinstance(clip_norm, float)):
            raise TypeError("The 'clip_norm' must be a value of int or float")

        cls.global_norm_var = layers.fill_constant(
            shape=[1], dtype="float32", value=0.0)
        cls.local_norm_var = framework.default_main_program().block(
            0).create_var(
                name=framework.unique_name("local_norm"),
                dtype="float32",
                persistable=False)
        cls.clip_norm_var = layers.fill_constant(
            shape=[1], dtype="float32", value=clip_norm)

    @classmethod
    def check_init(cls):
        if not (isinstance(cls.global_norm_var, framework.Variable) and
                isinstance(cls.local_norm_var, framework.Variable) and
                isinstance(cls.clip_norm_var, framework.Variable)):
            raise ValueError(
                "Class 'GradientClipByGlobalNorm' has not been properly initialized. \
                 Please call GradientClipByGlobalNorm.init() first.")

    def process_context(self, context, param, grad):
        cls = self.__class__
        cls.check_init()

        cls.local_norm_var = layers.reduce_sum(
            input=layers.pow(x=grad, factor=2.0))
        layers.sums(
            input=[cls.local_norm_var, cls.global_norm_var],
            out=[cls.global_norm_var])

    def create_operators(self, param, grad):
        cls = self.__class__
        cls.check_init()

        if cls.scale_var is None:
            layers.sqrt(x=cls.global_norm_var, out=cls.global_norm_var)
            cls.scale_var = layers.elementwise_div(
                x=cls.clip_norm_var,
                y=layers.elementwise_max(
                    x=cls.clip_norm_var, y=cls.global_norm_var))
            assert cls.scale_var.shape == (1L, )

        new_grad = layers.elementwise_mul(x=grad, y=cls.scale_var)
        return param, new_grad


def gradient_clip_by_global_norm(clip_norm, param_list=None, program=None):
    if program is None:
        program = framework.default_main_program()
    if param_list is None:
        param_list = program.block(0).all_parameters()
    if all(isinstance(elem, basestring) for elem in param_list):
        param_list = [program.block(0).var(elem) for elem in param_list]
    if not all(isinstance(elem, framework.Parameter) for elem in param_list):
        raise TypeError(
            "'param_list' should be a list of Parameter or basestring(parameter's name)."
        )

    GradientClipByGlobalNorm.init(clip_norm)
    for param in param_list:
        param.gradient_clip_attr = GradientClipByGlobalNorm()


def append_gradient_clip_ops(param_grad):
    context = dict()
    create_op_callbacks = []
    for p, g in param_grad:
        clip_attr = getattr(p, 'gradient_clip_attr', NullGradientClipAttr())
        if clip_attr is None:
            clip_attr = NullGradientClipAttr()
        if not isinstance(clip_attr, BaseGradientClipAttr):
            raise TypeError(
                "clip attribute should be an instance of BaseGradientClipAttr")

        clip_attr.process_context(context=context, param=p, grad=g)
        create_op_callbacks.append(
            functools.partial(
                clip_attr.create_operators, param=p, grad=g))

    return [each_callback() for each_callback in create_op_callbacks]


ClipByValue = GradientClipByValue
