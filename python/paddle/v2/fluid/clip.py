import functools
import layers
from framework import Variable
from . import core

__all__ = [
    'GradientClipByValue', 'append_gradient_clip_ops', 'error_clip_callback'
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
        block.append_op(
            type="clip",
            inputs={"X": grad_name},
            outputs={"Out": grad_name},
            attrs={"min": self.min,
                   "max": self.max})


def error_clip_callback(block, context):
    # the context is a grad_to_var map
    grad_to_var = context
    op_desc = block.desc.op(block.desc.op_size() - 1)
    for grad_n in filter(lambda n: grad_to_var.has_key(n),
                         op_desc.output_arg_names()):
        fwd_var = block.var_recursive(grad_to_var[grad_n])
        error_clip = getattr(fwd_var, "error_clip", None)
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
    clip_norm_var = None
    ratio_var = None

    @classmethod
    def init(cls, clip_norm):
        cls.global_norm_var = layers.fill_constant(
            shape=[1], dtype="float32", value=0.0)
        cls.clip_norm_var = layers.fill_constant(
            shape=[1], dtype="float32", value=clip_norm)

    def __init__(self):
        if not (isinstance(self.__class__.global_norm_var, Variable) and
                isinstance(self.__class__.clip_norm_var, Variable)):
            raise ValueError(
                "Class 'GradientClipByGlobalNorm' has not been properly initialized. Please call GradientClipByGlobalNorm.init() first."
            )

    def process_context(self, context, param, grad):
        local_norm_var = layers.reduce_sum(
            x=layers.pow(x=grad, factor=2), reduce_all=True)
        layers.sums(
            input=[local_norm_var, self.__class__.global_norm_var],
            out=[self.__class__.global_norm_var])

    def create_operators(self, param, grad):
        if self.__class__.ratio_var is None:
            self.__class__.global_norm_var = layers.sqrt(
                x=self.__class__.global_norm_var)
            self.__class__.ratio_var = layers.elementwise_div(
                x=self.__class__.clip_norm_var,
                y=layers.elementwise_max(
                    x=self.__class__.clip_norm_var,
                    y=self.__class__.global_norm_var))
        # 缺乏elementwise_max
        # 没法将ratio_var送给scale_op。
        # new_grad = layers.


def append_gradient_clip_ops(param_grad):
    context = dict()
    create_op_callbacks = []
    for p, g in param_grad:
        clip_attr = getattr(p, 'clip_attr', NullGradientClipAttr())
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
