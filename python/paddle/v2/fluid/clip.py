import functools
import layers
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
    def process_context(self, context, p_g):
        raise NotImplementedError()

    def create_operators(self, param, grad):
        raise NotImplementedError()


class NullGradientClipAttr(BaseGradientClipAttr):
    def process_context(self, context, p_g):
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

    def process_context(self, context, p_g):
        pass

    def create_operators(self, param, grad):
        new_grad = layers.clip(x=grad, min=self.min, max=self.max)
        return param, new_grad


def append_gradient_clip_ops(param_grad):
    context = dict()
    create_op_callbacks = []
    for p, g in param_grad:
        clip_attr = getattr(p, 'clip_attr', NullGradientClipAttr())
        if clip_attr is None:
            clip_attr = NullGradientClipAttr()
        if not isinstance(clip_attr, BaseGradientClipAttr):
            raise TypeError(
                "clip attribute should be an instance of BaseGradientClippingAttr"
            )

        clip_attr.process_context(context=context, p_g=param_grad)
        create_op_callbacks.append(
            functools.partial(
                clip_attr.create_operators, param=p, grad=g))

    return [each_callback() for each_callback in create_op_callbacks]


ClipByValue = GradientClipByValue
