import functools
import layers
from . import core

__all__ = ['GradientClipByValue', 'append_gradient_clip_ops']


class BaseErrorClipAttr(object):
    def create_clip_op_desc(self, grad_name):
        raise NotImplementedError()

    def prepend_clip_op_desc(self, op_descs):
        grad_names = set()
        for op_desc in op_descs:
            grad_names.update(
                filter(lambda n: n.find(core.grad_var_suffix()) != -1,
                       op_desc.output_arg_names()))
        for n in grad_names:
            op_descs.append(self.create_clip_op_desc(grad_name=n))


class ErrorClipByValue(BaseErrorClipAttr):
    def __init__(self, max, min=None):
        max = float(max)
        if min is None:
            min = -max
        else:
            min = float(min)
        self.max = max
        self.min = min

    def create_clip_op_desc(self, grad_name):
        desc = core.OpDesc()
        desc.set_type("clip")
        desc.set_input("X", grad_name)
        desc.set_output("Out", grad_name)
        desc.set_attr("min", self.min)
        desc.set_attr("max", self.max)
        return desc


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
