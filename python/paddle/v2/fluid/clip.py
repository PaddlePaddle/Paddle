import functools
import layers

__all__ = ['GradientClipByValue', 'append_gradient_clip_ops']


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
