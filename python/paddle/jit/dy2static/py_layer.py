import functools

from paddle.autograd.py_layer import PyLayerMeta
from .program_translator import (
    CONVERSION_OPTIONS,
    StaticFunction,
    convert_to_static,
    unwrap_decorators,
)

def is_pylayer_func(func):
    """predict whether a function is from PyLayer.
    """
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

        self.forward_fn_with_ctx = functools.partial(convert_to_static(self.orig_forward_fn), self.static_pylayer_context)
        self.backward_fn_with_ctx = functools.partial(convert_to_static(self.orig_backward_fn), self.static_pylayer_context)


    def __call__(self, *args, **kwargs):
        return self.forward_fn_with_ctx(*args, **kwargs)