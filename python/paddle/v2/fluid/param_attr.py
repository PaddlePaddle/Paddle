from initializer import Initializer, Xavier, Constant
from regularizer import WeightDecayRegularizer


class ParamAttr(object):
    def __init__(self,
                 name=None,
                 initializer=None,
                 learning_rate=1.0,
                 regularizer=None,
                 trainable=True):
        self.name = name
        self.initializer = initializer
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.trainable = trainable

    def set_default_initializer(self, initializer):
        if initializer is None:
            if self.initializer is None:
                raise ValueError("ParamAttr.initializer is not set")
            return

        if self.initializer is not None:
            return

        self.initializer = initializer

    def set_default_param_initializer(self):
        self.set_default_initializer(Xavier())

    def set_default_bias_initializer(self):
        self.set_default_initializer(Constant(0.0))

    @staticmethod
    def to_attr(arg):
        if arg is None:
            return ParamAttr()
        elif isinstance(arg, ParamAttr):
            return arg
        elif isinstance(arg, str) or isinstance(arg, unicode):
            return ParamAttr(name=arg)
        elif isinstance(arg, Initializer):
            return ParamAttr(initializer=arg)
        elif isinstance(arg, WeightDecayRegularizer):
            return ParamAttr(regularizer=arg)
        elif isinstance(arg, bool):
            return ParamAttr.to_attr(None) if arg else False
        else:
            raise TypeError("{0} cast to ParamAttr".format(type(arg)))

    def to_kwargs(self, with_initializer=False):
        kwargs = {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'regularizer': self.regularizer,
            'trainable': self.trainable
        }
        if with_initializer:
            kwargs['initializer'] = self.initializer
        return kwargs
