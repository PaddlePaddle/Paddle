# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


import abc
import inspect
from functools import partial

from paddle.nn import Layer

from .base_quanter import BaseQuanter


class ClassWithArguments(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @abc.abstractmethod
    def _get_class(self):
        pass

    def __str__(self):
        args_str = ",".join(
            list(self.args) + [f"{k}={v}" for k, v in self.kwargs.items()]
        )
        return f"{self.__class__.__name__}({args_str})"

    def __repr__(self):
        return self.__str__()


class QuanterFactory(ClassWithArguments):
    r"""
    The factory holds the quanter's class information and
    the arguments used to create quanter instance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_class = None

    def _instance(self, layer: Layer) -> BaseQuanter:
        r"""
        Create an instance of quanter for target layer.
        """
        if self.partial_class is None:
            self.partial_class = partial(
                self._get_class(), *self.args, **self.kwargs
            )
        return self.partial_class(layer)


ObserverFactory = QuanterFactory


def quanter(class_name):
    r"""
    Annotation to declare a factory class for quanter.

    Args:
        class_name (str) - The name of factory class to be declared.

    Examples:
       .. code-block:: python

            # Given codes in ./customized_quanter.py
            from paddle.quantization import quanter
            from paddle.quantization import BaseQuanter
            @quanter("CustomizedQuanter")
            class CustomizedQuanterLayer(BaseQuanter):
                def __init__(self, arg1, kwarg1=None):
                    pass

            # Used in ./test.py
            # from .customized_quanter import CustomizedQuanter
            from paddle.quantization import QuantConfig
            arg1_value = "test"
            kwarg1_value = 20
            quanter = CustomizedQuanter(arg1_value, kwarg1=kwarg1_value)
            q_config = QuantConfig(activation=quanter, weight=quanter)

    """

    def wrapper(target_class):
        init_function_str = f"""
def init_function(self, *args, **kwargs):
    super(type(self), self).__init__(*args, **kwargs)
    import importlib
    module = importlib.import_module("{target_class.__module__}")
    my_class = getattr(module, "{target_class.__name__}")
    globals()["{target_class.__name__}"] = my_class
def get_class_function(self):
    return {target_class.__name__}
locals()["init_function"]=init_function
locals()["get_class_function"]=get_class_function
"""
        exec(init_function_str)
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        new_class = type(
            class_name,
            (QuanterFactory,),
            {
                "__init__": locals()["init_function"],
                "_get_class": locals()["get_class_function"],
            },
        )
        setattr(mod, class_name, new_class)
        if "__all__" in mod.__dict__:
            mod.__all__.append(class_name)

        return target_class

    return wrapper
