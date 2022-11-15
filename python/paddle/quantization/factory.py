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

import six
import abc
import inspect
from typing import Union
from paddle.nn import Layer
from functools import partial
from .quanter import BaseQuanter
from .observer import BaseObserver


__all__ = ["ObserverFactory", "QuanterFactory", "quanter"]


@six.add_metaclass(abc.ABCMeta)
class ClassWithArguments(object):
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
    def get_class(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.args}, {self.kwargs})"

    def __repr__(self):
        return self.__str__()


class ObserverFactory(ClassWithArguments):
    def __init__(self, *args, **kwargs):
        super(ObserverFactory, self).__init__(*args, **kwargs)
        self.partial_class = None

    def instance(self, layer: Layer) -> Union[BaseObserver, BaseQuanter]:
        if self.partial_class is None:
            self.partial_class = partial(
                self.get_class(), *self.args, **self.kwargs
            )
        return self.partial_class(layer)


class QuanterFactory(ObserverFactory):
    def __init__(self, *args, **kwargs):
        super(QuanterFactory, self).__init__(*args, **kwargs)


def quanter(class_name):
    r"""
    Annotation to create a factory class for quanter.

    Args:
        class_name (str) - The name of factory class to be declared.
    """

    def wrapper(target_class):
        init_function_str = f"""
def init_function(self, *args, **kwargs):
    super(type(self), self).__init__(*args, **kwargs)
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
                "get_class": locals()["get_class_function"],
            },
        )
        setattr(mod, class_name, new_class)
        if "__all__" in mod.__dict__:
            mod.__all__.append(class_name)

        return target_class

    return wrapper
