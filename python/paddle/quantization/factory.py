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
from paddle.nn import Layer
from .quanter import BaseQuanter
from .observer import BaseObserver
from typing import Union

__all__ = ["ObserverFactory", "QuanterFactory", "quanter"]


@six.add_metaclass(abc.ABCMeta)
class ClassWithArguments(object):
    def __init__(self, **args):
        self._args = args

    @property
    def args(self):
        return self._args

    @abc.abstractmethod
    def get_class(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.args})"

    def __repr__(self):
        return self.__str__()


class ObserverFactory(ClassWithArguments):
    def __init__(self, **args):
        super(ObserverFactory, self).__init__(**args)

    def instance(self, layer: Layer) -> Union[BaseObserver, BaseQuanter]:
        return self.get_class()(layer, **self._args)


class QuanterFactory(ObserverFactory):
    def __init__(self, **args):
        super(QuanterFactory, self).__init__(**args)

    def get_class(self):
        return self.cls


def quanter(class_name, **kwargs):
    def wrapper(target_class):
        formal_kwargs_str = ",".join([f"{k}={v}" for k, v in kwargs.items()])
        actual_kwargs_str = ",".join([f"{k}={k}" for k, v in kwargs.items()])
        print(f"formal_kwargs_str: {formal_kwargs_str}")
        print(f"actual_kwargs_str: {actual_kwargs_str}")
        init_function_str = f"""
def init_function(self, {formal_kwargs_str}):
    super(type(self), self).__init__({actual_kwargs_str})
locals()["init_function"]=init_function"""
        print(f"init_function_str: {init_function_str}")
        exec(init_function_str)
        new_class = type(
            class_name,
            (QuanterFactory,),
            {
                "__init__": locals()["init_function"],
                "cls": target_class,
            },
        )
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        setattr(mod, class_name, new_class)
        if "__all__" in mod.__dict__:
            mod.__all__.append(class_name)
        return target_class

    return wrapper
