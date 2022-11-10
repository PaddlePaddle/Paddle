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
from paddle.nn import Layer
from .quanter import BaseQuanter
from .observer import BaseObserver
from typing import Union

__all__ = ["ObserverFactory", "QuanterFactory"]


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
