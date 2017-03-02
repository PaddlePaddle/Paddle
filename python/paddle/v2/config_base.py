# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import collections

from paddle.trainer_config_helpers.default_decorators import wrap_name_default
import paddle.trainer_config_helpers as conf_helps


class Layer(object):
    def __init__(self, name=None, parent_layers=None):
        assert isinstance(parent_layers, dict)
        self.name = name
        self.__parent_layers__ = parent_layers
        self.__extra_parent__ = []

    def append_extra_parent(self, p):
        self.__extra_parent__.append(p)

    def to_proto(self, context):
        """
        function to set proto attribute
        """
        kwargs = dict()
        for each in self.__extra_parent__:
            each.to_proto(context=context)

        for layer_name in self.__parent_layers__:
            if not isinstance(self.__parent_layers__[layer_name],
                              collections.Sequence):
                v1_layer = self.__parent_layers__[layer_name].to_proto(
                    context=context)
            else:
                v1_layer = map(lambda x: x.to_proto(context=context),
                               self.__parent_layers__[layer_name])
            kwargs[layer_name] = v1_layer

        if self.name is None:
            return self.to_proto_impl_with_context(context=context, **kwargs)
        elif self.name not in context:
            context[self.name] = self.to_proto_impl_with_context(
                context=context, **kwargs)

        return context[self.name]

    def to_proto_impl_with_context(self, context, **kwargs):
        return self.to_proto_impl(**kwargs)

    def to_proto_impl(self, **kwargs):
        raise NotImplementedError()


def __convert_to_v2__(method_name, parent_names, is_default_name=True):
    if is_default_name:
        wrapper = wrap_name_default(name_prefix=method_name)
    else:
        wrapper = None

    class V2LayerImpl(Layer):
        def __init__(self, **kwargs):
            parent_layers = dict()
            other_kwargs = dict()
            for pname in parent_names:
                if kwargs.has_key(pname):
                    parent_layers[pname] = kwargs[pname]

            for key in kwargs.keys():
                if key not in parent_names:
                    other_kwargs[key] = kwargs[key]

            name = kwargs.get('name', None)
            super(V2LayerImpl, self).__init__(name, parent_layers)
            self.__other_kwargs__ = other_kwargs

        if wrapper is not None:
            __init__ = wrapper(__init__)

        def to_proto_impl(self, **kwargs):
            args = dict()
            for each in kwargs:
                args[each] = kwargs[each]
            for each in self.__other_kwargs__:
                args[each] = self.__other_kwargs__[each]
            return getattr(conf_helps, method_name)(**args)

    return V2LayerImpl
