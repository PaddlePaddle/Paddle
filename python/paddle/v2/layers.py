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

import paddle.trainer_config_helpers as conf_helps
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config as __parse__
from paddle.trainer_config_helpers.default_decorators import wrap_name_default


class Layer(object):
    def __init__(self, name, parent_layer):
        assert isinstance(parent_layer, dict)
        assert isinstance(name, basestring)
        self.name = name
        self.__parent_layer__ = parent_layer

    def to_proto(self, context):
        """
        function to set proto attribute
        """
        kwargs = dict()
        for param_name in self.__parent_layer__:
            param_value = self.__parent_layer__[param_name].to_proto(
                context=context)
            kwargs[param_name] = param_value

        if self.name not in context:
            context[self.name] = self.to_proto_impl(**kwargs)
        return context[self.name]

    def to_proto_impl(self, **kwargs):
        raise NotImplementedError()


def parse_network(*outputs):
    def __real_func__():
        context = dict()
        real_output = [each.to_proto(context=context) for each in outputs]
        conf_helps.outputs(real_output)

    return __parse__(__real_func__)


def __convert__(method_name, name_prefix, parent_names):
    if name_prefix is not None:
        wrapper = wrap_name_default(name_prefix=name_prefix)
    else:
        wrapper = None

    class __Impl__(Layer):
        def __init__(self, name=None, **kwargs):
            parent_layers = dict()
            other_kwargs = dict()
            for pname in parent_names:
                parent_layers[pname] = kwargs[pname]

            for key in kwargs.keys():
                if key not in parent_names:
                    other_kwargs[key] = kwargs[key]

            super(__Impl__, self).__init__(name, parent_layers)
            self.__other_kwargs__ = other_kwargs

        if wrapper is not None:
            __init__ = wrapper(__init__)

        def to_proto_impl(self, **kwargs):
            args = dict()
            for each in kwargs:
                args[each] = kwargs[each]
            for each in self.__other_kwargs__:
                args[each] = self.__other_kwargs__[each]
            return getattr(conf_helps, method_name)(name=self.name, **args)

    return __Impl__


data_layer = __convert__('data_layer', None, [])
fc_layer = __convert__('fc_layer', name_prefix='fc', parent_names=['input'])
classification_cost = __convert__(
    'classification_cost',
    name_prefix='classification_cost',
    parent_names=['input', 'label'])

__all__ = ['data_layer', 'fc_layer', 'classification_cost', 'parse_network']

if __name__ == '__main__':
    data = data_layer(name='pixel', size=784)
    hidden = fc_layer(input=data, size=100, act=conf_helps.SigmoidActivation())
    predict = fc_layer(
        input=hidden, size=10, act=conf_helps.SoftmaxActivation())
    cost = classification_cost(
        input=predict, label=data_layer(
            name='label', size=10))
    print parse_network(cost)
