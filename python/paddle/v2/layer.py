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
"""
`paddle.v2.layer` is a part of model config packages in paddle.v2. In API v2,
we want to make Paddle a plain Python package. The model config package defined
the way how to configure a neural network topology in Paddle Python code.

The primary usage shows below.

..  code-block:: python

    import paddle.v2 as paddle

    img = paddle.layer.data(name='img', type=paddle.data_type.dense_vector(784))
    hidden = paddle.layer.fc(input=img, size=200)
    prediction = paddle.layer.fc(input=hidden, size=10,
                                 act=paddle.activation.Softmax())

    # use prediction instance where needed.
    parameters = paddle.parameters.create(cost)
"""

import collections
import copy
import paddle.trainer_config_helpers.layers as v1_layers
import paddle.trainer.config_parser as cp
from paddle.proto.ModelConfig_pb2 import ModelConfig, SubModelConfig

__all__ = ['data', 'parse_network']
__layer_map__ = {}


def __wrap__(f):
    def wrapped(*args, **xargs):
        out = f(*args, **xargs)
        outs = out
        if not isinstance(out, collections.Sequence):
            outs = [out]
        for l in outs:
            if isinstance(l, v1_layers.LayerOutput):
                __layer_map__[l.full_name] = l
        return out

    return wrapped


def __need_to_keep__(name):
    if name in ['StaticInput', 'LayerType', 'layer_support']:
        return False
    return True


def __need_to_wrap__(name):
    return name not in ['AggregateLevel', 'ExpandLevel']


def __convert_name__(inname):
    if inname == 'maxid_layer':
        return 'max_id'
    elif inname.endswith('memory') or inname.endswith(
            '_seq') or inname.endswith('_sim') or inname == 'hsigmoid':
        return inname
    elif inname in [
            'cross_entropy', 'multi_binary_label_cross_entropy',
            'cross_entropy_with_selfnorm'
    ]:
        return inname + "_cost"
    elif inname.endswith('_cost'):
        return inname
    elif inname.endswith("_layer"):
        return inname[:-len("_layer")]
    else:
        return inname


for name in v1_layers.__all__:
    obj = getattr(v1_layers, name)
    if not __need_to_keep__(name):
        continue
    new_name = __convert_name__(name)
    if callable(obj) and __need_to_wrap__(name):
        globals()[new_name] = __wrap__(obj)
    else:
        globals()[new_name] = obj
    __all__.append(new_name)


def __data_layer__(name, type, **kwargs):
    l = v1_layers.data_layer(name, type.dim, **kwargs)
    l.data_type = type
    return l


data = __wrap__(__data_layer__)

LayerV2 = v1_layers.LayerOutput


def __get_used_layers__(output_layers, extra_layers=None):
    layer_names = set()
    parents = {}

    def add_parent(child, parent):
        if child in parents:
            parents[child].append(parent)
        else:
            parents[child] = [parent]

    def add_additional_parents():
        for sub_model in cp.g_config.model_config.sub_models:
            if sub_model.name == 'root':
                continue
            for link in sub_model.in_links:
                add_parent(link.link_name, link.layer_name)
                add_parent(sub_model.name, link.layer_name)
            for link in sub_model.out_links:
                add_parent(link.link_name, link.layer_name)
                add_parent(link.link_name, sub_model.name)
            for mem in sub_model.memories:
                if mem.boot_layer_name:
                    add_parent(mem.layer_name, mem.boot_layer_name)
                add_parent(mem.link_name, mem.layer_name)

    def dfs_travel(layer_name):
        if layer_name in layer_names:
            return
        layer_names.add(layer_name)
        layer = cp.g_layer_map[layer_name]

        for inp in layer.inputs:
            dfs_travel(inp.input_layer_name)
        if layer.name in parents:
            for p in parents[layer.name]:
                dfs_travel(p)

    add_additional_parents()

    for layer in output_layers:
        dfs_travel(layer.full_name)

    return layer_names


def __get_used_parameters__(layer_names):
    parameter_names = set()
    for name in layer_names:
        l = cp.g_layer_map[name]
        for inp in l.inputs:
            if inp.input_parameter_name:
                parameter_names.add(inp.input_parameter_name)
        if l.bias_parameter_name:
            parameter_names.add(l.bias_parameter_name)
    return parameter_names


def __get_used_submodels__(layer_names):
    submodel_names = set()
    for submodel in cp.g_config.model_config.sub_models:
        if submodel.name in layer_names:
            submodel_names.add(submodel.name)
    return submodel_names


def __get_used_evaluators__(layer_names):
    evaluator_names = set()
    for e in cp.g_config.model_config.evaluators:
        used = True
        for name in e.input_layers:
            if name not in layer_names:
                used = False
                break
        if used:
            evaluator_names.add(e.name)
    return evaluator_names


def __trim_submodel__(old_submodel, layer_names, input_layer_names,
                      output_layer_names, evaluator_names):

    submodel = SubModelConfig()
    submodel.name = old_submodel.name
    submodel.layer_names.extend(
        filter(lambda x: x in layer_names, old_submodel.layer_names))
    submodel.input_layer_names.extend(
        filter(lambda x: x in input_layer_names, submodel.layer_names))
    submodel.output_layer_names.extend(
        filter(lambda x: x in output_layer_names, submodel.layer_names))
    submodel.evaluator_names.extend(
        filter(lambda x: x in evaluator_names, old_submodel.evaluator_names))

    submodel.is_recurrent_layer_group = old_submodel.is_recurrent_layer_group
    submodel.reversed = old_submodel.reversed

    submodel.memories.extend(
        filter(lambda x: x.link_name in layer_names, old_submodel.memories))
    target_inlinkid = (old_submodel.target_inlinkid
                       if old_submodel.HasField('target_inlinkid') else -1)
    in_links = []
    for i, link in enumerate(old_submodel.in_links):
        if link.link_name in layer_names or i == target_inlinkid:
            in_links.append(link)
            if i == target_inlinkid:
                target_inlinkid = len(in_links) - 1
    submodel.in_links.extend(in_links)

    submodel.out_links.extend(
        filter(lambda x: x.link_name in layer_names, old_submodel.out_links))
    if old_submodel.HasField('generator'):
        submodel.generator.CopyFrom(old_submodel.generator)

    if old_submodel.HasField('target_inlinkid'):
        submodel.target_inlinkid = target_inlinkid
    return submodel


def parse_network(output_layers, extra_layers=None):
    if not isinstance(output_layers, collections.Sequence):
        output_layers = [output_layers]
    if extra_layers is not None and not isinstance(extra_layers,
                                                   collections.Sequence):
        extra_layers = [extra_layers]
    else:
        extra_layers = []

    layer_names = __get_used_layers__(output_layers + extra_layers)
    submodel_names = __get_used_submodels__(layer_names)
    submodel_names.add('root')
    parameter_names = __get_used_parameters__(layer_names)
    evaluator_names = __get_used_evaluators__(layer_names)
    input_layer_names = set()
    output_layer_names = set()

    model_config = ModelConfig()
    model_config.type = cp.g_config.model_config.type
    for l in cp.g_config.model_config.layers:
        if l.name not in layer_names:
            continue
        model_config.layers.extend([l])
        if l.type == 'data':
            model_config.input_layer_names.append(l.name)
            input_layer_names.add(l.name)

    for p in cp.g_config.model_config.parameters:
        if p.name in parameter_names:
            model_config.parameters.extend([p])

    for layer in output_layers:
        model_config.output_layer_names.append(layer.full_name)
        output_layer_names.add(layer.full_name)

    for e in cp.g_config.model_config.evaluators:
        if e.name in evaluator_names:
            model_config.evaluators.extend([e])

    for s in cp.g_config.model_config.sub_models:
        if s.name in submodel_names:
            s = __trim_submodel__(s, layer_names, input_layer_names,
                                  output_layer_names, evaluator_names)
            model_config.sub_models.extend([s])

    return model_config


def get_layer(name):
    return __layer_map__.get(name)


cp.begin_parse()
