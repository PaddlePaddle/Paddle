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

# Generate dot diagram file for the given paddle model config
# The generated file can be viewed using Graphviz (http://graphviz.org)

from __future__ import print_function

import six
import sys
import traceback

from paddle.trainer.config_parser import parse_config


def make_layer_label(layer_config):
    label = '%s type=%s' % (layer_config.name, layer_config.type)
    if layer_config.reversed:
        label += ' <=='

    label2 = ''
    if layer_config.active_type:
        label2 += 'act=%s ' % layer_config.active_type
    if layer_config.bias_parameter_name:
        label2 += 'bias=%s ' % layer_config.bias_parameter_name

    if label2:
        label += '\l' + label2
    return label


def make_diagram(config_file, dot_file, config_arg_str):
    config = parse_config(config_file, config_arg_str)
    make_diagram_from_proto(config.model_config, dot_file)


def make_diagram_from_proto(model_config, dot_file):
    # print >> sys.stderr, config
    name2id = {}
    f = open(dot_file, 'w')
    submodel_layers = set()

    def make_link(link):
        return 'l%s -> l%s;' % (name2id[link.layer_name],
                                name2id[link.link_name])

    def make_mem(mem):
        s = ''
        if mem.boot_layer_name:
            s += 'l%s -> l%s;\n' % (name2id[mem.boot_layer_name],
                                    name2id[mem.layer_name])
        s += 'l%s -> l%s [style=dashed];' % (name2id[mem.layer_name],
                                             name2id[mem.link_name])
        return s

    print('digraph graphname {', file=f)
    print('node [width=0.375,height=0.25];', file=f)
    for i in six.moves.xrange(len(model_config.layers)):
        l = model_config.layers[i]
        name2id[l.name] = i

    i = 0
    for sub_model in model_config.sub_models:
        if sub_model.name == 'root':
            continue
        print('subgraph cluster_%s {' % i, file=f)
        print('style=dashed;', file=f)
        label = '%s ' % sub_model.name
        if sub_model.reversed:
            label += '<=='
        print('label = "%s";' % label, file=f)
        i += 1
        submodel_layers.add(sub_model.name)
        for layer_name in sub_model.layer_names:
            submodel_layers.add(layer_name)
            lid = name2id[layer_name]
            layer_config = model_config.layers[lid]
            label = make_layer_label(layer_config)
            print('l%s [label="%s", shape=box];' % (lid, label), file=f)
        print('}', file=f)

    for i in six.moves.xrange(len(model_config.layers)):
        l = model_config.layers[i]
        if l.name not in submodel_layers:
            label = make_layer_label(l)
            print('l%s [label="%s", shape=box];' % (i, label), file=f)

    for sub_model in model_config.sub_models:
        if sub_model.name == 'root':
            continue
        for link in sub_model.in_links:
            print(make_link(link), file=f)
        for link in sub_model.out_links:
            print(make_link(link), file=f)
        for mem in sub_model.memories:
            print(make_mem(mem), file=f)

    for i in six.moves.xrange(len(model_config.layers)):
        for l in model_config.layers[i].inputs:
            print(
                'l%s -> l%s [label="%s"];' % (name2id[l.input_layer_name], i,
                                              l.input_parameter_name),
                file=f)

    print('}', file=f)
    f.close()


def usage():
    print(
        ("Usage: python show_model_diagram.py" +
         " CONFIG_FILE DOT_FILE [config_str]"),
        file=sys.stderr)
    exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        usage()

    config_file = sys.argv[1]
    dot_file = sys.argv[2]
    config_arg_str = sys.argv[3] if len(sys.argv) == 4 else ''

    try:
        make_diagram(config_file, dot_file, config_arg_str)
    except:
        traceback.print_exc()
        raise
