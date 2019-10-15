#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import inspect
import funcsigs
import yaml
from collections import OrderedDict
from ..prune import *
from ..quantization import *
from .strategy import *
from ..distillation import *
from ..searcher import *
from ..nas import *

__all__ = ['ConfigFactory']
"""This factory is used to create instances by loading and parsing configure file with yaml format.
"""

PLUGINS = ['pruners', 'quantizers', 'distillers', 'strategies', 'controllers']


class ConfigFactory(object):
    def __init__(self, config):
        """Init a factory from configure file."""
        self.instances = {}
        self.compressor = {}
        self.version = None
        self._parse_config(config)

    def instance(self, name):
        """
        Get instance from factory.
        """
        if name in self.instances:
            return self.instances[name]
        else:
            return None

    def _new_instance(self, name, attrs):
        if name not in self.instances:
            class_ = globals()[attrs['class']]
            sig = funcsigs.signature(class_.__init__)
            keys = [
                param.name for param in sig.parameters.values()
                if (param.kind == param.POSITIONAL_OR_KEYWORD)
            ][1:]
            keys = set(attrs.keys()).intersection(set(keys))
            args = {}
            for key in keys:
                value = attrs[key]
                if isinstance(value, str) and value.lower() == 'none':
                    value = None
                if isinstance(value, str) and value in self.instances:
                    value = self.instances[value]
                if isinstance(value, list):
                    for i in range(len(value)):
                        if isinstance(value[i],
                                      str) and value[i] in self.instances:
                            value[i] = self.instances[value[i]]

                args[key] = value
            self.instances[name] = class_(**args)
        return self.instances.get(name)

    def _parse_config(self, config):
        assert config
        with open(config, 'r') as config_file:
            key_values = self._ordered_load(config_file)
            for key in key_values:
                # parse version
                if key == 'version' and self.version is None:
                    self.version = int(key_values['version'])
                    assert self.version == int(key_values['version'])

                # parse pruners
                if key in PLUGINS:
                    instances = key_values[key]
                    for name in instances:
                        self._new_instance(name, instances[name])

                if key == 'compressor':
                    self.compressor['strategies'] = []
                    self.compressor['epoch'] = key_values[key]['epoch']
                    if 'init_model' in key_values[key]:
                        self.compressor['init_model'] = key_values[key][
                            'init_model']
                    if 'checkpoint_path' in key_values[key]:
                        self.compressor['checkpoint_path'] = key_values[key][
                            'checkpoint_path']
                    if 'eval_epoch' in key_values[key]:
                        self.compressor['eval_epoch'] = key_values[key][
                            'eval_epoch']
                    if 'strategies' in key_values[key]:
                        for name in key_values[key]['strategies']:
                            strategy = self.instance(name)
                            self.compressor['strategies'].append(strategy)

                if key == 'include':
                    for config_file in key_values[key]:
                        self._parse_config(config_file.strip())

    def _ordered_load(self,
                      stream,
                      Loader=yaml.Loader,
                      object_pairs_hook=OrderedDict):
        """
        See: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
        """

        class OrderedLoader(Loader):
            pass

        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))

        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
        return yaml.load(stream, OrderedLoader)
