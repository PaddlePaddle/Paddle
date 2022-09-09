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
# limitations under the License

import os
import yaml
import copy
import argparse
from . import constants


class BaseConfig(object):

    def __init__(self, category, config_dict=None):
        self._category = category
        self._config_dict = None
        if config_dict is not None:
            if isinstance(config_dict, dict):
                self._config_dict = config_dict
            else:
                raise ValueError(
                    "Expected a dictionary. But received: {}".format(
                        config_dict))
        # Initialize attributes by the default config
        config = constants.get_category_default_config(self._category)
        for field, default_value in config.items():
            setattr(self, field, default_value)

        # Overide attributes by the config_dict
        if self._config_dict:
            self.from_dict(self._config_dict)

    def from_dict(self, config_dict):
        config = constants.get_category_default_config(self._category)
        for field in config.keys():
            value = config_dict.get(field, constants.NOT_FOUND)
            # Use the default value if we cannot found the value
            if value != constants.NOT_FOUND:
                setattr(self, field, value)

    def to_dict(self):
        result_dict = {}
        config = constants.get_category_default_config(self._category)
        for field in config.keys():
            value = getattr(self, field)
            result_dict[field] = value
        for field, value in self.__dict__.items():
            if isinstance(value, BaseConfig):
                result_dict[field] = value.to_dict()
        return result_dict

    def __repr__(self):
        return yaml.dump(self.to_dict(),
                         default_flow_style=False,
                         sort_keys=True,
                         indent=4)


class RecomputeConfig(BaseConfig):

    def __init__(self, config_dict=None):
        category = constants.RECOMPUTE
        super(RecomputeConfig, self).__init__(category, config_dict)


class AMPConfig(BaseConfig):

    def __init__(self, config_dict=None):
        category = constants.AMP
        super(AMPConfig, self).__init__(category, config_dict)


class ShardingConfig(BaseConfig):

    def __init__(self, config_dict=None):
        category = constants.SHARDING
        super(ShardingConfig, self).__init__(category, config_dict)


class GradientMergeConfig(BaseConfig):

    def __init__(self, config_dict=None):
        category = constants.GRADIENT_MERGE
        super(GradientMergeConfig, self).__init__(category, config_dict)


class Strategy(BaseConfig):

    def __init__(self, config=None):
        if config is not None:
            if isinstance(config, dict):
                self._config_dict = copy.deepcopy(config)
            elif os.path.exists(config):
                with open(config, "rb") as yaml_file:
                    self._config_dict = yaml.load(yaml_file, Loader=yaml.Loader)
            else:
                raise ValueError(
                    "Expected a string path to an existing configuration file, or a dictionary. But received: {}"
                    .format(config))
        else:
            self._config_dict = {}
        category = constants.BASE
        super(Strategy, self).__init__(category, self._config_dict)

        config_dict = self._config_dict.get(constants.RECOMPUTE, None)
        self.recompute = RecomputeConfig(config_dict)

        config_dict = self._config_dict.get(constants.AMP, None)
        self.amp = AMPConfig(config_dict)

        config_dict = self._config_dict.get(constants.SHARDING, None)
        self.sharding = ShardingConfig(config_dict)

        config_dict = self._config_dict.get(constants.GRADIENT_MERGE, None)
        self.gradient_merge = GradientMergeConfig(config_dict)

        config_dict = self._config_dict.get(constants.GRADIENT_MERGE, None)
        self.gradient_merge = GradientMergeConfig(config_dict)


# def _parse_yaml_config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-c", "--config", help="Path of YAML configuration file for Auto Parallel")
#     yaml_args = parser.parse_args()
#     yaml_dict = yaml.load(
#         open(yaml_args.config, 'rb'), Loader=yaml.Loader)
#     _print_args(yaml_dict)
#     return yaml_dict

# def _print_args(yaml_dict):
#     """Print arguments."""

#     def add_dict(config, k, v):
#         if not isinstance(v, dict):
#             config[k] = v
#             return
#         for ik, iv in v.items():
#             add_dict(config, ik, iv)

#     print(
#         '------------------------ YAML configuration for Auto Parallel ------------------------',
#         flush=True)

#     for key, value in yaml_dict.items():
#         args = {}
#         add_dict(args, key, value)

#         print("{}:".format(key), flush=True)
#         str_list = []
#         for key, value in args.items():
#             dots = '.' * (48 - len(key))
#             str_list.append('  {} {} {}'.format(key, dots, value))
#         for arg in sorted(str_list, key=lambda x: x.lower()):
#             print(arg, flush=True)

#     print(
#         '--------------------------- End of YAML configuration --------------------------------',
#         flush=True)
