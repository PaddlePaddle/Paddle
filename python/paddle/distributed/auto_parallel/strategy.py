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

import copy
from . import constants


class BaseConfig:
    def __init__(self, category, config_dict=None):
        self._category = category
        self._config_dict = None
        if config_dict is not None:
            if isinstance(config_dict, dict):
                self._config_dict = config_dict
            else:
                raise ValueError(
                    "Expected a dictionary. But received: {}".format(
                        config_dict
                    )
                )
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
        result_dict = self.to_dict()
        string = "{"
        for k, v in result_dict.items():
            string += "\"%s\":\"%s\"," % (k, v)
        return string + "}"

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


class RecomputeConfig(BaseConfig):
    def __init__(self, config_dict=None):
        category = constants.RECOMPUTE
        super().__init__(category, config_dict)


class AMPConfig(BaseConfig):
    def __init__(self, config_dict=None):
        category = constants.AMP
        super().__init__(category, config_dict)


class ShardingConfig(BaseConfig):
    def __init__(self, config_dict=None):
        category = constants.SHARDING
        super().__init__(category, config_dict)


class GradientMergeConfig(BaseConfig):
    def __init__(self, config_dict=None):
        category = constants.GRADIENT_MERGE
        super().__init__(category, config_dict)


class QATConfig(BaseConfig):
    def __init__(self, config_dict=None):
        category = constants.QAT
        super().__init__(category, config_dict)


class TuningConfig(BaseConfig):
    def __init__(self, config_dict=None):
        category = constants.TUNING
        super().__init__(category, config_dict)


class DatasetConfig(BaseConfig):
    def __init__(self, config_dict=None):
        category = constants.DATASET
        super().__init__(category, config_dict)


class Strategy(BaseConfig):
    """
    The `Strategy` object is used to configure the paralleization and optimization beheviors.

    Args:
        config (dict|string, optional): If this is None, the default configurations will used.
        If this is a dictionary, the recognized key-value of it will be used to override the default
        configurations while other default configurations are left unchanged. If this is a string,
        it is interpreted as the path to a YAML configuration and will be loaded to override the
        corresponding default configurations.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed.fleet import auto

            strategy = auto.Strategy()
            sharding = strategy.sharding
            self.assertEqual(sharding.enabled, False)
            self.assertEqual(sharding.stage, 1)
            self.assertEqual(sharding.degree, 8)
            sharding.enabled = True
            sharding.stage = 2
            sharding.degree = 2
            self.assertEqual(sharding.enabled, True)
            self.assertEqual(sharding.stage, 2)
            self.assertEqual(sharding.degree, 2)

    """

    def __init__(self, config=None):
        if config is not None:
            if isinstance(config, dict):
                self._config_dict = copy.deepcopy(config)
            # elif os.path.exists(config):
            #     with open(config, "rb") as yaml_file:
            #         self._config_dict = yaml.load(yaml_file, Loader=yaml.Loader)
            else:
                raise ValueError(
                    "Expected a dictionary. But received: {}".format(config)
                )
        else:
            self._config_dict = {}

        category = constants.BASE
        super().__init__(category, self._config_dict)

        config_dict = self._config_dict.get(constants.RECOMPUTE, None)
        self.recompute = RecomputeConfig(config_dict)

        config_dict = self._config_dict.get(constants.AMP, None)
        self.amp = AMPConfig(config_dict)

        config_dict = self._config_dict.get(constants.SHARDING, None)
        self.sharding = ShardingConfig(config_dict)

        config_dict = self._config_dict.get(constants.GRADIENT_MERGE, None)
        self.gradient_merge = GradientMergeConfig(config_dict)

        config_dict = self._config_dict.get(constants.QAT, None)
        self.qat = QATConfig(config_dict)

        config_dict = self._config_dict.get(constants.TUNING, None)
        self.tuning = TuningConfig(config_dict)

        config_dict = self._config_dict.get(constants.DATASET, None)
        self.dataset = DatasetConfig(config_dict)
