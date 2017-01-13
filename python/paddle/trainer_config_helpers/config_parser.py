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

import paddle.trainer.config_parser as config_parser
'''
This file is a wrapper of formal config_parser. The main idea of this file is to 
separete different config logic into different function, such as network configuration
 and optimizer configuration.
'''

__all__ = [
    "parse_trainer_config", "parse_network_config", "parse_optimizer_config"
]


def parse_trainer_config(trainer_conf, config_arg_str):
    return config_parser.parse_config(trainer_conf, config_arg_str)


def parse_network_config(network_conf):
    config = config_parser.parse_config(network_conf, '')
    return config.model_config


def parse_optimizer_config(optimizer_conf):
    config = config_parser.parse_config(optimizer_conf, '')
    return config.opt_config
