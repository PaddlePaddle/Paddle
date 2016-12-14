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

from paddle.proto.DataConfig_pb2 import DataConfig

g_config = None


def SimpleData(files=None,
               feat_dim=None,
               context_len=None,
               buffer_capacity=None):

    data_config = DataConfig()
    data_config.type = 'simple'
    data_config.files = files
    data_config.feat_dim = feat_dim
    if context_len is not None:
        data_config.context_len = context_len
    if buffer_capacity:
        data_config.buffer_capacity = buffer_capacity
    return data_config


def get_config_funcs(trainer_config):
    global g_config
    g_config = trainer_config
    return dict(SimpleData=SimpleData)
