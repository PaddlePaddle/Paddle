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

from .compress_pass import CompressPass
from .config import ConfigFactory

__all__ = ['build_compressor']


def build_compressor(place=None,
                     data_reader=None,
                     data_feeder=None,
                     scope=None,
                     metrics=None,
                     epoch=None,
                     config=None):
    if config is not None:
        factory = ConfigFactory(config)
        comp_pass = factory.get_compress_pass()
    else:
        comp_pass = CompressPass()
    comp_pass.place = place
    comp_pass.data_reader = data_reader
    comp_pass.data_feeder = data_feeder
    comp_pass.scope = scope
    comp_pass.metrics = metrics
    comp_pass.epoch = epoch
    return comp_pass
