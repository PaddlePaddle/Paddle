# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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

from paddle.trainer.config_parser import parse_config
from paddle.proto import TrainerConfig_pb2
import sys

__all__ = []

if __name__ == '__main__':
    if len(sys.argv) == 2:
        conf = parse_config(sys.argv[1], '')
    elif len(sys.argv) == 3:
        conf = parse_config(sys.argv[1], sys.argv[2])
    else:
        raise RuntimeError()

    assert isinstance(conf, TrainerConfig_pb2.TrainerConfig)

    print conf.model_config
