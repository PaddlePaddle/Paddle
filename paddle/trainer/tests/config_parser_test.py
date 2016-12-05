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

from paddle.trainer.config_parser import parse_config_and_serialize

if __name__ == '__main__':
    parse_config_and_serialize('trainer/tests/test_config.conf', '')
    parse_config_and_serialize(
        'trainer/tests/sample_trainer_config.conf',
        'extension_module_name=paddle.trainer.config_parser_extension')
    parse_config_and_serialize('gserver/tests/pyDataProvider/trainer.conf', '')
