#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import paddle
import paddle.static

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestIpuStrategy(unittest.TestCase):
    def test_set_options(self):
        ipu_strategy = paddle.static.IpuStrategy()
        all_option_names = ipu_strategy._ipu_strategy.get_all_option_names()
        for option_name in all_option_names:
            option = ipu_strategy._ipu_strategy.get_option(option_name)
            option_type = option['type']
            option_value = option['value']
            if option_type in ['double']:
                set_value = option_value + 0.5
            elif option_type == 'uint64':
                set_value = option_value + 1
            elif option_type == 'bool':
                set_value = not option_value
            else:
                continue
            ipu_strategy.set_options({option_name: set_value})
            new_value = ipu_strategy.get_option(option_name)
            assert new_value == set_value, f"set {option_name} to {set_value} failed"

    def test_set_string_options(self):
        ipu_strategy = paddle.static.IpuStrategy()
        options = {
            'cache_path': 'paddle_cache',
            'log_dir': 'paddle_log',
            'partials_type_matmuls': 'half',
            'partials_type_matmuls': 'float',
        }
        ipu_strategy.set_options(options)
        for k, v in options.items():
            assert v == ipu_strategy.get_option(k), f"set {k} to {v} failed "

    def test_set_other_options(self):
        ipu_strategy = paddle.static.IpuStrategy()
        options = {}
        options['dot_checks'] = ['0', '1', '2', '3']
        options['engine_options'] = {
            'debug.allowOutOfMemory': 'true',
            'autoReport.directory': 'path',
            'autoReport.all': 'true'
        }
        for k, v in options.items():
            ipu_strategy.set_options({k: v})
            assert v == ipu_strategy.get_option(k), f"set {k} to {v} failed "


if __name__ == "__main__":
    unittest.main()
