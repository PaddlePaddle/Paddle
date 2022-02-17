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
class TestDefaultConfigure(unittest.TestCase):
    def test_func(self):
        ipu_strategy = paddle.static.IpuStrategy()

        confs = {}
        confs['num_ipus'] = 1

        for k, v in confs.items():
            assert v == ipu_strategy.get_option(
                k), f"Check default option: {k} to value: {v} failed "


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestConfigure(unittest.TestCase):
    def test_func(self):
        ipu_strategy = paddle.static.IpuStrategy()

        confs = {}
        confs['is_training'] = False
        confs['enable_pipelining'] = False
        confs['enable_manual_shard'] = True
        confs['save_init_onnx'] = True
        confs['save_onnx_checkpoint'] = True
        confs['need_avg_shard'] = True
        confs['enable_fp16'] = True
        confs['enable_pipelining'] = True
        confs['enable_manual_shard'] = True
        confs['enable_half_partial'] = True
        confs['enable_stochastic_rounding'] = True

        confs['num_ipus'] = 2
        confs['batches_per_step'] = 5
        confs['micro_batch_size'] = 4
        confs['save_per_n_step'] = 10

        confs['loss_scaling'] = 5.0
        confs['max_weight_norm'] = 100.0
        confs['available_memory_proportion'] = 0.3

        for k, v in confs.items():
            ipu_strategy.set_option({k: v})
            assert v == ipu_strategy.get_option(
                k), f"Setting option: {k} to value: {v} failed "


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestEnablePattern(unittest.TestCase):
    def test_enable_patern(self):
        ipu_strategy = paddle.static.IpuStrategy()
        pattern = 'LSTMOp'
        # LSTMOp Pattern is not enabled by default
        # assert not ipu_strategy.is_pattern_enabled(pattern)
        ipu_strategy.enable_pattern(pattern)
        assert ipu_strategy.is_pattern_enabled(pattern) == True

    def test_disable_pattern(self):
        ipu_strategy = paddle.static.IpuStrategy()
        pattern = 'LSTMOp'
        ipu_strategy.enable_pattern(pattern)
        ipu_strategy.disable_pattern(pattern)
        assert ipu_strategy.is_pattern_enabled(pattern) == False


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestIpuStrategyLoadDict(unittest.TestCase):
    def test_enable_patern(self):
        ipu_strategy = paddle.static.IpuStrategy()
        test_conf = {
            "micro_batch_size": 23,
            "batches_per_step": 233,
            "enableGradientAccumulation": True,
            "enableReplicatedGraphs": True,
            "enable_fp16": True,
            "save_init_onnx": True,
            "save_onnx_checkpoint": True
        }
        ipu_strategy.set_option(test_conf)
        for k, v in test_conf.items():
            assert v == ipu_strategy.get_option(k)


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestIpuStrategyEngineOptions(unittest.TestCase):
    def test_enable_patern(self):
        ipu_strategy = paddle.static.IpuStrategy()
        engine_conf = {
            'debug.allowOutOfMemory': 'true',
            'autoReport.directory': 'path',
            'autoReport.all': 'true'
        }
        ipu_strategy.set_option({'engineOptions': engine_conf})
        for k, v in ipu_strategy.get_option('engineOptions').items():
            assert v == engine_conf[k]


if __name__ == "__main__":
    unittest.main()
