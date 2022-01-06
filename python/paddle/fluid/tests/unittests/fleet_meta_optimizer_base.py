# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import paddle
from paddle import fluid
import os
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker


class TestFleetMetaOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "1"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001,127.0.0.1:36002"
        self._debug = False

    def debug_program(self, main_prog, startup_prog):
        if not self._debug: return

        main_prog_ops = main_prog.global_block().ops
        startup_prog_ops = startup_prog.global_block().ops

        main_prog_op_types = [op.type for op in main_prog_ops]
        startup_prog_op_types = [op.type for op in startup_prog_ops]

        print("=== debug program and ops in func [{}] ==="
              .format(inspect.stack()[1].function))
        print(main_prog)
        print(main_prog_op_types)
        print(startup_prog)
        print(startup_prog_op_types)

    def net(self, main_prog, startup_prog):
        with fluid.program_guard(main_prog, startup_prog):
            with fluid.unique_name.guard():
                role = role_maker.PaddleCloudRoleMaker(is_collective=True)
                fleet.init(role)
                input_x = paddle.fluid.layers.data(
                    name="x", shape=[32], dtype='float32')
                input_y = paddle.fluid.layers.data(
                    name="y", shape=[1], dtype='int64')

                fc_1 = paddle.fluid.layers.fc(input=input_x,
                                              size=64,
                                              act='tanh')
                fc_2 = paddle.fluid.layers.fc(input=fc_1, size=256, act='tanh')
                prediction = paddle.fluid.layers.fc(input=[fc_2],
                                                    size=2,
                                                    act='softmax')
                cost = paddle.fluid.layers.cross_entropy(
                    input=prediction, label=input_y)
                avg_cost = paddle.fluid.layers.mean(x=cost)

                strategy = paddle.distributed.fleet.DistributedStrategy()
        return avg_cost, strategy

    def pp_net(self, main_prog, startup_prog, pp_degree=2):
        def fc_block(input_x):
            fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
            fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
            fc_3 = paddle.fluid.layers.fc(input=fc_2, size=64, act='tanh')
            return fc_3

        with fluid.program_guard(main_prog, startup_prog):
            with fluid.unique_name.guard():
                role = role_maker.PaddleCloudRoleMaker(is_collective=True)
                fleet.init(role)
                with fluid.device_guard("gpu:0"):
                    input_x = paddle.fluid.layers.data(
                        name="x", shape=[32], dtype='float32')
                    input_y = paddle.fluid.layers.data(
                        name="y", shape=[1], dtype='int64')

                for stage_idx in range(pp_degree):
                    with fluid.device_guard("gpu:" + str(stage_idx)):
                        input_x = fc_block(input_x)

                with fluid.device_guard("gpu:" + str(pp_degree - 1)):
                    prediction = paddle.fluid.layers.fc(input=[input_x],
                                                        size=2,
                                                        act='softmax')
                    cost = paddle.fluid.layers.cross_entropy(
                        input=prediction, label=input_y)
                    avg_cost = paddle.fluid.layers.mean(x=cost)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        return avg_cost, strategy

    def boundary_net(self, main_prog, startup_prog):
        with fluid.program_guard(main_prog, startup_prog):
            fleet.init(is_collective=True)
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            with paddle.static.device_guard('gpu:0'):
                linear = fluid.Linear(4, 8, bias_attr=False)
                out = linear(x)
            with paddle.static.device_guard('gpu:1'):
                linear = fluid.Linear(8, 5, bias_attr=False)
                out = linear(out)
                avg_cost = paddle.mean(out)
            strategy = fleet.DistributedStrategy()
        return avg_cost, strategy

    def optimizer(self,
                  loss,
                  strategy,
                  train_prog,
                  startup_prog,
                  name='momentum',
                  regularization=None,
                  grad_clip=None):
        with fluid.program_guard(train_prog, startup_prog):
            with fluid.unique_name.guard():
                if name == 'momentum':
                    optimizer = paddle.fluid.optimizer.Momentum(
                        learning_rate=0.01,
                        momentum=0.9,
                        regularization=regularization,
                        grad_clip=grad_clip)
                elif name == 'adam':
                    optimizer = paddle.fluid.optimizer.Adam(
                        learning_rate=0.01,
                        regularization=regularization,
                        grad_clip=grad_clip)
                optimizer = fleet.distributed_optimizer(
                    optimizer, strategy=strategy)
                optimizer.minimize(loss)

    def set_strategy(self, strategy, name):
        if name == 'amp':
            strategy.amp = True
            strategy.amp_configs = {
                "init_loss_scaling": 32768,
                "decr_every_n_nan_or_inf": 2,
                "incr_every_n_steps": 1000,
                "incr_ratio": 2.0,
                "use_dynamic_loss_scaling": True,
                "decr_ratio": 0.5,
                "custom_white_list": ['softmax'],
                "custom_black_list": ['tanh'],
            }
        elif name == 'pure_fp16':
            strategy.amp = True
            strategy.amp_configs = {
                "init_loss_scaling": 32768,
                "decr_every_n_nan_or_inf": 2,
                "incr_every_n_steps": 1000,
                "incr_ratio": 2.0,
                "use_dynamic_loss_scaling": True,
                "decr_ratio": 0.5,
                "custom_white_list": ['softmax'],
                "custom_black_list": ['tanh'],
                "use_pure_fp16": True,
                "use_fp16_guard": False,
            }

        elif name == 'dgc':
            strategy.dgc = True
            strategy.dgc_configs = {
                "rampup_begin_step": 128,
                "rampup_step": 100,
                "sparsity": [0.996, 0.999]
            }
        elif name == 'recompute':
            strategy.recompute = True
            strategy.recompute_configs = {
                "checkpoints": ["fc_0.tmp_2", "fc_1.tmp_2"]
            }
        elif name == 'lars':
            strategy.lars = True
            strategy.lars_configs = {
                "lars_coeff": 0.001,
                "lars_weight_decay": 0.0005,
                "epsilon": 0,
                "exclude_from_weight_decay": ["batch_norm", ".b"],
            }
        elif name == 'lamb':
            strategy.lamb = True
            strategy.lamb_configs = {
                'lamb_weight_decay': 0.01,
                'exclude_from_weight_decay': [],
            }
        elif name == 'localsgd':
            strategy.localsgd = True
            strategy.localsgd_configs = {
                'k_steps': 1,
                'begin_step': 1,
            }
        elif name == 'adaptive_localsgd':
            strategy.adaptive_localsgd = True
            strategy.adaptive_localsgd_configs = {
                'init_k_steps': 1,
                'begin_step': 1,
            }
        elif name == "gradient_merge":
            strategy.gradient_merge = True
            strategy.gradient_merge_configs = {"k_steps": 2, "avg": True}
        elif name == "sharding":
            strategy.sharding = True
            strategy.sharding_configs = {
                "sharding_segment_strategy": "segment_broadcast_MB",
                "segment_broadcast_MB": 0.2,
                "sharding_degree": 2,
            }
        elif name == "recompute-offload":
            strategy.recompute = True
            strategy.recompute_configs = {
                "checkpoints": ["fc_0.tmp_2", "fc_1.tmp_2"],
                "enable_offload": True,
                "checkpoint_shape": [256]
            }
        elif name == "pipeline":
            strategy.pipeline = True
            strategy.pipeline_configs = {
                "schedule_mode": "1F1B",
                "micro_batch_size": 2,
                "accumulate_steps": 4,
            }
        elif name == 'asp':
            strategy.asp = True
        else:
            raise NotImplementedError()
