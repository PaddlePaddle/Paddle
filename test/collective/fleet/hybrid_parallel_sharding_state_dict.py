# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import random
import unittest

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.mix_precision_utils import (
    MixPrecisionOptimizer,
)

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4
STEPS = 10


class SimpleDPNet(paddle.nn.Layer):
    def __init__(
        self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
    ):
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(x, self.embedding.weight, transpose_y=True)
        return x


class TestDistShardingTraining(unittest.TestCase):
    def setUp(self):
        random.seed(2021)
        np.random.seed(2021)
        paddle.seed(2021)

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": 2,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=self.strategy)
        self.data = [
            np.random.randint(
                0,
                vocab_size,
                (
                    batch_size,
                    seq_length,
                ),
            )
            for _ in range(STEPS)
        ]

    def build_adam_optimizer(self, model, lr=0.001):
        clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=lr,
            weight_decay=0.00001,
            grad_clip=clip,
        )
        return optimizer

    def test_set_state_dict(self):
        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        init_lr = 0.001
        init_lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=init_lr, T_max=1
        )
        local_optimizer = self.build_adam_optimizer(model, init_lr_scheduler)
        dist_optimizer = fleet.distributed_optimizer(local_optimizer)
        # prepare state_dict
        state_dict = {}
        # lr_scheduler
        base_lr = 0.1
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=base_lr, T_max=1
        )
        state_dict["LR_Scheduler"] = lr_scheduler.state_dict()
        # master_weights and accumulators
        state_dict["master_weights"] = {}
        all_param_names = []
        accumulator_names = ["moment1", "moment2", "moment2_max"]
        #
        local_params = dist_optimizer._rank2params[
            dist_optimizer._sharding_rank
        ]
        local_param_names = [p.name for p in local_params]
        local_acc_names = []
        other_acc_names = []
        for p in model.parameters():
            var_name = dist_optimizer._gen_master_weight_var_name(p)
            var = paddle.static.create_global_var(
                name=var_name,
                shape=p.shape,
                value=0,
                dtype='float32',
                persistable=True,
            )
            var = paddle.randn(shape=var.shape, dtype=var.dtype, name=var.name)
            state_dict["master_weights"][p.name] = var
            # accumulator
            for name in accumulator_names:
                acc_name = p.name + '_' + name
                state_dict[acc_name] = paddle.randn(
                    shape=var.shape, dtype=var.dtype, name=acc_name
                )
                if p.name in local_param_names:
                    local_acc_names.append(acc_name)
                else:
                    other_acc_names.append(acc_name)
            all_param_names.append(p.name)
        # test api
        tmp_state_dict = copy.deepcopy(state_dict)
        dist_optimizer.set_state_dict(state_dict)
        # check result
        other_param_names = [
            p_name
            for p_name in all_param_names
            if p_name not in local_param_names
        ]
        inner_opt = dist_optimizer._inner_opt
        self.assertEqual(inner_opt._learning_rate.last_lr, base_lr)
        assert hasattr(inner_opt, "_master_weights")
        for p_name, weight in inner_opt._master_weights.items():
            assert p_name in local_param_names
            assert p_name not in other_param_names
            assert p_name in tmp_state_dict["master_weights"]
            np.testing.assert_array_almost_equal(
                weight.numpy(), tmp_state_dict["master_weights"][p_name].numpy()
            )
        for acc_name, val in inner_opt._accumulators_holder.items():
            assert acc_name in local_acc_names
            assert acc_name not in other_acc_names
            assert acc_name in tmp_state_dict
            np.testing.assert_array_almost_equal(
                val.numpy(), tmp_state_dict[acc_name].numpy()
            )

    def test_clear_grad(self):
        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )

        local_optimizer = self.build_adam_optimizer(model)
        dist_optimizer = fleet.distributed_optimizer(local_optimizer)

        tmp_parameter_list = []
        for p in dist_optimizer._inner_opt._parameter_list:
            main_grad = paddle.randn(shape=p.shape, dtype=p.dtype, name=p.name)
            p.main_grad = main_grad
            tmp_parameter_list.append(p)

        assert hasattr(
            dist_optimizer._inner_opt._parameter_list[0], "main_grad"
        )
        # test set_to_zero True
        dist_optimizer._inner_opt.clear_grad(set_to_zero=True)
        for p in dist_optimizer._inner_opt._parameter_list:
            np.testing.assert_array_almost_equal(
                p.main_grad.numpy(), np.zeros(p.main_grad.numpy().shape)
            )
        # test set_to_zero False
        dist_optimizer._inner_opt.clear_grad(set_to_zero=False)
        for p in dist_optimizer._inner_opt._parameter_list:
            self.assertTrue(p.main_grad is None)

    def test_set_inner_opt_attr(self):
        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )

        local_optimizer = self.build_adam_optimizer(model)
        local_optimizer = MixPrecisionOptimizer(local_optimizer)
        dist_optimizer = fleet.distributed_optimizer(local_optimizer)
        sharding_opt = dist_optimizer._inner_opt
        sharding_opt._set_inner_opt_attr('_parameter_list', 123)
        self.assertTrue(hasattr(sharding_opt._inner_opt, '_parameter_list'))
        self.assertTrue(
            hasattr(sharding_opt._inner_opt._inner_opt, '_parameter_list')
        )
        self.assertEqual(sharding_opt._inner_opt._parameter_list, 123)
        self.assertEqual(
            sharding_opt._inner_opt._inner_opt._parameter_list, 123
        )

        sharding_opt._set_inner_opt_attr('_param_groups', 123)
        self.assertTrue(hasattr(sharding_opt._inner_opt, '_param_groups'))
        self.assertTrue(
            hasattr(sharding_opt._inner_opt._inner_opt, '_param_groups')
        )
        self.assertEqual(sharding_opt._inner_opt._param_groups, 123)
        self.assertEqual(sharding_opt._inner_opt._inner_opt._param_groups, 123)

        # test bad case
        try:
            sharding_opt._set_inner_opt_attr(123, 123)
            self.assertTrue(False)
        except:
            pass


if __name__ == "__main__":
    unittest.main()
