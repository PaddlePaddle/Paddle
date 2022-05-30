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

import paddle
import os
import random
import sys
import pickle
import shlex
import shutil
import inspect
import numpy as np
from collections import OrderedDict
from dist_pass_test_base import DistPassTestBase

import paddle.distributed.fleet as fleet
import paddle.distributed.auto_parallel as auto
sys.path.append("..")
import auto_parallel_gpt_model as modeling
from auto_parallel_gpt_model import GPTModel, GPTForPretraining, GPTPretrainingCriterion


class AutoPallelPassTestBase(DistPassTestBase):
    def setUp(self):
        paddle.enable_static()
        seed = int(os.environ.get('SEED', -1))
        if seed <= 0:
            seed = np.random.randint(low=1, high=1000000, size=[1])[0]
            os.environ['SEED'] = str(seed)
        self.seed = seed
        paddle.seed(self.seed)

        self.rtol = 1e-5
        self.atol = 1e-8
        self.equal_nan = False

        self.init()

    def init(self):
        pass

    def get_model(self, place, **kwargs):
        raise NotImplementedError()

    def apply_passes(self):
        raise NotImplementedError()

    def apply_no_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

    def check_main(self, gpus=None, **kwargs):
        no_pass_rets = self._distributed_launch(
            model=None, apply_pass=False, gpus=gpus, **kwargs)
        pass_rets = self._distributed_launch(
            model=None, apply_pass=True, gpus=gpus, **kwargs)
        self.check_results(no_pass_rets, pass_rets)

    def _run_gpu_main(self, model, apply_pass, dump_file, **kwargs):
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        place = paddle.CUDAPlace(gpu_id)
        scope = paddle.static.Scope()
        if apply_pass:
            self.apply_passes()
        else:
            self.apply_no_passes()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            with paddle.static.scope_guard(scope):
                with paddle.fluid.unique_name.guard():
                    main_prog, startup_prog, inputs, outputs, reader = self.get_model(
                        place, **kwargs)
                    inputs = self._to_var_names(inputs)
                    outputs = self._to_var_names(outputs)

        all_fetch_values = []
        exe = paddle.static.Executor(place)
        with paddle.static.scope_guard(scope):
            exe.run(startup_prog)
            for batch_id, input_data in enumerate(reader()):
                assert len(input_data) == len(inputs), "{} vs {}".format(
                    len(input_data), len(inputs))
                feed = dict(zip(inputs, input_data))
                fetch_values = exe.run(main_prog, feed=feed, fetch_list=outputs)
                if paddle.distributed.get_rank() == 0:
                    output_dict = OrderedDict(zip(outputs, fetch_values))
                    print('batch {}, outputs {}'.format(batch_id, output_dict))
                all_fetch_values.append(fetch_values)
        with open(dump_file, "wb") as f:
            pickle.dump(all_fetch_values, f)

    def get_gpt_model(self, strategy, place, batch_size, sequence_len,
                      vocab_size):
        modeling.init_global()
        if strategy == "dp":
            modeling._global_parallel_strategy = "dp"
            modeling._global_process_mesh = auto.ProcessMesh(mesh=[0, 1])
        elif strategy == "mp":
            modeling._global_parallel_strategy = "mp"
            modeling._global_process_mesh = auto.ProcessMesh(mesh=[0, 1])
        else:
            raise ValueError("'get_gpt_model' only support dp and mp.")

        tokens = paddle.static.data(
            name="tokens", shape=[batch_size, sequence_len], dtype='int64')
        position_ids = paddle.static.data(
            name="position_ids",
            shape=[batch_size, sequence_len],
            dtype='int64')
        attention_mask = paddle.static.data(
            name="attention_mask",
            shape=[batch_size, 1, sequence_len, sequence_len],
            dtype='float32')
        labels = paddle.static.data(
            name="labels", shape=[batch_size, sequence_len], dtype='int64')
        loss_mask = paddle.static.data(
            name="loss_mask", shape=[batch_size, sequence_len], dtype='float32')
        data_holder = [tokens, position_ids, attention_mask, labels, loss_mask]

        if modeling._global_parallel_strategy == "dp":
            auto.shard_tensor(
                tokens,
                dist_attr={
                    "process_mesh": modeling._global_process_mesh,
                    "dims_mapping": [0, -1]
                })
        elif modeling._global_parallel_strategy == "pp":
            auto.shard_tensor(
                tokens,
                dist_attr={
                    "process_mesh": modeling.PP_MESH_LIST[0],
                    "dims_mapping": [-1, -1]
                })
            auto.shard_tensor(
                attention_mask,
                dist_attr={
                    "process_mesh": modeling.PP_MESH_LIST[0],
                    "dims_mapping": [-1, -1, -1, -1]
                })

        gpt = GPTModel(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=256,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=1024,
            type_vocab_size=1,
            initializer_range=0.02,
            pad_token_id=0,
            eos_token_id=7,
            bos_token_id=0,
            eol_token_id=3)

        model = GPTForPretraining(
            gpt, vocab_size=1000, hidden_size=64, initializer_range=0.02)
        preds = model(tokens, position_ids, attention_mask)
        criterion = GPTPretrainingCriterion()
        loss = criterion(preds, labels, loss_mask)
        clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
        optimizer = paddle.fluid.optimizer.AdamOptimizer(
            learning_rate=0.00001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            grad_clip=clip)
        optimizer = fleet.distributed_optimizer(optimizer)
        startup_program = paddle.static.default_startup_program()
        _, _, dist_startup_prog, dist_main_prog = optimizer.minimize(
            loss, startup_program)

        def gen_data():
            np.random.seed(2021)
            for _ in range(10):
                tokens = []
                position_ids = []
                attention_mask = []
                labels = []
                loss_mask = []
                for _ in range(batch_size):
                    tokens.append(
                        np.random.randint(
                            vocab_size, size=sequence_len))
                    position_ids.append(np.arange(sequence_len))
                    attention_mask.append([np.tril(np.ones(sequence_len))])
                    labels.append(
                        np.random.randint(
                            vocab_size, size=sequence_len))
                    loss_mask.append(np.ones(sequence_len))

                yield tokens, position_ids, attention_mask, labels, loss_mask

        return dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data
