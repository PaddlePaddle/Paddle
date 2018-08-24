#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import argparse
import time
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import os
import sys
import six

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1

from transformer_config import ModelHyperParams, TrainTaskConfig
from transformer_train import train_loop

from transformer_model import transformer
from optim import LearningRateScheduler


def get_model(is_dist, is_async):
    sum_cost, avg_cost, predict, token_num = transformer(
        ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size,
        ModelHyperParams.max_length + 1, ModelHyperParams.n_layer,
        ModelHyperParams.n_head, ModelHyperParams.d_key,
        ModelHyperParams.d_value, ModelHyperParams.d_model,
        ModelHyperParams.d_inner_hid, ModelHyperParams.dropout,
        ModelHyperParams.weight_sharing, TrainTaskConfig.label_smooth_eps)

    local_lr_scheduler = LearningRateScheduler(ModelHyperParams.d_model,
                                               TrainTaskConfig.warmup_steps,
                                               TrainTaskConfig.learning_rate)

    if not is_dist:
        optimizer = fluid.optimizer.Adam(
            learning_rate=local_lr_scheduler.learning_rate,
            beta1=TrainTaskConfig.beta1,
            beta2=TrainTaskConfig.beta2,
            epsilon=TrainTaskConfig.eps)
        optimizer.minimize(sum_cost)
    elif is_async:
        optimizer = fluid.optimizer.SGD(0.003)
        optimizer.minimize(sum_cost)
    else:
        lr_decay = fluid.layers\
         .learning_rate_scheduler\
         .noam_decay(ModelHyperParams.d_model,
            TrainTaskConfig.warmup_steps)

        optimizer = fluid.optimizer.Adam(
            learning_rate=lr_decay,
            beta1=TrainTaskConfig.beta1,
            beta2=TrainTaskConfig.beta2,
            epsilon=TrainTaskConfig.eps)
        optimizer.minimize(sum_cost)

    return sum_cost, avg_cost, predict, token_num, local_lr_scheduler


def get_transpiler(trainer_id, main_program, pserver_endpoints, trainers):
    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=trainer_id,
        program=main_program,
        pservers=pserver_endpoints,
        trainers=trainers)
    return t


class DistTransformer2x2(object):
    def run_pserver(self, pserver_endpoints, trainers, current_endpoint,
                    trainer_id, is_async):
        get_model(True, is_async)
        t = get_transpiler(trainer_id,
                           fluid.default_main_program(), pserver_endpoints,
                           trainers)
        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(pserver_prog)

    def _wait_ps_ready(self, pid):
        retry_times = 20
        while True:
            assert retry_times >= 0, "wait ps ready failed"
            time.sleep(3)
            print("waiting ps ready: ", pid)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                retry_times -= 1

    def run_trainer(self,
                    place,
                    dev_count,
                    endpoints,
                    trainer_id,
                    trainers,
                    is_dist=True,
                    is_async=False):

        sum_cost, avg_cost, predict, token_num, local_lr_scheduler = get_model(
            is_dist, is_async)

        if is_dist:
            t = get_transpiler(trainer_id,
                               fluid.default_main_program(), endpoints,
                               trainers)
            trainer_prog = t.get_trainer_program()
            TrainTaskConfig.batch_size = 10
            TrainTaskConfig.train_file_pattern = "test_data/transformer/train.tok.clean.bpe.32000.en-de.train_{}".format(
                trainer_id)
        else:
            trainer_prog = fluid.default_main_program()

        startup_exe = fluid.Executor(place)

        TrainTaskConfig.local = not is_dist

        train_loop(startup_exe, trainer_prog, dev_count, sum_cost, avg_cost,
                   local_lr_scheduler, token_num, predict)


def main(role="pserver",
         endpoints="127.0.0.1:9123",
         trainer_id=0,
         current_endpoint="127.0.0.1:9123",
         trainers=1,
         is_dist=True,
         is_async=False):

    model = DistTransformer2x2()

    if role == "PSERVER" or (not TrainTaskConfig.use_gpu):
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(0)
        dev_count = 1

    if role == "pserver":
        model.run_pserver(endpoints, trainers, current_endpoint, trainer_id,
                          is_async)
    else:
        model.run_trainer(place, dev_count, endpoints, trainer_id, trainers,
                          is_dist, is_async)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python dist_transformer.py [pserver/trainer] [endpoints] [trainer_id] [current_endpoint] [trainers] [is_dist] [sync_mode]"
        )
    role = sys.argv[1]
    endpoints = sys.argv[2]
    trainer_id = int(sys.argv[3])
    current_endpoint = sys.argv[4]
    trainers = int(sys.argv[5])
    is_dist = True if sys.argv[6] == "TRUE" else False
    # FIXME(typhoonzero): refine this test.
    is_async = True if sys.argv[7] == "TRUE" else False
    main(
        role=role,
        endpoints=endpoints,
        trainer_id=trainer_id,
        current_endpoint=current_endpoint,
        trainers=trainers,
        is_dist=is_dist,
        is_async=is_async)
