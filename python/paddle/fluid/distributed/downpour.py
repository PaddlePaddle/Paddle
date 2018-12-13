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

from .node import DownpourServer
from .node import DownpourWorker
from ..backward import append_backward
import ps_pb2 as pslib
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_inputs
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_outputs
from google.protobuf import text_format


class DownpourSGD(object):
    """
    Distributed optimizer of downpour stochastic gradient descent
    Standard implementation of Google's Downpour SGD
    in Large Scale Distributed Deep Networks

    Args:
        learning_rate (float): the learning rate used to update parameters. \
        Can be a float value
    Examples:
        .. code-block:: python
    
             downpour_sgd = fluid.distributed.DownpourSGD(learning_rate=0.2)
             downpour_sgd.minimize(cost)
    """

    def __init__(self, learning_rate=0.001, window=1):
        # todo(guru4elephant): add more optimizers here as argument
        # todo(guru4elephant): make learning_rate as a variable
        self.learning_rate_ = learning_rate
        self.window_ = window
        self.type = "downpour"

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        DownpounSGD is a distributed optimizer so
        that user can call minimize to generate backward
        operators and optimization operators within minmize function
        Args:
            loss(Variable): loss variable defined by user
            startup_program(Program): startup program that defined by user
            parameter_list(str list): parameter names defined by users
            no_grad_set(set): a set of variables that is defined by users
            so that these variables do not need gradient computation
        Returns:
            [ps_param, worker_skipped_ops]
            ps_param: parameter server protobuf desc
            worker_skipped_ops: operator names that need
            to be skipped during execution
        """
        params_grads = sorted(
            append_backward(loss, parameter_list, no_grad_set),
            key=lambda x: x[0].name)
        table_name = find_distributed_lookup_table(loss.block.program)
        prefetch_slots = find_distributed_lookup_table_inputs(
            loss.block.program, table_name)
        prefetch_slots_emb = find_distributed_lookup_table_outputs(
            loss.block.program, table_name)
        server = DownpourServer()
        # window is communication strategy
        worker = DownpourWorker(self.window_)
        # Todo(guru4elephant): support multiple tables definitions
        # currently support one big sparse table
        sparse_table_index = 0
        # currently merge all dense parameters into one dense table
        dense_table_index = 1
        params = []
        grads = []
        for i in params_grads:
            params.append(i[0])
        for i in params_grads:
            grads.append(i[1])
        server.add_sparse_table(sparse_table_index, self.learning_rate_,
                                prefetch_slots, prefetch_slots_emb)
        server.add_dense_table(dense_table_index, self.learning_rate_, params,
                               grads)
        worker.add_sparse_table(sparse_table_index, self.learning_rate_,
                                prefetch_slots, prefetch_slots_emb)
        worker.add_dense_table(dense_table_index, self.learning_rate_, params,
                               grads)
        ps_param = pslib.PSParameter()
        ps_param.server_param.CopyFrom(server.get_desc())
        ps_param.trainer_param.CopyFrom(worker.get_desc())
        # Todo(guru4elephant): figure out how to support more sparse parameters
        # currently only support lookup_table
        worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
        ps_param.trainer_param.skip_op.extend(worker_skipped_ops)
        return [ps_param, worker_skipped_ops]
