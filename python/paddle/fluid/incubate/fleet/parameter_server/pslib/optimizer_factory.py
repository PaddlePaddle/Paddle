#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ["DistributedAdam"]
import ps_pb2 as pslib
import paddle.fluid as fluid
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_inputs
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_outputs
from google.protobuf import text_format
from .node import DownpourWorker, DownpourServer


class DistributedOptimizerImplBase(object):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._learning_rate = optimizer._learning_rate
        self._regularization = optimizer.regularization

    def minimize(self,
                 losses,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        pass


class DistributedAdam(DistributedOptimizerImplBase):
    def __init__(self, optimizer):
        # todo(guru4elephant): add more optimizers here as argument
        # todo(guru4elephant): make learning_rate as a variable
        super(DistributedAdam, self).__init__(optimizer)
        self._window = 1
        self.type = "downpour"
        self.data_norm_name = [
            ".batch_size", ".batch_square_sum", ".batch_sum",
            ".batch_size@GRAD", ".batch_square_sum@GRAD", ".batch_sum@GRAD"
        ]

    def _minimize(self,
                  losses,
                  startup_program=None,
                  parameter_list=None,
                  no_grad_set=None,
                  strategy={}):
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
            [optimize_ops, grads_and_weights]
        """
        if not isinstance(losses, list):
            losses = [losses]

        table_name = find_distributed_lookup_table(losses[0].block.program)
        prefetch_slots = find_distributed_lookup_table_inputs(
            losses[0].block.program, table_name)
        prefetch_slots_emb = find_distributed_lookup_table_outputs(
            losses[0].block.program, table_name)

        ps_param = pslib.PSParameter()
        server = DownpourServer()
        worker = DownpourWorker(self.window_)
        if strategy.get("fleet_desc_file") is not None:
            fleet_desc_file = strategy["fleet_desc_file"]
            with open(fleet_desc_file) as f:
                text_format.Merge(f.read(), ps_param)
            server.get_desc().CopyFrom(ps_param.server_param)
            worker.get_desc().CopyFrom(ps_param.trainer_param)
        sparse_table_index = 0
        server.add_sparse_table(sparse_table_index, self._learning_rate,
                                prefetch_slots, prefetch_slots_emb)
        worker.add_sparse_table(sparse_table_index, self._learning_rate,
                                prefetch_slots, prefetch_slots_emb)
        dense_table_index = 1
        program_configs = {}
        param_grads_list = []

        for loss_index in range(len(losses)):
            #program_config = ps_param.trainer_param.program_config.add()
            #program_config.program_id = str(
            #    id(losses[loss_index].block.program))
            program_id = str(id(losses[loss_index].block.program))
            program_configs[program_id] = {
                "pull_sparse": [sparse_table_index],
                "push_sparse": [sparse_table_index]
            }

            #program_config.pull_sparse_table_id.extend([sparse_table_index])
            #program_config.push_sparse_table_id.extend([sparse_table_index])
            params_grads = sorted(
                fluid.backward.append_backward(losses[loss_index],
                                               parameter_list, no_grad_set),
                key=lambda x: x[0].name)
            param_grads_list.append(params_grads)
            params = []
            grads = []
            data_norm_params = []
            data_norm_grads = []
            for i in params_grads:
                is_data_norm_data = False
                for data_norm_name in self.data_norm_name:
                    if i[0].name.endswith(data_norm_name):
                        is_data_norm_data = True
                        data_norm_params.append(i[0])
                if not is_data_norm_data:
                    params.append(i[0])
            for i in params_grads:
                is_data_norm_data = False
                for data_norm_grad in self.data_norm_name:
                    if i[0].name.endswith(data_norm_grad):
                        is_data_norm_data = True
                        data_norm_grads.append(i[1])
                if not is_data_norm_data:
                    grads.append(i[1])
            server.add_dense_table(dense_table_index, self._learning_rate,
                                   params, grads)
            worker.add_dense_table(dense_table_index, self._learning_rate,
                                   params, grads)
            program_configs[program_id]["pull_dense"] = [dense_table_index]
            program_configs[program_id]["push_dense"] = [dense_table_index]
            #program_config.pull_dense_table_id.extend([dense_table_index])
            #program_config.push_dense_table_id.extend([dense_table_index])
            if len(data_norm_params) != 0 and len(data_norm_grads) != 0:
                dense_table_index += 1
                server.add_data_norm_table(dense_table_index,
                                           self._learning_rate,
                                           data_norm_params, data_norm_grads)
                worker.add_dense_table(dense_table_index, self._learning_rate,
                                       data_norm_params, data_norm_grads)
                #program_config.pull_dense_table_id.extend([dense_table_index])
                #program_config.push_dense_table_id.extend([dense_table_index])
                program_configs[program_id]["pull_dense"].extend(
                    [dense_table_index])
                program_configs[program_id]["push_dense"].extend(
                    [dense_table_index])
            dense_table_index += 1
            #program_configs.append(program_config)
        ps_param.server_param.CopyFrom(server.get_desc())
        ps_param.trainer_param.CopyFrom(worker.get_desc())
        #for program_config in program_configs:
        #    ps_param.trainer_param.program_config.extend([program_config])
        # Todo(guru4elephant): figure out how to support more sparse parameters
        # currently only support lookup_table
        worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
        ps_param.trainer_param.skip_op.extend(worker_skipped_ops)

        opt_info = {}
        opt_info["program_configs"] = program_configs
        opt_info["trainer"] = "DistMultiTrainer"
        opt_info["device_worker"] = "DownpourSGD"
        opt_info["optimizer"] = "DownpourSGD"
        opt_info["fleet_desc"] = ps_param
        opt_info["worker_skipped_ops"] = worker_skipped_ops
        opt_info["use_cvm"] = False
        if strategy.get("use_cvm") is not None:
            opt_info["use_cvm"] = strategy["use_cvm"]

        for loss in losses:
            loss.block.program._fleet_opt = opt_info

        return None, param_grads_list[0], opt_info
