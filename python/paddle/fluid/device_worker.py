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

__all__ = ['DeviceWorker', 'Hogwild', 'DownpourSGD']


class DeviceWorker(object):
    """
    DeviceWorker is an abstract class, which generates worker desc.
    This class is an inner class that we do computation logics within
    the implementation. For example, execution of a program or a graph.
    """

    def __init__(self):
        """
        Init.
        """
        self._program = None
        self._infer = None

    def _set_infer(self, infer=False):
        """
        set inference flag for current device worker
        
        Args:
            infer(bool): whether to do inference
        """
        self._infer = infer

    def _set_fleet_desc(self, fleet_desc):
        """
        Set fleet desc.

        Args:
            fleet_desc(PSParameter): pslib.PSParameter object
        """
        self._fleet_desc = fleet_desc

    def _set_program(self, program):
        """
        Set program.

        Args:
            program(Program): a Program object
        """
        self._program = program

    def _gen_worker_desc(self, trainer_desc):
        """
        Generator worker desc.

        Args:
            trainer_desc(TrainerDesc): a TrainerDesc object
        """
        raise NotImplementedError(
            "DeviceWorker does not implement gen_worker_desc, "
            "please use Hogwild or DownpourSGD, etc.")


class Hogwild(DeviceWorker):
    """
    Hogwild is a kind of SGD algorithm.

    """

    def __init__(self):
        """
        Init.
        """
        super(Hogwild, self).__init__()

    def _gen_worker_desc(self, trainer_desc):
        """
        Generator worker desc, which device worker is HogwildWorker.

        Args:
            trainer_desc(TrainerDesc): a TrainerDesc object
        """
        trainer_desc.device_worker_name = "HogwildWorker"
        if self._infer:
            # just ignore feed op for inference model
            trainer_desc.hogwild_param.skip_ops.extend(["feed"])


class DownpourSGD(DeviceWorker):
    """
    DownpourSGD is a kind of distributed SGD algorithm.
    """

    def __init__(self):
        """
        Init.
        initialize downpourSGD device worker
        """
        super(DownpourSGD, self).__init__()

    def _gen_worker_desc(self, trainer_desc):
        """
        Generator worker desc, which device worker is DownpourWorker.

        Args:
            trainer_desc(TrainerDesc): a TrainerDesc object
        """
        dense_table_set = set()
        program_id = str(id(self._program))
        if self._program == None:
            print("program of current device worker is not configured")
            exit(-1)
        opt_info = self._program._fleet_opt
        program_configs = opt_info["program_configs"]
        downpour = trainer_desc.downpour_param

        for pid in program_configs:
            if pid == program_id:
                pc = downpour.program_config.add()
                pc.program_id = program_id
                for i in program_configs[program_id]["push_sparse"]:
                    pc.push_sparse_table_id.extend([i])
                for i in program_configs[program_id]["push_dense"]:
                    pc.push_dense_table_id.extend([i])
                    dense_table_set.add(i)
                for i in program_configs[program_id]["pull_sparse"]:
                    pc.pull_sparse_table_id.extend([i])
                for i in program_configs[program_id]["pull_dense"]:
                    pc.pull_dense_table_id.extend([i])
                    dense_table_set.add(i)
                break

        trainer_desc.device_worker_name = "DownpourWorker"
        pull_thread = trainer_desc.pull_dense_param
        pull_thread.device_num = trainer_desc.thread_num
        for i in self._fleet_desc.trainer_param.dense_table:
            if i.table_id in dense_table_set:
                dense_table = pull_thread.dense_table.add()
                dense_table.dense_value_name.extend(i.dense_variable_name)
                dense_table.table_id = \
                    i.table_id
        sparse_table = downpour.sparse_table.add()
        sparse_table.table_id = \
                    self._fleet_desc.trainer_param.sparse_table[0].table_id
        sparse_table.sparse_key_name.extend(
            self._fleet_desc.trainer_param.sparse_table[0].slot_key)
        sparse_table.sparse_value_name.extend(
            self._fleet_desc.trainer_param.sparse_table[0].slot_value)
        sparse_table.sparse_grad_name.extend(
            self._fleet_desc.trainer_param.sparse_table[0].slot_gradient)
        sparse_table.emb_dim = \
                    self._fleet_desc.server_param.downpour_server_param.downpour_table_param[
                        0].accessor.fea_dim - 2
        sparse_table.fea_dim = sparse_table.emb_dim + 2
        # TODO(guru4elephant): hard code here, need to improve
        sparse_table.label_var_name = "click"

        for i in self._fleet_desc.trainer_param.dense_table:
            if i.table_id in dense_table_set:
                dense_table = downpour.dense_table.add()
                dense_table.table_id = i.table_id
                dense_table.dense_value_name.extend(i.dense_variable_name)
                dense_table.dense_grad_name.extend(
                    i.dense_gradient_variable_name)
                downpour.skip_ops.extend(self._fleet_desc.trainer_param.skip_op)
        if self._infer:
            downpour.push_dense = False
            downpour.push_sparse = False


class DeviceWorkerFactory(object):
    def _create_device_worker(self, worker_type):
        classname = worker_type.capitalize()
        return globals()[classname]()
