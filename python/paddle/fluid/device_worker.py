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


class DeviceWorker(object):
    def __init__(self):
        pass

    def gen_worker_desc(self, trainer_desc, fleet_desc):
        pass


class Hogwild(DeviceWorker):
    def __init__(self):
        super(Hogwild, self).__init__()

    def gen_worker_desc(self, trainer_desc, fleet_desc):
        trainer_desc.device_worker_name = "HogwildWorker"


class Downpour(DeviceWorker):
    def __init__(self):
        super(Downpour, self).__init__()

    def gen_worker_desc(self, trainer_desc, fleet_desc):
        trainer_desc.device_worker_name = "DownpourWorker"
        pull_thread = trainer_desc.pull_dense_param
        pull_thread.device_num = trainer_desc.thread_num
        dense_table = pull_thread.dense_table.add()
        dense_table.dense_value_name.extend(
            fleet_desc.trainer_param.dense_table[0].dense_variable_name)
        dense_table.table_id = \
                    fleet_desc.trainer_param.dense_table[0].table_id
        downpour = trainer_desc.downpour_param
        sparse_table = downpour.sparse_table.add()
        sparse_table.table_id = \
                    fleet_desc.trainer_param.sparse_table[0].table_id
        sparse_table.sparse_key_name.extend(
            fleet_desc.trainer_param.sparse_table[0].slot_key)
        sparse_table.sparse_value_name.extend(
            fleet_desc.trainer_param.sparse_table[0].slot_value)
        sparse_table.sparse_grad_name.extend(
            fleet_desc.trainer_param.sparse_table[0].slot_gradient)
        sparse_table.emb_dim = fleet_desc.server_param.downpour_server_param.downpour_table_param[
            0].accessor.fea_dim - 2
        sparse_table.fea_dim = sparse_table.emb_dim + 2
        sparse_table.label_var_name = "click"

        dense_table = downpour.dense_table.add()
        dense_table.table_id = \
                    fleet_desc.trainer_param.dense_table[0].table_id
        dense_table.dense_value_name.extend(
            fleet_desc.trainer_param.dense_table[0].dense_variable_name)
        dense_table.dense_grad_name.extend(fleet_desc.trainer_param.dense_table[
            0].dense_gradient_variable_name)
        downpour.skip_ops.extend(fleet_desc.trainer_param.skip_op)


class DeviceWorkerFactory(object):
    def create_device_worker(self, worker_type):
        classname = worker_type.capitalize()
        print("------------")
        print(classname)
        return globals()[classname]()
