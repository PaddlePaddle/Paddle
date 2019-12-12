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
"""
TestCases for fleet,
including config, train, etc.
"""
from __future__ import print_function
import paddle.fluid as fluid
import paddle.compat as cpt
import paddle.fluid.core as core
import numpy as np
import os
import shutil
import unittest
import time


class TestPSlib(unittest.TestCase):
    """  TestCases for Fleet. """

    def setUp(self):
        pass

    def test_fleet3(self):
        """
        Testcase for Fleet.
        """
        try:
            import mpi4py
            import mpi4py.rc
            mpi4py.rc.finalize = False
        except ImportError as e:
            print("warning: no mpi4py, skip pslib test")
            return

        with open("test_fleet3_a.txt", "w") as f:
            data = "1 1 1 1 1 1 1 1 2 3 3 4 5 5 5 5 1 1\n"
            data += "1 1 1 1 1 1 1 2 2 3 4 4 6 6 6 6 1 2\n"
            data += "1 1 1 1 1 1 1 3 2 3 5 4 7 7 7 7 1 3\n"
            f.write(data)
        with open("test_fleet3_b.txt", "w") as f:
            data = "1 1 1 1 1 1 1 4 2 3 3 4 5 5 5 5 1 4\n"
            data += "1 1 1 1 1 0 1 5 2 3 4 4 6 6 6 6 1 5\n"
            data += "1 1 1 1 1 1 1 6 2 3 5 4 7 7 7 7 1 6\n"
            data += "1 1 1 1 1 0 1 7 2 3 6 4 8 8 8 8 1 7\n"
            f.write(data)

        train_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
        with fluid.program_guard(train_program, startup_program):
            show = fluid.layers.data(name="show", shape=[-1, 1], \
                dtype="int64", lod_level=1, append_batch_size=False)
            label = fluid.layers.data(name="click", shape=[-1, 1], \
                dtype="int64", lod_level=1, append_batch_size=False)
            ins_weight = fluid.layers.data(name="0", shape=[-1, 1], \
                dtype="float32", lod_level=1, append_batch_size=False)
            slots = ["1", "2", "3", "4"]
            slots_vars = [show, label, ins_weight]
            emb_vars = []
            bow_vars = []
            for slot in slots:
                var = fluid.layers.data(\
                    name=slot, shape=[1], dtype="int64", lod_level=1)
                emb = fluid.layers.embedding(input=var, size=[1, 11], \
                    is_sparse=True, is_distributed=True, \
                    param_attr=fluid.ParamAttr(name="embedding"))
                emb1 = fluid.layers.embedding(input=var, size=[1, 11], \
                    is_sparse=True, is_distributed=True, \
                    param_attr=fluid.ParamAttr(name="embedding1"))
                emb1.stop_gradient = True
                slots_vars.append(var)
                bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
                emb_vars.append(emb)
                bow_vars.append(bow)
            concat = fluid.layers.concat(bow_vars, axis=1)
            bn = fluid.layers.data_norm(input=concat, name="bn6048", \
                epsilon=1e-4, param_attr={ "batch_size":1e4, \
                "batch_sum_default":0.0, "batch_square":1e4})
            fc = fluid.layers.fc(input=bn, size=1, act=None)
            fc1 = fluid.layers.fc(input=bn, size=1, act=None)
            similarity_norm = fluid.layers.sigmoid(\
                fluid.layers.clip(fc, min=-15.0, max=15.0))
            cost = fluid.layers.log_loss(input=similarity_norm, \
                label=fluid.layers.cast(x=label, dtype='float32'))
            cost = fluid.layers.elementwise_mul(cost, ins_weight)
            avg_cost = fluid.layers.mean(cost)
            binary_predict = fluid.layers.concat(\
                input=[fluid.layers.elementwise_sub(fluid.layers.ceil(\
                similarity_norm), similarity_norm), similarity_norm], \
                axis=1)
            auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, \
                stat_neg] = fluid.layers.auc(input=binary_predict,\
                label=label, curve='ROC', num_thresholds=4096)
            sqrerr, abserr, prob, q, pos, total = \
                fluid.contrib.layers.ctr_metric_bundle(similarity_norm, \
                fluid.layers.cast(x=label, dtype='float32'))

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        fleet.init(exe)
        adjust_ins_weight = {
            "need_adjust": True,
            "nid_slot": emb_vars[0].name,
            "nid_adjw_threshold": 1000,
            "nid_adjw_ratio": 20,
            "ins_weight_slot": ins_weight.name
        }
        copy_table = {
            "need_copy": True,
            "batch_num": 1,
            "src_sparse_tables": [1],
            "dest_sparse_tables": [0],
            "src_var_list": ["fc_0.w_0"],
            "dest_var_list": ["fc_1.w_0"]
        }
        thread_stat_var_names = [stat_pos.name, stat_neg.name, sqrerr.name, \
            abserr.name, prob.name, q.name, pos.name, total.name]

        adam = fluid.optimizer.Adam(learning_rate=0.000005)
        adam = fleet.distributed_optimizer(adam, strategy={
            "use_cvm" : True,
            "adjust_ins_weight" : adjust_ins_weight,
            "scale_datanorm" : 1e-4,
            "dump_slot": True,
            "stat_var_names": thread_stat_var_names,
            "dump_fields": ["click"],
            "dump_fields_path": "./fleet_dump_fields_3",
            "dump_param": ["fc_0.b_0"],
            "check_nan_var_names": ["click"],
            "copy_table" : copy_table,
            "embedding" : { "sparse_shard_num": 1 },
            "embedding1" : { "sparse_shard_num": 1 }
        })
        adam.minimize([avg_cost], [scope])

        if fleet.is_server():
            fleet.run_server()
        elif fleet.is_worker():
            with fluid.scope_guard(scope):
                exe.run(startup_program)
            fleet.init_worker()
            dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
            dataset.set_batch_size(1)
            dataset.set_thread(3)
            dataset.set_filelist([
                "test_fleet3_a.txt",
                "test_fleet3_b.txt"
            ] * 300)
            dataset.set_pipe_command("cat")
            dataset.set_use_var(slots_vars)
            dataset.load_into_memory()

            with fluid.scope_guard(scope):
                exe.train_from_dataset(train_program, dataset, scope,\
                    thread=3, debug=True, \
                    fetch_list=[train_program.global_block().var(i) \
                               for i in ["click"]],
                    fetch_info=["click"],
                    print_period=1)

            prog_id = str(id(train_program))
            tables = fleet._opt_info["program_id_to_worker"][prog_id].\
            	get_desc().dense_table
            for table in tables:
                var_name_list = []
                for i in range(0, len(table.dense_variable_name)):
                    var_name = table.dense_variable_name[i]
                    var_name_list.append(var_name)
                fleet._fleet_ptr.pull_dense(scope, int(table.table_id),
                                            var_name_list)

            fleet.print_table_stat(0)
            fleet.save_persistables(exe, "./fleet_model_3")
            fleet.shrink_sparse_table()
            fleet.shrink_dense_table(0.98, scope=scope)
            time.sleep(1)
            fleet.load_one_table(0, "./fleet_model_3")
            time.sleep(1)
            fleet._fleet_ptr.copy_table(0, 1)
            time.sleep(1)
            fleet.clear_model()

            os.remove("./test_fleet3_a.txt")
            os.remove("./test_fleet3_b.txt")

        fleet._role_maker._barrier_all()


if __name__ == '__main__':
    unittest.main()
