#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Test fleet."""

from __future__ import print_function
import os
import unittest
import paddle.fluid.incubate.fleet.base.role_maker as role_maker


class TestFleet2(unittest.TestCase):
    """Test cases for fleet ops."""

    def setUp(self):
        """Set up, set envs."""
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001,127.0.0.2:36001"

    def test_pslib_1(self):
        """Test cases for pslib."""
        import paddle.fluid as fluid
        from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
        from paddle.fluid.incubate.fleet.parameter_server.pslib import \
            fleet_embedding, _prepare_params, _fleet_embedding, \
            _fleet_embedding_v2, FLEET_GLOBAL_DICT
        from paddle.fluid.incubate.fleet.base.role_maker import GeneralRoleMaker
        try:
            import netifaces
        except:
            print("warning: no netifaces, skip test_pslib_1")
            return
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36002"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        role_maker = GeneralRoleMaker()
        role_maker.generate_role()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        fleet.init(role_maker)
        train_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        global FLEET_GLOBAL_DICT
        with fluid.program_guard(train_program, startup_program):
            show = fluid.layers.data(name="show", shape=[-1, 1], \
                dtype="int64", lod_level=1, append_batch_size=False)
            click = fluid.layers.data(name="click", shape=[-1, 1], \
                dtype="int64", lod_level=1, append_batch_size=False)
            with fleet_embedding(click_name=click.name):
                emb = fluid.layers.embedding(input=show, size=[1, 1], \
                    is_sparse=True, is_distributed=True, \
                    param_attr=fluid.ParamAttr(name="embedding"))
            emb = fluid.layers.data_norm(
                input=emb,
                name="a",
                epsilon=1e-4,
                param_attr={
                    "batch_size": 1e4,
                    "batch_sum_default": 0.0,
                    "batch_square": 1e4
                })
            fc = fluid.layers.fc(input=emb, size=1, act=None)
            label = fluid.layers.data(name="click", shape=[-1, 1], \
                dtype="int64", lod_level=1, append_batch_size=False)
            label_cast = fluid.layers.cast(label, dtype='float32')
            cost = fluid.layers.log_loss(fc, label_cast)
        try:
            adam = fluid.optimizer.Adam(learning_rate=0.000005)
            adam = fleet.distributed_optimizer(
                adam,
                strategy={
                    "embedding": {
                        "sparse_accessor_class": "DownpourSparseValueAccessor"
                    }
                })
            adam.minimize([cost], [scope])
        except:
            print("do not support pslib test, skip")
            return
        FLEET_GLOBAL_DICT["cur_accessor"] = "DownpourCtrAccessor"
        try:
            _prepare_params(input=show, size=[1, 1])
        except:
            print("catch expected exception of param_attr=None")
        try:
            _prepare_params(
                input=show, size=[1, 1], param_attr=fluid.ParamAttr())
        except:
            print("catch expected exception of name=None")
        try:
            tmp = fluid.ParamAttr(name="embedding")
            _prepare_params(input=show, size=1, param_attr=tmp)
        except:
            print("catch expected exception of size not list")
        try:
            tmp = fluid.ParamAttr(name="embedding")
            _prepare_params(input=show, size=[-1, 12], param_attr=tmp)
        except:
            print("catch expected exception of size not equal")
        try:
            tmp = fluid.ParamAttr(name="embedding")
            _prepare_params(
                input=show, size=[-1, 1], param_attr=tmp, is_sparse=False)
        except:
            print("catch expected exception of is_sparse=False")
        try:
            tmp = fluid.ParamAttr(name="embedding")
            _prepare_params(input=show, size=[-1, 1], param_attr=tmp, \
                            is_sparse=True, is_distributed=False)
        except:
            print("catch expected exception of is_distributed=False")
        try:
            _prepare_params(input=show, size=[-1, 1], \
                            param_attr=fluid.ParamAttr(name="embedding"), \
                            is_sparse=True, is_distributed=True, dtype="abc")
        except:
            print("catch expected exception of unknown dtype")
        try:
            FLEET_GLOBAL_DICT["emb_to_accessor"]["embedding"] = "unknown"
            tmp = fluid.ParamAttr(name="embedding")
            _prepare_params(input=show, size=[-1, 1], param_attr=tmp)
        except:
            print("catch expected exception of unknown accessor")
        FLEET_GLOBAL_DICT["cur_accessor"] = "DownpourCtrAccessor"
        try:
            _fleet_embedding(input=show, size=[-1, 1], is_sparse=True, \
                             is_distributed=True, dtype="float32", \
                             param_attr=fluid.ParamAttr(name="embedding"))
        except:
            print("catch expected exception of unknown accessor")
        try:
            _fleet_embedding_v2(input=show, size=[-1, 1], is_sparse=True, \
                                is_distributed=True, dtype="float32", \
                                param_attr=fluid.ParamAttr(name="embedding"))
        except:
            print("catch expected exception of unknown accessor")

        adam1 = fluid.optimizer.Adam(learning_rate=0.000005)
        adam1 = fleet.distributed_optimizer(
            adam1,
            strategy={
                "embedding": {
                    "sparse_accessor_class": "DownpourSparseValueAccessor"
                }
            })
        try:
            pre = FLEET_GLOBAL_DICT["emb_to_table"]
            FLEET_GLOBAL_DICT["emb_to_table"] = {}
            adam1.minimize([cost], [scope])
        except:
            FLEET_GLOBAL_DICT["emb_to_table"] = pre
            print("catch expected exception of empty emb_to_table")
        try:
            pre = FLEET_GLOBAL_DICT["emb_to_table"]
            FLEET_GLOBAL_DICT["emb_to_table"] = {}
            FLEET_GLOBAL_DICT["emb_to_table"]["emb1"] = 0
            adam1.minimize([cost], [scope])
        except:
            FLEET_GLOBAL_DICT["emb_to_table"] = pre
            print("catch expected exception of error emb_to_table")
        try:
            adam2 = fluid.optimizer.Adam(learning_rate=0.000005)
            adam2 = fleet.distributed_optimizer(adam2)
            adam2.supported_embedding_types = []
            adam2.minimize([cost], [scope])
        except:
            print("catch expected exception of embedding_types")
        try:
            adam3 = fluid.optimizer.Adam(learning_rate=0.000005)
            adam3 = fleet.distributed_optimizer(
                adam3,
                strategy={
                    "embedding": {
                        "sparse_accessor_class": "DownpourSparseValueAccessor",
                        "sparse_embedx_dim": 999
                    }
                })
            adam3.minimize([cost], [scope])
        except:
            print("catch expected exception of embedx_dim error")

        try:
            adam4 = fluid.optimizer.Adam(learning_rate=0.000005)
            adam4 = fleet.distributed_optimizer(
                adam4,
                strategy={
                    "embedding": {
                        "sparse_accessor_class": "DownpourCtrAccessor",
                        "sparse_embedx_dim": 999
                    }
                })
            adam4.minimize([cost], [scope])
        except:
            print("catch expected exception of embedx_dim error")
        train_program1 = fluid.Program()
        startup_program1 = fluid.Program()
        FLEET_GLOBAL_DICT["emb_to_accessor"] = {}
        with fluid.program_guard(train_program1, startup_program1):
            show = fluid.layers.data(name="show", shape=[-1, 1], \
                dtype="int64", lod_level=1, append_batch_size=False)
            with fleet_embedding(click_name=click.name):
                emb = fluid.layers.embedding(input=show, size=[1, 1], \
                    is_sparse=True, is_distributed=True, \
                    param_attr=fluid.ParamAttr(name="embedding"))
            with fleet_embedding(click_name=click.name):
                emb1 = fluid.embedding(input=show, size=[1, 1], \
                    is_sparse=True, is_distributed=True, \
                    param_attr=fluid.ParamAttr(name="embedding"))
        fleet.save_model("./tmodel_000")
        fleet.save_one_table(0, "./tmodel_001")
        fleet.save_one_table(0, "./tmodel_002", prefix="thahaha")
        fleet.load_model("./tmodel_0003")
        fleet.load_one_table(0, "./tmodel_004")


if __name__ == "__main__":
    unittest.main()
