# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import unittest
import numpy as np
import tarfile
import os
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.utils.fleet_barrier_util import check_all_trainers_ready
from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
import paddle.fluid.incubate.fleet.utils.utils as utils

DATA_URL = "https://fleet.bj.bcebos.com/fleet_util_data.tgz"
DATA_MD5 = "8f49f582aee0821898fb79c8b7bd8a21"
DATA_PATH = ""


def download_files():
    global DATA_PATH
    DATA_PATH = paddle.dataset.common.download(DATA_URL, "fleet_util_data",
                                               DATA_MD5)
    tar = tarfile.open(DATA_PATH, "r:gz")
    DATA_PATH = os.path.dirname(DATA_PATH)
    tar.extractall(DATA_PATH)


class TestFleetUtils(unittest.TestCase):
    def test_fleet_barrier(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=1,
            server_endpoints=['127.0.0.1'])
        fleet.init(role)
        check_all_trainers_ready("/ready_path/", 0)

    def test_parse_program_proto(self):
        global DATA_PATH
        parse_program_file_path = os.path.join(
            DATA_PATH, "fleet_util_data/pruned_model/pruned_main_program.pbtxt")
        is_text_parse_program = True
        parse_output_dir = os.path.join(DATA_PATH,
                                        "fleet_util_data/pruned_model")
        fleet_util = FleetUtil()
        fleet_util.parse_program_proto(parse_program_file_path,
                                       is_text_parse_program, parse_output_dir)
        ops_log = os.path.join(parse_output_dir, "ops.log")
        vars_log = os.path.join(parse_output_dir, "vars_all.log")
        vars_persistable = os.path.join(parse_output_dir,
                                        "vars_persistable.log")
        self.assertTrue(os.path.exists(ops_log))
        self.assertTrue(os.path.exists(vars_log))
        self.assertTrue(os.path.exists(vars_persistable))

    def test_check_vars_and_dump(self):
        global DATA_PATH

        class config:
            pass

        feed_config = config()
        feed_config.feeded_vars_names = ['concat_1.tmp_0', 'concat_2.tmp_0']
        feed_config.feeded_vars_dims = [682, 1199]
        feed_config.feeded_vars_types = [np.float32, np.float32]
        feed_config.feeded_vars_filelist = [
            os.path.join(DATA_PATH, 'fleet_util_data/pruned_model/concat_1'),
            os.path.join(DATA_PATH, 'fleet_util_data/pruned_model/concat_2')
        ]

        fetch_config = config()
        fetch_config.fetch_vars_names = ['similarity_norm.tmp_0']

        conf = config()
        conf.batch_size = 1
        conf.feed_config = feed_config
        conf.fetch_config = fetch_config
        conf.dump_model_dir = os.path.join(DATA_PATH,
                                           "fleet_util_data/pruned_model")
        conf.dump_program_filename = "pruned_main_program.pbtxt"
        conf.is_text_dump_program = True
        conf.save_params_filename = None

        fleet_util = FleetUtil()
        results = fleet_util.check_vars_and_dump(conf)
        self.assertTrue(len(results) == 1)
        np.testing.assert_array_almost_equal(
            results[0], np.array(
                [[3.0590223e-07]], dtype=np.float32))
        conf.feed_config.feeded_vars_filelist = None
        results = fleet_util.check_vars_and_dump(conf)
        self.assertTrue(len(results) == 1)

    def test_check_two_programs(self):
        global DATA_PATH

        class config:
            pass

        conf = config()
        conf.train_prog_path = os.path.join(
            DATA_PATH, "fleet_util_data/train_program/join_main_program.pbtxt")
        conf.is_text_train_program = True
        conf.pruned_prog_path = os.path.join(
            DATA_PATH, "fleet_util_data/pruned_model/pruned_main_program.pbtxt")
        conf.is_text_pruned_program = True
        conf.draw = True
        conf.draw_out_name = "pruned_check"
        fleet_util = FleetUtil()
        res = fleet_util.check_two_programs(conf)
        self.assertTrue(res)

    def test_draw_program(self):
        global DATA_PATH
        program_path = os.path.join(
            DATA_PATH, "fleet_util_data/train_program/join_main_program.pbtxt")
        is_text = True
        program = utils.load_program(program_path, is_text)
        output_dir = os.path.join(DATA_PATH, "fleet_util_data/train_program")
        output_filename_1 = "draw_prog_1"
        output_filename_2 = "draw_prog_2"
        fleet_util = FleetUtil()
        fleet_util.draw_from_program_file(program_path, is_text, output_dir,
                                          output_filename_1)
        fleet_util.draw_from_program(program, output_dir, output_filename_2)
        self.assertTrue(
            os.path.exists(
                os.path.join(output_dir, output_filename_1 + ".dot")))
        self.assertTrue(
            os.path.exists(
                os.path.join(output_dir, output_filename_1 + ".pdf")))
        self.assertTrue(
            os.path.exists(
                os.path.join(output_dir, output_filename_2 + ".dot")))
        self.assertTrue(
            os.path.exists(
                os.path.join(output_dir, output_filename_2 + ".pdf")))


if __name__ == '__main__':
    download_files()
    unittest.main()
