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

import unittest
import numpy as np
import tarfile
import tempfile
import os
import sys
from paddle.dataset.common import download
from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
import paddle.fluid.incubate.fleet.utils.utils as utils


class TestFleetUtils(unittest.TestCase):
    proto_data_url = "https://fleet.bj.bcebos.com/fleet_util_data.tgz"
    proto_data_md5 = "59b7f12fd9dc24b64ae8e4629523a92a"
    module_name = "fleet_util_data"
    pruned_dir = os.path.join("fleet_util_data", "pruned_model")
    train_dir = os.path.join("fleet_util_data", "train_program")

    def download_files(self):
        path = download(self.proto_data_url, self.module_name,
                        self.proto_data_md5)
        print('data is downloaded at ' + path)
        tar = tarfile.open(path)
        unzip_folder = tempfile.mkdtemp()
        tar.extractall(unzip_folder)
        return unzip_folder

    def test_fleet_util_init(self):
        fleet_util_pslib = FleetUtil()
        fleet_util_transpiler = FleetUtil(mode="transpiler")
        self.assertRaises(Exception, FleetUtil, "other")

    def test_program_type_trans(self):
        data_dir = self.download_files()
        program_dir = os.path.join(data_dir, self.pruned_dir)
        text_program = "pruned_main_program.pbtxt"
        binary_program = "pruned_main_program.bin"
        fleet_util = FleetUtil()
        text_to_binary = fleet_util.program_type_trans(program_dir,
                                                       text_program, True)
        binary_to_text = fleet_util.program_type_trans(program_dir,
                                                       binary_program, False)
        self.assertTrue(
            os.path.exists(os.path.join(program_dir, text_to_binary)))
        self.assertTrue(
            os.path.exists(os.path.join(program_dir, binary_to_text)))

    def test_parse_program_proto(self):
        data_dir = self.download_files()
        parse_program_file_path = os.path.join(
            data_dir, os.path.join(self.pruned_dir,
                                   "pruned_main_program.pbtxt"))
        is_text_parse_program = True
        parse_output_dir = os.path.join(data_dir, self.pruned_dir)
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
        data_dir = self.download_files()

        class config:
            pass

        feed_config = config()
        feed_config.feeded_vars_names = ['concat_1.tmp_0', 'concat_2.tmp_0']
        feed_config.feeded_vars_dims = [682, 1199]
        feed_config.feeded_vars_types = [np.float32, np.float32]
        feed_config.feeded_vars_filelist = [
            os.path.join(data_dir, os.path.join(self.pruned_dir, "concat_1")),
            os.path.join(data_dir, os.path.join(self.pruned_dir, "concat_2"))
        ]

        fetch_config = config()
        fetch_config.fetch_vars_names = ['similarity_norm.tmp_0']

        conf = config()
        conf.batch_size = 1
        conf.feed_config = feed_config
        conf.fetch_config = fetch_config
        conf.dump_model_dir = os.path.join(data_dir, self.pruned_dir)
        conf.dump_program_filename = "pruned_main_program.pbtxt"
        conf.is_text_dump_program = True
        conf.save_params_filename = None

        fleet_util = FleetUtil()
        # test saved var's shape
        conf.dump_program_filename = "pruned_main_program.save_var_shape_not_match"
        self.assertRaises(Exception, fleet_util.check_vars_and_dump, conf)

        # test program.proto without feed_op and fetch_op
        conf.dump_program_filename = "pruned_main_program.no_feed_fetch"
        results = fleet_util.check_vars_and_dump(conf)
        self.assertTrue(len(results) == 1)
        np.testing.assert_array_almost_equal(
            results[0], np.array([[3.0590223e-07]], dtype=np.float32))

        # test feed_var's shape
        conf.dump_program_filename = "pruned_main_program.feed_var_shape_not_match"
        self.assertRaises(Exception, fleet_util.check_vars_and_dump, conf)

        # test correct case with feed_vars_filelist
        conf.dump_program_filename = "pruned_main_program.pbtxt"
        results = fleet_util.check_vars_and_dump(conf)
        self.assertTrue(len(results) == 1)
        np.testing.assert_array_almost_equal(
            results[0], np.array([[3.0590223e-07]], dtype=np.float32))

        # test correct case without feed_vars_filelist
        conf.feed_config.feeded_vars_filelist = None
        # test feed var with lod_level >= 2
        conf.dump_program_filename = "pruned_main_program.feed_lod2"
        self.assertRaises(Exception, fleet_util.check_vars_and_dump, conf)

        conf.dump_program_filename = "pruned_main_program.pbtxt"
        results = fleet_util.check_vars_and_dump(conf)
        self.assertTrue(len(results) == 1)

    def test_check_two_programs(self):
        data_dir = self.download_files()

        class config:
            pass

        conf = config()
        conf.train_prog_path = os.path.join(
            data_dir, os.path.join(self.train_dir, "join_main_program.pbtxt"))
        conf.is_text_train_program = True

        # test not match
        conf.pruned_prog_path = os.path.join(
            data_dir,
            os.path.join(self.pruned_dir,
                         "pruned_main_program.save_var_shape_not_match"))
        conf.is_text_pruned_program = True
        conf.draw = False
        fleet_util = FleetUtil()
        res = fleet_util.check_two_programs(conf)
        self.assertFalse(res)

        # test match
        conf.pruned_prog_path = os.path.join(
            data_dir, os.path.join(self.pruned_dir,
                                   "pruned_main_program.pbtxt"))
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            conf.draw = False
        else:
            conf.draw = True
            conf.draw_out_name = "pruned_check"
        res = fleet_util.check_two_programs(conf)
        self.assertTrue(res)

    def test_draw_program(self):
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            pass
        else:
            data_dir = self.download_files()
            program_path = os.path.join(
                data_dir, os.path.join(self.train_dir,
                                       "join_main_program.pbtxt"))
            is_text = True
            program = utils.load_program(program_path, is_text)
            output_dir = os.path.join(data_dir, self.train_dir)
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
    unittest.main()
