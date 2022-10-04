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
"""
TestCases for TrainerDesc,
including config, etc.
"""

import paddle.fluid as fluid
import numpy as np
import os
import shutil
import unittest


class TestTrainerDesc(unittest.TestCase):
    """  TestCases for TrainerDesc. """

    def test_config(self):
        """
        Testcase for python config.
        """
        trainer_desc = fluid.trainer_desc.TrainerDesc()
        trainer_desc._set_dump_fields(["a", "b"])
        trainer_desc._set_mpi_rank(1)
        trainer_desc._set_dump_fields_path("path")

        dump_fields = trainer_desc.proto_desc.dump_fields
        mpi_rank = trainer_desc.proto_desc.mpi_rank
        dump_fields_path = trainer_desc.proto_desc.dump_fields_path
        self.assertEqual(len(dump_fields), 2)
        self.assertEqual(dump_fields[0], "a")
        self.assertEqual(dump_fields[1], "b")
        self.assertEqual(mpi_rank, 1)
        self.assertEqual(dump_fields_path, "path")

    def test_config_dump_simple(self):
        """
        Testcase for dump_in_simple_mode
        """
        trainer_desc = fluid.trainer_desc.TrainerDesc()
        trainer_desc._set_dump_fields(["a", "b"])
        trainer_desc._set_is_dump_in_simple_mode(True)

        is_dump_in_simple_mode = trainer_desc.proto_desc.is_dump_in_simple_mode
        self.assertEqual(is_dump_in_simple_mode, 1)


if __name__ == '__main__':
    unittest.main()
