#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import unittest
import sys

sys.path.append("..")
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.nn import Embedding
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Adam
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope
from paddle.fluid.executor import global_scope
import numpy as np
<<<<<<< HEAD
=======
import six
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import pickle
import os
import errno
from test_static_save_load import *

paddle.enable_static()


class TestNPUSaveLoadBase(TestSaveLoadBase):
<<<<<<< HEAD
    def set_place(self):
        return (
            fluid.CPUPlace()
            if not core.is_compiled_with_npu()
            else paddle.NPUPlace(0)
        )


class TestNPUSaveLoadPartial(TestSaveLoadPartial):
    def set_place(self):
        return (
            fluid.CPUPlace()
            if not core.is_compiled_with_npu()
            else paddle.NPUPlace(0)
        )


class TestNPUSaveLoadSetStateDict(TestSaveLoadSetStateDict):
    def set_place(self):
        return (
            fluid.CPUPlace()
            if not core.is_compiled_with_npu()
            else paddle.NPUPlace(0)
        )


class TestNPUProgramStatePartial(TestProgramStatePartial):
    def set_place(self):
        return (
            fluid.CPUPlace()
            if not core.is_compiled_with_npu()
            else paddle.NPUPlace(0)
        )


class TestNPULoadFromOldInterface(TestLoadFromOldInterface):
    def set_place(self):
        return (
            fluid.CPUPlace()
            if not core.is_compiled_with_npu()
            else paddle.NPUPlace(0)
        )


class TestNPULoadFromOldInterfaceSingleFile(TestLoadFromOldInterfaceSingleFile):
    def set_place(self):
        return (
            fluid.CPUPlace()
            if not core.is_compiled_with_npu()
            else paddle.NPUPlace(0)
        )


class TestNPUProgramStateOldSave(TestProgramStateOldSave):
=======

    def set_place(self):
        return fluid.CPUPlace(
        ) if not core.is_compiled_with_npu() else paddle.NPUPlace(0)


class TestNPUSaveLoadPartial(TestSaveLoadPartial):

    def set_place(self):
        return fluid.CPUPlace(
        ) if not core.is_compiled_with_npu() else paddle.NPUPlace(0)


class TestNPUSaveLoadSetStateDict(TestSaveLoadSetStateDict):

    def set_place(self):
        return fluid.CPUPlace(
        ) if not core.is_compiled_with_npu() else paddle.NPUPlace(0)


class TestNPUProgramStatePartial(TestProgramStatePartial):

    def set_place(self):
        return fluid.CPUPlace(
        ) if not core.is_compiled_with_npu() else paddle.NPUPlace(0)


class TestNPULoadFromOldInterface(TestLoadFromOldInterface):

    def set_place(self):
        return fluid.CPUPlace(
        ) if not core.is_compiled_with_npu() else paddle.NPUPlace(0)


class TestNPULoadFromOldInterfaceSingleFile(TestLoadFromOldInterfaceSingleFile):

    def set_place(self):
        return fluid.CPUPlace(
        ) if not core.is_compiled_with_npu() else paddle.NPUPlace(0)


class TestNPUProgramStateOldSave(TestProgramStateOldSave):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.test_dygraph = False

    def set_place(self):
<<<<<<< HEAD
        return (
            fluid.CPUPlace()
            if not core.is_compiled_with_npu()
            else paddle.NPUPlace(0)
        )


class TestNPUProgramStateOldSaveSingleModel(TestProgramStateOldSaveSingleModel):
    def set_place(self):
        return (
            fluid.CPUPlace()
            if not core.is_compiled_with_npu()
            else paddle.NPUPlace(0)
        )
=======
        return fluid.CPUPlace(
        ) if not core.is_compiled_with_npu() else paddle.NPUPlace(0)


class TestNPUProgramStateOldSaveSingleModel(TestProgramStateOldSaveSingleModel):

    def set_place(self):
        return fluid.CPUPlace(
        ) if not core.is_compiled_with_npu() else paddle.NPUPlace(0)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
