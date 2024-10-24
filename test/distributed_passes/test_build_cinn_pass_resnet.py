# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from dist_pass_test_base import DistPassTestBase

from paddle.distributed.passes import PassManager, new_pass


class TestBuildCINNPass(DistPassTestBase):
    def init(self):
        self.atol = 0.5
        self.rtol = 0.0

    def apply_passes(self, main_prog, startup_prog):
        pass_manager = PassManager(
            [
                new_pass("build_cinn"),
                new_pass("fuse_elewise_add_act"),
            ]
        )
        pass_manager.apply([main_prog], [startup_prog])
        print(pass_manager.names)


if __name__ == "__main__":
    unittest.main()
