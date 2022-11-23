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

import unittest

import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestFakeInitOpSelectedRows(unittest.TestCase):

    def check_with_place(self, place, is_selected_rows):
        scope = core.Scope()

        out_var_name = 'Out'
        if is_selected_rows:
            out_tensor = scope.var(
                out_var_name).get_selected_rows().get_tensor()
        else:
            out_tensor = scope.var(out_var_name).get_tensor()

        var_shape = [4, 784]

        # create and run fake_init_op
        fake_init_op = Operator("fake_init", Out=out_var_name, shape=var_shape)
        fake_init_op.run(scope, place)

        self.assertEqual(var_shape, out_tensor._get_dims())

    def test_fake_init_selected_rows(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            for is_selected_rows in [True, False]:
                self.check_with_place(place, is_selected_rows)


if __name__ == "__main__":
    unittest.main()
