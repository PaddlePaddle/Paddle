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
from paddle.fluid.framework import default_main_program
from paddle.fluid.incubate.fleet.parameter_server.ir.pserver_pass import _get_optimizer_input_shape

main_program = default_main_program()


class TestFleetPS(unittest.TestCase):

    def test_version(self):
        from paddle.fluid.incubate.fleet.parameter_server import version
        transpiler = version.is_transpiler()
        self.assertEqual(transpiler, True)

    def test_optimizer_shape(self):
        optimizers = []
        optimizers.append(("adam", "Moment1", [100, 1], [50, 1]))
        optimizers.append(("adam", "Moment2", [100, 1], [50, 1]))
        optimizers.append(("adagrad", "Moment", [100, 1], [50, 1]))
        optimizers.append(("adamax", "Moment", [100, 1], [50, 1]))
        optimizers.append(("adamax", "InfNorm", [100, 1], [50, 1]))
        optimizers.append(("momentum", "Velocity", [100, 1], [50, 1]))
        optimizers.append(("lars_momentum", "Velocity", [100, 1], [50, 1]))
        optimizers.append(("decayed_adagrad", "Moment", [100, 1], [50, 1]))
        optimizers.append(("rmsprop", "Moment", [100, 1], [50, 1]))
        optimizers.append(("rmsprop", "MeanSquare", [100, 1], [50, 1]))
        optimizers.append(("ftrl", "SquaredAccumulator", [100, 1], [50, 1]))
        optimizers.append(("ftrl", "LinearAccumulator", [100, 1], [50, 1]))

        for attrs in optimizers:
            op_type, varkey, orig_shape, param_shape = attrs
            new_shape = _get_optimizer_input_shape(op_type, varkey, orig_shape,
                                                   param_shape)
            self.assertListEqual(new_shape, param_shape)

        optimizers = []
        optimizers.append(("sgd", "", [100, 1], [50, 1]))

        for attrs in optimizers:
            op_type, varkey, orig_shape, param_shape = attrs
            new_shape = _get_optimizer_input_shape(op_type, varkey, orig_shape,
                                                   param_shape)
            self.assertListEqual(new_shape, orig_shape)

        with self.assertRaises(ValueError):
            optimizers = []
            optimizers.append(("new_opti", "", [100, 1], [50, 1]))

            for attrs in optimizers:
                op_type, varkey, orig_shape, param_shape = attrs
                _get_optimizer_input_shape(op_type, varkey, orig_shape,
                                           param_shape)


if __name__ == '__main__':
    unittest.main()
