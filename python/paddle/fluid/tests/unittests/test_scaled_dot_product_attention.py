# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

=======
from __future__ import print_function

import unittest
import numpy as np
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestScaledDotProductAttentionError(unittest.TestCase):
<<<<<<< HEAD
    def test_errors(self):
        with program_guard(Program(), Program()):
            queries = fluid.data(
                name="queries", shape=[3, 5, 9], dtype="float32"
            )
            keys = fluid.data(name="keys", shape=[3, 6, 9], dtype="float32")
            values = fluid.data(
                name="values", shape=[3, 6, 10], dtype="float32"
            )

            def test_queries_Variable():
                queries_data = np.random.rand(3, 5, 9).astype("float32")
                fluid.nets.scaled_dot_product_attention(
                    queries_data, keys, values
                )
=======

    def test_errors(self):
        with program_guard(Program(), Program()):
            queries = fluid.data(name="queries",
                                 shape=[3, 5, 9],
                                 dtype="float32")
            keys = fluid.data(name="keys", shape=[3, 6, 9], dtype="float32")
            values = fluid.data(name="values",
                                shape=[3, 6, 10],
                                dtype="float32")

            def test_queries_Variable():
                queries_data = np.random.rand(3, 5, 9).astype("float32")
                fluid.nets.scaled_dot_product_attention(queries_data, keys,
                                                        values)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_queries_Variable)

            def test_keys_Variable():
                keys_data = np.random.rand(3, 6, 9).astype("float32")
<<<<<<< HEAD
                fluid.nets.scaled_dot_product_attention(
                    queries, keys_data, values
                )
=======
                fluid.nets.scaled_dot_product_attention(queries, keys_data,
                                                        values)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_keys_Variable)

            def test_values_Variable():
                values_data = np.random.rand(3, 6, 10).astype("float32")
<<<<<<< HEAD
                fluid.nets.scaled_dot_product_attention(
                    queries, keys, values_data
                )
=======
                fluid.nets.scaled_dot_product_attention(queries, keys,
                                                        values_data)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_values_Variable)

            def test_diff_dtype():
<<<<<<< HEAD
                keys_error = fluid.data(
                    name="keys_error", shape=[3, 6, 9], dtype="float64"
                )
                values_error = fluid.data(
                    name="values_error", shape=[3, 6, 10], dtype="float64"
                )
                fluid.nets.scaled_dot_product_attention(
                    queries, keys_error, values_error
                )
=======
                keys_error = fluid.data(name="keys_error",
                                        shape=[3, 6, 9],
                                        dtype="float64")
                values_error = fluid.data(name="values_error",
                                          shape=[3, 6, 10],
                                          dtype="float64")
                fluid.nets.scaled_dot_product_attention(queries, keys_error,
                                                        values_error)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_diff_dtype)

            def test_diff_dim():
<<<<<<< HEAD
                keys_error_dim = fluid.data(
                    name="keys_error_dim", shape=[3, 6], dtype="float32"
                )
                values_error_dim = fluid.data(
                    name="values_error_dim", shape=[3], dtype="float32"
                )
                fluid.nets.scaled_dot_product_attention(
                    queries, keys_error_dim, values_error_dim
                )
=======
                keys_error_dim = fluid.data(name="keys_error_dim",
                                            shape=[3, 6],
                                            dtype="float32")
                values_error_dim = fluid.data(name="values_error_dim",
                                              shape=[3],
                                              dtype="float32")
                fluid.nets.scaled_dot_product_attention(queries, keys_error_dim,
                                                        values_error_dim)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(ValueError, test_diff_dim)

            def test_diff_hidden_size():
<<<<<<< HEAD
                queries_error_hs = fluid.data(
                    name="queries_error_hs", shape=[3, 5, 9], dtype="float32"
                )
                keys_error_hs = fluid.data(
                    name="keys_error_hs", shape=[3, 6, 10], dtype="float32"
                )
                fluid.nets.scaled_dot_product_attention(
                    queries_error_hs, keys_error_hs, values
                )
=======
                queries_error_hs = fluid.data(name="queries_error_hs",
                                              shape=[3, 5, 9],
                                              dtype="float32")
                keys_error_hs = fluid.data(name="keys_error_hs",
                                           shape=[3, 6, 10],
                                           dtype="float32")
                fluid.nets.scaled_dot_product_attention(queries_error_hs,
                                                        keys_error_hs, values)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(ValueError, test_diff_hidden_size)

            def test_diff_max_len():
<<<<<<< HEAD
                keys_error_len = fluid.data(
                    name="keys_error_len", shape=[3, 7, 9], dtype="float32"
                )
                values_error_len = fluid.data(
                    name="values_error_len", shape=[3, 6, 10], dtype="float32"
                )
                fluid.nets.scaled_dot_product_attention(
                    queries, keys_error_len, values_error_len
                )
=======
                keys_error_len = fluid.data(name="keys_error_len",
                                            shape=[3, 7, 9],
                                            dtype="float32")
                values_error_len = fluid.data(name="values_error_len",
                                              shape=[3, 6, 10],
                                              dtype="float32")
                fluid.nets.scaled_dot_product_attention(queries, keys_error_len,
                                                        values_error_len)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(ValueError, test_diff_max_len)


if __name__ == "__main__":
    unittest.main()
