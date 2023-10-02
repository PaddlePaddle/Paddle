# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

from test_reduce_op_new import TestReduceAll


class TestReduceForBool(TestReduceAll):
    def init_attrs(self):
        super().init_attrs()
        self.dtypes = [{"dtype": "bool"}]
        self.attrs = [
            {"op_type": "all", "keepdim": True},
            {"op_type": "all", "keepdim": False},
            {"op_type": "any", "keepdim": True},
            {"op_type": "any", "keepdim": False},
        ]


class TestReduceAxis(TestReduceAll):
    def init_attrs(self):
        super().init_attrs()
        self.inputs = [
            {
                "shape": [1, 512, 1],
                "axis": [1],
            },
            {
                "shape": [1, 1024, 1],
                "axis": [1],
            },
            {
                "shape": [1, 2048, 1],
                "axis": [1],
            },
            {
                "shape": [64, 32, 16, 8, 4],
                "axis": [0, 2],
            },
            {
                "shape": [64, 32, 16, 8, 4],
                "axis": [1, 2, 3],
            },
            {
                # No axis, all reduce
                "shape": [64, 32, 16, 8, 4],
                "axis": [],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {
                "op_type": "sum",
                "keepdim": True,
            },
            {
                "op_type": "sum",
                "keepdim": False,
            },
        ]


if __name__ == "__main__":
    TestReduceForBool().run()
    TestReduceAxis().run()
