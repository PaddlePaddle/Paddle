# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import unittest as unittest


class VarInfo(object):
    def __init__(self, var_name, var_type, writable):
        self.name = var_name
        self.type = var_type
        self.writable = writable


class TestGlobalVarGetterSetter(unittest.TestCase):
    def test_main(self):
        var_infos = [
            VarInfo("FLAGS_use_mkldnn", bool, False),
            VarInfo("FLAGS_use_ngraph", bool, False),
            VarInfo("FLAGS_eager_delete_tensor_gb", float, True),
        ]

        g = fluid.core.globals()
        for var in var_infos:
            self.assertTrue(var.name in g.keys())
            if var.writable:
                value1 = g.__getpublic__(var.name)
                value2 = g.getpublic(var.name, None)
            else:
                value1 = g.__getprivate__(var.name)
                value2 = g.getprivate(var.name, None)
            self.assertTrue(value1 is not None)
            self.assertEqual(value1, value2)
            self.assertEqual(type(value1), var.type)
            self.assertEqual(type(value2), var.type)

            if var.writable:
                g.__setpublic__(var.name) = -1
            else:
                try:
                    g.__setpublic__(var.name) = False
                    self.assertTrue(False)
                except:
                    self.assertTrue(True)

        name = "__any_non_exist_name__"
        self.assertFalse(name in g.keys())
        self.assertTrue(g.getpublic(name, None) is None)
        self.assertEquals(g.getpublic(name, -1), -1)


if __name__ == '__main__':
    unittest.main()
