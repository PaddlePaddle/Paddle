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

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.proto.framework_pb2 as framework_pb2
import numpy as np


class OpBuilder(object):
    def __init__(self, scope, place):
        self.scope = scope
        self.place = place
        self.op_desc = framework_pb2.OpDesc()

    def set_type(self, op_type):
        self.op_desc.type = op_type

    def add_input(self, name, var_name=None, value=None):
        self._add_input_or_output(name, True, var_name, value)

    def add_output(self, name, var_name=None):
        self._add_input_or_output(name, False, var_name, None)

    @staticmethod
    def _isinstancelist(value, value_type):
        if (isinstance(value, list) or isinstance(value,
                                                  tuple)) and len(value) > 0:
            for e in value:
                if not isinstance(e, valuetype):
                    return False

            return True
        else:
            return False

    @staticmethod
    def _set_attr_value(apt, value):
        if isinstance(value, bool):
            apt.type = framework_pb2.BOOLEAN
            apt.b = value
        elif isinstance(value, int):
            apt.type = framework_pb2.INT
            apt.i = value
        elif isinstance(value, long):
            apt.type = framework_pb2.LONG
            apt.l = value
        elif isinstance(value, float):
            apt.type = framework_pb2.FLOAT
            apt.f = value
        elif isinstance(value, basestring):
            apt.type = framework_pb2.STRING
            apt.s = value
        elif _isinstancelist(value, bool):
            apt.type = framework_pb2.BOOLEANS
            apt.bools = value
        elif _isinstancelist(value, int):
            apt.type = framework_pb2.INTS
            apt.ints = value
        elif _isinstancelist(value, float):
            apt.type = framework_pb2.FLOATS
            apt.floats = value
        elif _isinstancelist(value, basestring):
            apt.type = framework_pb2.STRINGS
            apt.strings = value
        else:
            raise TypeError('Unsupported attr type %s with %s' %
                            (type(value), value))

    def add_attr(self, name, value):
        apt = self.op_desc.attrs.add()
        apt.name = name
        self._set_attr_value(apt, value)

    def build(self):
        return core.Operator.create(self.op_desc.SerializeToString())

    def build_and_run(self, fetch_list=None, return_numpy=True):
        self.build().run(self.scope, self.place)
        if fetch_list is None:
            return

        ret = []
        for fetch in fetch_list:
            tensor = self.scope.find_var(fetch).get_tensor()
            ret.append(np.array(tensor) if return_numpy else tensor)

        return tuple(ret)

    def _add_input_or_output(self, name, is_input, var_name, value):
        if is_input:
            pt = self.op_desc.inputs.add()
        else:
            pt = self.op_desc.outputs.add()

        if var_name is None:
            var_name = name

        pt.parameter = name
        if isinstance(var_name, list):
            if value is None:
                value = [None] * len(var_name)

            assert len(value) == len(var_name)

            for var_name_e, value_e in zip(var_name, value):
                self._add_val(is_input, var_name_e, value_e)

            pt.arguments.extend(var_name)
        else:
            pt.arguments.append(var_name)
            self._add_val(is_input, var_name, value)

    def _add_val(self, is_input, var_name, value):
        if value is None:
            value = np.array([])

        tensor = self.scope.var(var_name).get_tensor()
        if is_input:
            tensor.set(value, self.place)
