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

from __future__ import print_function

from .proto import pass_desc_pb2
from .core import register_pass

ALL_REGISTER_PASS = dict()


class Var(object):
    def __init__(self, name):
        self._name = name
        self._op_type = None
        self._from_var = None

    def __call__(self, name):
        var = Var(name)
        var.SetFromVar(self)
        return var

    def Name(self):
        return self._name

    def OpType(self):
        return self._op_type

    def FromVar(self):
        return self._from_var

    def SetOpType(self, op_type):
        self._op_type = op_type

    def SetFromVar(self, var):
        self._from_var = var


class Op(object):
    def __init__(self, type):
        self._type = type
        self._inputs = dict()
        self._outputs = dict()

    def __call__(self, inputs, outputs):
        for var in inputs:
            var.SetOpType(self._type)
            self._inputs[var.Name()] = var
        for var in outputs:
            var.SetOpType(self._type)
            self._outputs[var.Name()] = var
        return self

    def Type(self):
        return self._type

    def Inputs(self):
        return self._inputs.values()

    def Outputs(self):
        return self._outputs.values()


def CreateVars(*names):
    return list(map(Var, names))


def CreateOps(*names):
    return list(map(Op, names))


def CreatePassPair(pattern_ops, algebra_ops):
    def var2pb(var):
        pb = pass_desc_pb2.PassDesc.Var()
        pb.name = var.Name()
        from_var = var.FromVar()
        if isinstance(from_var, Var):
            pb.from_op_type = from_var.OpType()
            pb.from_op_var = from_var.Name()
        return pb

    def op2pb(op):
        pb = pass_desc_pb2.PassDesc.Op()
        pb.type = op.Type()
        pb.input.extend(list(map(var2pb, op.Inputs())))
        pb.output.extend(list(map(var2pb, op.Outputs())))
        return pb

    desc_pb = pass_desc_pb2.PassDesc()
    desc_pb.pattern_op.extend(list(map(op2pb, pattern_ops)))
    desc_pb.algebra_op.extend(list(map(op2pb, algebra_ops)))
    return desc_pb


def RegisterPass(func):
    pass_name = func.__name__
    ALL_REGISTER_PASS[pass_name] = func
    return func


def UsePass(pass_name):
    func = ALL_REGISTER_PASS[pass_name]
    multi_pass_desc = pass_desc_pb2.MultiPassDesc()
    multi_pass_desc.name = pass_name
    multi_pass_desc.pass_desc.extend([func()])
    register_pass(pass_name, multi_pass_desc.SerializeToString())
