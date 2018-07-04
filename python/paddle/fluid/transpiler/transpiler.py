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

from paddle.fluid.proto.framework_pb2 import *
import paddle.fluid as fluid

opid_idx = 0



def _gen_opid():
    global opid_idx
    prev = opid_idx
    opid_idx += 1
    return str(prev)

def _get_global_block(program_desc):
    if len(program_desc.blocks) == 0:
        raise ValueError("program have no blocks")
    return program_desc.blocks[0]

def _get_op_outputs_argument_names(op):
    out_arg_names = []
    for o in op.outputs:
        for n in o.arguments:
            out_arg_names.append(n)
    return out_arg_names

def _get_op_inputs_argument_names(op):
    in_arg_names = []
    for i in op.inputs:
        for n in i.arguments:
            in_arg_names.append(n)
    return in_arg_names

class Transpiler:
    def __init__(self, program_desc):
        self.program_desc = program_desc
        # varname -> [0,1,2...] indicating variables with versions in
        # the ssa.
        self.var_versions = dict()

    def build_op_id(self):
        g_block = _get_global_block(self.program_desc)
        for op in g_block.ops:
            op.opid = _gen_opid()

    def _gen_var_version(self, var_desc):
        if self.var_versions.has_key(var_desc.name):
            new_version = 0
            if self.var_versions[var_desc.name]:
                new_version = self.var_versions[var_desc.name][-1] + 1
            self.var_versions[var_desc.name].append(new_version)
            return new_version
        else:
            raise ValueError("%s is not in the varmap yet!" % var_desc.name)


    def build_ssa(self):
        g_block = _get_global_block(self.program_desc)
        for var in g_block.vars:
            self.var_versions[var.name] = []
            for op in g_block.ops:
                for out_var_placeholder in op.outputs:
                    if var.name in out_var_placeholder.arguments:
                        var.upstream_opid = op.opid
                        out_var_placeholder.version = self._gen_var_version(var)

        for var in g_block.vars:
            for op in g_block.ops:
                for in_var_placeholder in op.inputs:
                    if var.name in in_var_placeholder.arguments:
                        var.downstream_opid = op.opid
                        # No write to this var
                        if len(self.var_versions[var.name]) == 0:
                            self.var_versions[var.name].append(0)
                        in_var_placeholder.version = self.var_versions[var.name][-1]
        print(self.var_versions)
    
    def resolve_war(self):
        pass
    
    def build_multi_dev(self, dev_type, num_devs):
        assert dev_type in ["CUDA", "CPU"]
        assert num_devs > 1

        g_block = _get_global_block(self.program_desc)

        # TEST: put all vars on the device, this should be done when
        # building the original program.
        #
        # Or, if we are going to use this, we may need to run every
        # opertor's infershape to determine whether the variable can
        # actually put on that place.
        for var in g_block.vars:
            var.place = "%s:%d" % (dev_type, num_devs)

        for i in xrange(num_devs):
            pass

