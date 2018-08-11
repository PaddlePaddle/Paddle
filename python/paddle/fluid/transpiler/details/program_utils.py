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


def delete_ops(block, ops):
    try:
        start = list(block.ops).index(ops[0])
        end = list(block.ops).index(ops[-1])
        [block._remove_op(start) for _ in xrange(end - start + 1)]
    except Exception, e:
        raise e
    block.program._sync_with_cpp()


def find_op_by_input_arg(block, arg_name):
    for index, op in enumerate(block.ops):
        if arg_name in op.input_arg_names:
            return index
    return -1


def find_op_by_output_arg(block, arg_name):
    for index, op in enumerate(block.ops):
        if arg_name in op.output_arg_names:
            return index
    return -1


def get_indent_space(indent, space_num=4):
    ret = ""
    for i in range(0, indent * space_num):
        ret += " "

    return ret


def get_variable(var):
    buf = "{name} = fluid.{type}.shape{shape}.astype({dtype})".\
        format(i="{", e="}", name=var.name, type=var.type, shape=var.shape, dtype=var.dtype)
    return buf


def get_op(op):
    #buf = "{outputs}={op_type}({inputs}, attrs"
    return


def program_to_code(prog):
    indent = 0
    for block in prog.blocks:
        print "{0}{1}".format(get_indent_space(indent), '{')
        indent += 1
        # sort all vars
        all_vars = sorted(block.vars.iteritems(), key=lambda x: x[0])
        for var in all_vars:
            print "{}{}".format(get_indent_space(indent), get_variable(var[1]))
        indent -= 1
        print "{0}{1}".format(get_indent_space(indent), '}')
