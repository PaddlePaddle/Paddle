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

LOOKUP_TABLE_TYPE = "lookup_table"


def find_distributed_lookup_table(program):
    # process lookup_table_op
    # 1. check all lookup_table_op is distributed
    # 2. check all lookup_table_op share the same table.
    distributed_lookup_table_ops = []
    # support only one distributed_lookup_table now
    table_name = None

    for op in program.global_block().ops:
        if op.type == LOOKUP_TABLE_TYPE:
            if op.attr('is_distributed') is True:
                if table_name is None:
                    table_name = op.input("W")[0]
                if table_name != op.input("W")[0]:
                    raise RuntimeError("all distributed lookup_table_ops"
                                       " should have only one table")
                distributed_lookup_table_ops.append(op)
            else:
                if table_name is not None:
                    assert op.input("W")[0] != table_name

    return table_name
