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

from __future__ import print_function
from paddle.fluid.framework import OpProtoHolder
import json

op_proto_holder = OpProtoHolder.instance()
ops = dict()
for op_type in op_proto_holder.op_proto_map:
    op_proto = op_proto_holder.get_op_proto(op_type)
    ops[op_type] = {
        'inputs': [ipt.name for ipt in op_proto.inputs],
        'outputs': [opt.name for opt in op_proto.outputs],
        'attrs': [{
            'name': attr.name,
            'type': attr.type
        } for attr in op_proto.attrs if not attr.generated]
    }
print(json.dumps(ops, indent=4, sort_keys=True))
