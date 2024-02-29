# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import count_fluid_operator
import yaml

from paddle.base.framework import (
    OpProtoHolder,
)

if __name__ == '__main__':
    op_names = count_fluid_operator.get_all_old_ir_ops()
    op_names = set(op_names)

    print("op_name length")
    print(len(op_names))

    deleted = set(
        count_fluid_operator.read_txt_file(
            r"/home/aistudio/test/Paddle/tools/count_op/old_ir_ops/deleted.txt"
        )
    )

    excel_all_ops_path = r"/home/aistudio/test/Paddle/tools/count_op/old_ir_ops/excel_all_old_ir_ops.txt"
    excel_all_ops = set(count_fluid_operator.read_txt_file(excel_all_ops_path))

    all_ops = op_names.union(excel_all_ops)

    print("源码统计到的op 和excel中给出的op")
    print(len(all_ops))

    all_ops = all_ops - deleted

    print("已经删除的op")
    print(len(deleted))

    print("去掉已经删除了的op")
    print(len(all_ops))

    registed_op = []
    not_registed_op = []
    for i in all_ops:
        try:
            op_proto = OpProtoHolder.instance().get_op_proto(i)
            registed_op.append(i)
        except ValueError:
            print("算子 : " + i + " 没有被注册!")
            not_registed_op.append(i)
            continue

    print("registed_op: ")
    print(registed_op)
    print("not_registed_op")
    print(not_registed_op)

    inplace_infos = count_fluid_operator.get_inplace_info(
        r"/home/aistudio/test/Paddle/tools/count_op/old_ir_ops/inplace_info.txt"
    )
    all_fluid_op_infos = []
    for op in registed_op:
        op_name = op
        input_types = []
        input_names = []
        optional_names = []
        attr_names = []
        attr_types = []
        output_names = []
        output_types = []

        inplaces = ""

        op_proto = OpProtoHolder.instance().get_op_proto(op_name)
        for input in op_proto.inputs:
            input_names.append(input.name)
            if input.dispensable:
                optional_names.append(input.name)
            if input.duplicable:
                input_types.append("Tensor[]")
            else:
                if "TensorArray" in input.comment:
                    input_types.append("Tensor|TensorArray?")
                else:
                    input_types.append("Tensor")

        for attr in op_proto.attrs:
            attr_names.append(attr.name)
            attr_types.append(
                count_fluid_operator.get_attr_type_string(attr.type)
            )

        for output in op_proto.outputs:
            output_names.append(output.name)
            if output.dispensable:
                optional_names.append(output.name)
            if output.duplicable:
                output_types.append("Tensor[]")
            else:
                if "TensorArray" in output.comment:
                    output_types.append("Tensor|TensorArray?")
                else:
                    output_types.append("Tensor")

        data = count_fluid_operator.convert_opinfo_to_dict(
            op_name,
            input_types,
            input_names,
            attr_types,
            attr_names,
            output_types,
            output_names,
            optional_names,
            inplace_infos[op_name],
        )

        all_fluid_op_infos.append(data)

        # if len(inplace_infos[op_name]) > 0:
        #     data = convert_opinfo_to_dict( op_name + "_",input_types,input_names, attr_types,attr_names,output_types , output_names,optional_names,inplace_infos[op_name],)
        #     all_fluid_op_infos.append(data)

    print(all_fluid_op_infos)
    save_path = (
        r"/home/aistudio/test/Paddle/tools/count_op/old_ir_ops/old_ops.yaml"
    )
    with open(save_path, 'w') as file:
        for op_info in all_fluid_op_infos:
            temp = [op_info]
            yaml.dump(
                temp,
                file,
                default_flow_style=False,
                sort_keys=False,
                indent=1,
            )
            file.write("\n")
