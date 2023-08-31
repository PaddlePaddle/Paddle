# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import re

import yaml
from op_gen import OpCompatParser, OpInfoParser, to_pascal_case

H_FILE_TEMPLATE = """

#pragma once

#include <vector>

#include "paddle/ir/core/value.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_manual_api.h"

{body}

"""

CPP_FILE_TEMPLATE = """

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_api.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/api_builder.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_op.h"

{body}

"""


NAMESPACE_TEMPLATE = """
namespace {namespace} {{
{body}
}} // namespace {namespace}
"""


API_DECLARE_TEMPLATE = """
{ret_type} {api_name}({args});
"""


API_IMPL_TEMPLATE = """
{ret_type} {api_name}({args}){{
    {in_combine}
    {compute_op}
    {out_split}
    {return_result}
}}

"""

COMBINE_OP_TEMPLATE = """
    auto {op_name} = APIBuilder::Instance().GetBuilder()->Build<ir::CombineOp>({in_name});"""

SPLIT_OP_TEMPLATE = """
    auto {op_name} = APIBuilder::Instance().GetBuilder()->Build<ir::SplitOp>({in_name});"""

COMPUTE_OP_TEMPLATE = """
    paddle::dialect::{op_class_name} {op_inst_name} = APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::{op_class_name}>({args});"""

OP_RESULT = 'ir::OpResult'
VECTOR_TYPE = 'ir::VectorType'
PD_MANUAL_OP_LIST = ['add_n']


def get_op_class_name(op_name):
    return to_pascal_case(op_name) + 'Op'


class CodeGen:
    def __init__(self) -> None:
        self._type_map = {
            'paddle::dialect::DenseTensorType': 'ir::OpResult',
            'paddle::dialect::SelectedRowsType': 'ir::OpResult',
            'ir::VectorType<paddle::dialect::DenseTensorType>': 'std::vector<ir::OpResult>',
        }

    def _parse_yaml(self, op_yaml_files, op_compat_yaml_file):
        op_compat_parser = OpCompatParser(op_compat_yaml_file)

        op_yaml_items = []
        for yaml_file in op_yaml_files:
            with open(yaml_file, "r") as f:
                ops = yaml.safe_load(f)
                op_yaml_items = op_yaml_items + ops
        op_info_items = []
        for op in op_yaml_items:
            op_info_items.append(
                OpInfoParser(op, op_compat_parser.get_compat(op['name']))
            )
        return op_info_items

    # =====================================
    # Gen declare functions
    # =====================================
    def _gen_api_inputs(self, op_info):
        name_list = op_info.input_name_list
        type_list = op_info.input_type_list
        assert len(name_list) == len(type_list)
        ret = []
        for name, type in zip(name_list, type_list):
            ret.append(f'{self._type_map[type]} {name}')
        return ', '.join(ret)

    def _gen_api_attrs(self, op_info, with_default, is_mutable_attr):
        name_list = op_info.attribute_name_list
        type_list = op_info.attribute_build_arg_type_list
        default_value_list = op_info.attribute_default_value_list
        mutable_name_list = op_info.mutable_attribute_name_list
        assert len(name_list) == len(type_list) == len(default_value_list)
        no_mutable_attr = []
        mutable_attr = []
        for name, type, default_value in zip(
            name_list, type_list, default_value_list
        ):
            if is_mutable_attr and name in mutable_name_list:
                mutable_attr.append(f'{OP_RESULT} {name}')
                continue
            if with_default and default_value is not None:
                if type in ['float', 'double']:
                    default_value = default_value.strip('"')
                no_mutable_attr.append(
                    '{type} {name} = {default_value}'.format(
                        type=type, name=name, default_value=default_value
                    )
                )
            else:
                no_mutable_attr.append(f'{type} {name}')
        return ', '.join(mutable_attr + no_mutable_attr)

    def _gen_api_args(self, op_info, with_default_attr, is_mutable_attr):
        inputs = self._gen_api_inputs(op_info)
        attrs = self._gen_api_attrs(op_info, with_default_attr, is_mutable_attr)
        return (inputs + ', ' + attrs).strip(', ')

    def _gen_ret_type(self, op_info):
        type_list = op_info.output_type_list
        if len(type_list) > 1:
            return 'std::tuple<{}>'.format(
                ', '.join([self._type_map[type] for type in type_list])
            )
        elif len(type_list) == 1:
            return self._type_map[type_list[0]]
        elif len(type_list) == 0:
            return 'void'

    def _gen_one_declare(self, op_info, op_name, is_mutable_attr):
        return API_DECLARE_TEMPLATE.format(
            ret_type=self._gen_ret_type(op_info),
            api_name=op_name,
            args=self._gen_api_args(op_info, True, is_mutable_attr),
        )

    def _gen_h_file(self, op_info_items, namespaces, h_file_path):
        declare_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if (
                    op_info.infer_meta_func is None
                    and op_name not in PD_MANUAL_OP_LIST
                ):
                    continue
                declare_str += self._gen_one_declare(op_info, op_name, False)
                if len(op_info.mutable_attribute_name_list) > 0:
                    declare_str += self._gen_one_declare(op_info, op_name, True)

        body = declare_str
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(h_file_path, 'w') as f:
            f.write(H_FILE_TEMPLATE.format(body=body))

    # =====================================
    # Gen impl functions
    # =====================================
    def _gen_in_combine(self, op_info):
        name_list = op_info.input_name_list
        type_list = op_info.input_type_list
        assert len(name_list) == len(type_list)
        combine_op = ''
        combine_op_list = []
        for name, type in zip(name_list, type_list):
            if VECTOR_TYPE in type:
                op_name = f'{name}_combine_op'
                combine_op += COMBINE_OP_TEMPLATE.format(
                    op_name=op_name, in_name=name
                )
                combine_op_list.append(op_name)
            else:
                combine_op_list.append(None)
        return combine_op, combine_op_list

    def _gen_compute_op_args(
        self, op_info, in_combine_op_list, is_mutable_attr
    ):
        input_name_list = op_info.input_name_list
        all_attr_list = op_info.attribute_name_list
        no_mutable_attr_list = op_info.non_mutable_attribute_name_list
        mutable_attr_list = op_info.mutable_attribute_name_list
        assert len(input_name_list) == len(in_combine_op_list)
        ret = []
        for input_name, combine_op in zip(input_name_list, in_combine_op_list):
            if combine_op is None:
                ret.append(input_name)
            else:
                ret.append(f'{combine_op}.out()')
        if is_mutable_attr:
            ret += list(mutable_attr_list + no_mutable_attr_list)
        else:
            ret += list(all_attr_list)
        return ', '.join(ret)

    def _gen_compute_op(
        self, op_info, op_name, in_combine_op_list, is_mutable_attr
    ):
        op_class_name = to_pascal_case(op_name) + 'Op'
        op_inst_name = op_name + '_op'
        return (
            COMPUTE_OP_TEMPLATE.format(
                op_class_name=op_class_name,
                op_inst_name=op_inst_name,
                args=self._gen_compute_op_args(
                    op_info, in_combine_op_list, is_mutable_attr
                ),
            ),
            op_inst_name,
        )

    def _gen_out_split_and_ret_list(self, op_info, op_inst_name):
        name_list = op_info.output_name_list
        type_list = op_info.output_type_list

        split_op_str = ''
        ret_list = []
        for i, (name, type) in enumerate(zip(name_list, type_list)):
            if VECTOR_TYPE in type:
                split_op_name = f'{name}_split_op'
                split_op_str += SPLIT_OP_TEMPLATE.format(
                    op_name=split_op_name, in_name=f'{op_inst_name}.result({i})'
                )
                ret_list.append(f'{split_op_name}.outputs()')
            else:
                ret_list.append(f'{op_inst_name}.result({i})')
        return split_op_str, ret_list

    def _gen_return_result(self, ret_list):
        if len(ret_list) > 1:
            return 'return std::make_tuple({});'.format(', '.join(ret_list))
        elif len(ret_list) == 1:
            return f'return {ret_list[0]};'
        elif len(ret_list) == 0:
            return 'return;'

    def _gen_one_impl(self, op_info, op_name, is_mutable_attr):
        ret_type = self._gen_ret_type(op_info)
        in_combine, in_combine_op_list = self._gen_in_combine(op_info)
        compute_op, op_inst_name = self._gen_compute_op(
            op_info, op_name, in_combine_op_list, is_mutable_attr
        )
        if ret_type == 'void':
            compute_op += f' (void){op_inst_name};'

        out_split, ret_list = self._gen_out_split_and_ret_list(
            op_info, op_inst_name
        )

        ret = API_IMPL_TEMPLATE.format(
            ret_type=ret_type,
            api_name=op_name,
            args=self._gen_api_args(op_info, False, is_mutable_attr),
            in_combine=in_combine,
            compute_op=compute_op,
            out_split=out_split,
            return_result=self._gen_return_result(ret_list),
        )

        ret = re.sub(r' +\n', '', ret)
        return ret

    def _gen_cpp_file(self, op_info_items, namespaces, cpp_file_path):
        impl_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if (
                    op_info.infer_meta_func is None
                    and op_name not in PD_MANUAL_OP_LIST
                ):
                    continue
                impl_str += self._gen_one_impl(op_info, op_name, False)
                if len(op_info.mutable_attribute_name_list) > 0:
                    impl_str += self._gen_one_impl(op_info, op_name, True)
        body = impl_str
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(cpp_file_path, 'w') as f:
            f.write(CPP_FILE_TEMPLATE.format(body=body))

    def gen_h_and_cpp_file(
        self,
        op_yaml_files,
        op_compat_yaml_file,
        namespaces,
        h_file_path,
        cpp_file_path,
    ):
        if os.path.exists(h_file_path):
            os.remove(h_file_path)
        if os.path.exists(cpp_file_path):
            os.remove(cpp_file_path)

        op_info_items = self._parse_yaml(op_yaml_files, op_compat_yaml_file)

        self._gen_h_file(op_info_items, namespaces, h_file_path)
        self._gen_cpp_file(op_info_items, namespaces, cpp_file_path)


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Dialect API Files By Yaml'
    )
    parser.add_argument('--op_yaml_files', type=str)
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--namespaces', type=str)
    parser.add_argument('--api_def_h_file', type=str)
    parser.add_argument('--api_def_cc_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArguments()

    op_yaml_files = args.op_yaml_files.split(",")
    op_compat_yaml_file = args.op_compat_yaml_file
    if args.namespaces is not None:
        namespaces = args.namespaces.split(",")
    api_def_h_file = args.api_def_h_file
    api_def_cc_file = args.api_def_cc_file

    code_gen = CodeGen()
    code_gen.gen_h_and_cpp_file(
        op_yaml_files,
        op_compat_yaml_file,
        namespaces,
        api_def_h_file,
        api_def_cc_file,
    )
