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
from op_gen import (
    PD_MANUAL_OP_LIST,
    OpCompatParser,
    OpInfoParser,
    to_pascal_case,
)

H_FILE_TEMPLATE = """

#pragma once

#include <vector>

#include "paddle/utils/optional.h"
#include "paddle/pir/core/value.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"

{body}

"""

CPP_FILE_TEMPLATE = """

#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"

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
    {handle_optional_inputs}
    {in_combine}
    {compute_op}
    {handle_optional_outputs}
    {out_split}
    {return_result}
}}

"""

OPTIONAL_VECTOR_VALUE_INPUT_TEMPLATE = """
    paddle::optional<pir::Value> optional_{name};
    if (!{name}) {{
        optional_{name} = paddle::make_optional<pir::Value>(pir::Value());
    }} else {{
        auto optional_{name}_combine_op = APIBuilder::Instance().GetBuilder()->Build<pir::CombineOp>({name}.get());
        optional_{name} = paddle::make_optional<pir::Value>(optional_{name}_combine_op.out());
    }}"""

OPTIONAL_VALUE_INPUT_TEMPLATE = """
    paddle::optional<pir::Value> optional_{name};
    if (!{name}) {{
        optional_{name} = paddle::make_optional<pir::Value>(pir::Value());
    }} else {{
        optional_{name} = {name};
    }}"""

OPTIONAL_OPRESULT_OUTPUT_TEMPLATE = """
    paddle::optional<pir::OpResult> optional_{name};
    if (!IsEmptyOpResult({op_name}_op.result({index}))) {{
        optional_{name} = paddle::make_optional<pir::OpResult>({op_name}_op.result({index}));
    }}"""

OPTIONAL_VECTOR_OPRESULT_OUTPUT_TEMPLATE = """
    paddle::optional<std::vector<pir::OpResult>> optional_{name};
    if (!IsEmptyOpResult({op_name}_op.result({index}))) {{
        auto optional_{name}_slice_op = APIBuilder::Instance().GetBuilder()->Build<pir::SplitOp>({op_name}_op.result({index}));
        optional_{name} = paddle::make_optional<std::vector<pir::OpResult>>(optional_{name}_slice_op.outputs());
    }}"""

COMBINE_OP_TEMPLATE = """
    auto {op_name} = APIBuilder::Instance().GetBuilder()->Build<pir::CombineOp>({in_name});"""

SPLIT_OP_TEMPLATE = """
    auto {op_name} = APIBuilder::Instance().GetBuilder()->Build<pir::SplitOp>({in_name});"""

COMPUTE_OP_TEMPLATE = """
    paddle::dialect::{op_class_name} {op_inst_name} = APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::{op_class_name}>({args});"""

OP_INPUT = 'pir::Value'
VECTOR_TYPE = 'pir::VectorType'
INTARRAY_ATTRIBUTE = "paddle::dialect::IntArrayAttribute"

INPUT_TYPE_MAP = {
    'paddle::dialect::DenseTensorType': 'pir::Value',
    'paddle::dialect::SelectedRowsType': 'pir::Value',
    'pir::VectorType<paddle::dialect::DenseTensorType>': 'std::vector<pir::Value>',
}
OPTIONAL_INPUT_TYPE_MAP = {
    'paddle::dialect::DenseTensorType': 'paddle::optional<pir::Value>',
    'paddle::dialect::SelectedRowsType': 'paddle::optional<pir::Value>',
    'pir::VectorType<paddle::dialect::DenseTensorType>': 'paddle::optional<std::vector<pir::Value>>',
}
OUTPUT_TYPE_MAP = {
    'paddle::dialect::DenseTensorType': 'pir::OpResult',
    'paddle::dialect::SelectedRowsType': 'pir::OpResult',
    'pir::VectorType<paddle::dialect::DenseTensorType>': 'std::vector<pir::OpResult>',
}
OPTIONAL_OUTPUT_TYPE_MAP = {
    'paddle::dialect::DenseTensorType': 'paddle::optional<pir::OpResult>',
    'paddle::dialect::SelectedRowsType': 'paddle::optional<pir::OpResult>',
    'pir::VectorType<paddle::dialect::DenseTensorType>': 'paddle::optional<std::vector<pir::OpResult>>',
}


def get_op_class_name(op_name):
    return to_pascal_case(op_name) + 'Op'


class CodeGen:
    def __init__(self) -> None:
        pass

    def _parse_yaml(self, op_yaml_files, op_compat_yaml_file):
        op_compat_parser = OpCompatParser(op_compat_yaml_file)

        op_yaml_items = []
        for yaml_file in op_yaml_files:
            with open(yaml_file, "r") as f:
                ops = yaml.safe_load(f)
                op_yaml_items = op_yaml_items + ops
        op_info_items = []
        for op in op_yaml_items:
            op_compat_item = op_compat_parser.get_compat(op['name'])
            if (
                op_compat_item is None
                and op['name'].endswith(('_grad', '_grad_'))
                and 'forward' in op
            ):
                op_compat_item = op_compat_parser.get_compat(
                    op['forward']['name']
                )

            if (
                op_compat_item is not None
                and op_compat_item['op'] == "pow"
                and 'scalar' in op_compat_item
            ):
                op_compat_item = op_compat_item.pop('scalar')

            op_info_items.append(OpInfoParser(op, op_compat_item))
        return op_info_items

    def _need_skip(self, op_info, op_name):
        return (
            op_info.infer_meta_func is None and op_name not in PD_MANUAL_OP_LIST
        )

    def _is_optional_input(self, op_info, input_name):
        name_list = op_info.input_name_list
        optional_list = op_info.input_optional_list
        if (
            input_name in name_list
            and optional_list[name_list.index(input_name)] == 'true'
        ):
            return True
        return False

    def _is_optinonal_output(self, op_info, output_name):
        inplace_map = op_info.inplace_map
        input_optional_list = op_info.input_optional_list
        input_name_list = op_info.input_name_list
        if inplace_map is None:
            return False

        if output_name in inplace_map.keys():
            input_index = input_name_list.index(inplace_map[output_name])
            if input_optional_list[input_index] == 'true':
                return True
        return False

    # =====================================
    # Gen declare functions
    # =====================================
    def _gen_api_inputs(self, op_info):
        name_list = op_info.input_name_list
        type_list = op_info.input_type_list
        optional_list = op_info.input_optional_list
        assert len(name_list) == len(type_list) == len(optional_list)
        ret = []
        for name, type, optional in zip(name_list, type_list, optional_list):
            if optional == 'true':
                ret.append(f'const {OPTIONAL_INPUT_TYPE_MAP[type]}& {name}')
            else:
                ret.append(f'const {INPUT_TYPE_MAP[type]}& {name}')
        return ', '.join(ret)

    def _gen_api_attrs(
        self, op_info, with_default, is_mutable_attr, is_vector_mutable_attr
    ):
        name_list = op_info.attribute_name_list
        type_list = op_info.attribute_build_arg_type_list
        default_value_list = op_info.attribute_default_value_list
        mutable_name_list = op_info.mutable_attribute_name_list
        mutable_type_list = op_info.mutable_attribute_type_list
        assert len(name_list) == len(type_list) == len(default_value_list)
        no_mutable_attr = []
        mutable_attr = []
        for name, type, default_value in zip(
            name_list, type_list, default_value_list
        ):
            if is_mutable_attr and name in mutable_name_list:
                if (
                    mutable_type_list[mutable_name_list.index(name)][0]
                    == INTARRAY_ATTRIBUTE
                    and is_vector_mutable_attr
                ):
                    mutable_attr.append(f'std::vector<{OP_INPUT}> {name}')
                else:
                    mutable_attr.append(f'{OP_INPUT} {name}')
                continue
            if with_default and default_value is not None:
                if type in ['float', 'double']:
                    default_value = default_value.strip('"')
                no_mutable_attr.append(f'{type} {name} = {default_value}')
            else:
                no_mutable_attr.append(f'{type} {name}')
        return ', '.join(mutable_attr + no_mutable_attr)

    def _gen_api_args(
        self,
        op_info,
        with_default_attr,
        is_mutable_attr,
        is_vector_mutable_attr,
    ):
        inputs = self._gen_api_inputs(op_info)
        attrs = self._gen_api_attrs(
            op_info, with_default_attr, is_mutable_attr, is_vector_mutable_attr
        )
        return (inputs + ', ' + attrs).strip(', ')

    def _gen_ret_type(self, op_info):
        name_list = op_info.output_name_list
        type_list = op_info.output_type_list
        intermediate_list = op_info.output_intermediate_list
        assert len(name_list) == len(type_list) == len(intermediate_list)

        output_num = len(type_list) - intermediate_list.count('true')
        if output_num > 1:
            ret = []
            for name, type, intermediate in zip(
                name_list, type_list, intermediate_list
            ):
                if intermediate == 'true':
                    continue
                if self._is_optinonal_output(op_info, name):
                    ret.append(OPTIONAL_OUTPUT_TYPE_MAP[type])
                else:
                    ret.append(OUTPUT_TYPE_MAP[type])
            return 'std::tuple<{}>'.format(', '.join(ret))
        elif output_num == 1:
            index = intermediate_list.index('false')
            name = name_list[index]
            if self._is_optinonal_output(op_info, name):
                return OPTIONAL_OUTPUT_TYPE_MAP[type_list[index]]
            else:
                return OUTPUT_TYPE_MAP[type_list[index]]
        elif output_num == 0:
            return 'void'

    def _gen_one_declare(
        self, op_info, op_name, is_mutable_attr, is_vector_mutable_attr
    ):
        return API_DECLARE_TEMPLATE.format(
            ret_type=self._gen_ret_type(op_info),
            api_name=op_name,
            args=self._gen_api_args(
                op_info, True, is_mutable_attr, is_vector_mutable_attr
            ),
        )

    def _gen_h_file(self, op_info_items, namespaces, h_file_path):
        declare_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if self._need_skip(op_info, op_name):
                    continue
                declare_str += self._gen_one_declare(
                    op_info, op_name, False, False
                )
                if len(op_info.mutable_attribute_name_list) > 0:
                    declare_str += self._gen_one_declare(
                        op_info, op_name, True, False
                    )
                    if INTARRAY_ATTRIBUTE in {
                        type[0] for type in op_info.mutable_attribute_type_list
                    }:
                        declare_str += self._gen_one_declare(
                            op_info, op_name, True, True
                        )
        body = declare_str
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(h_file_path, 'w') as f:
            f.write(H_FILE_TEMPLATE.format(body=body))

    # =====================================
    # Gen impl functions
    # =====================================
    def _gen_handle_optional_inputs(self, op_info):
        name_list = op_info.input_name_list
        optional_list = op_info.input_optional_list
        type_list = op_info.input_type_list
        assert len(name_list) == len(optional_list) == len(type_list)
        ret = ''
        for name, optional, type in zip(name_list, optional_list, type_list):
            if optional == 'true':
                if VECTOR_TYPE in type:
                    ret += OPTIONAL_VECTOR_VALUE_INPUT_TEMPLATE.format(
                        name=name
                    )
                else:
                    ret += OPTIONAL_VALUE_INPUT_TEMPLATE.format(name=name)
        return ret

    def _gen_handle_optional_outputs(self, op_info, op_name):
        name_list = op_info.output_name_list
        type_list = op_info.output_type_list
        intermediate_list = op_info.output_intermediate_list
        ret = ''
        for i, (name, type, intermediate) in enumerate(
            zip(name_list, type_list, intermediate_list)
        ):
            if intermediate == 'true':
                continue
            if self._is_optinonal_output(op_info, name):
                if VECTOR_TYPE in type:
                    ret += OPTIONAL_VECTOR_OPRESULT_OUTPUT_TEMPLATE.format(
                        name=name,
                        op_name=op_name,
                        index=i,
                    )
                else:
                    ret += OPTIONAL_OPRESULT_OUTPUT_TEMPLATE.format(
                        name=name,
                        op_name=op_name,
                        index=i,
                    )
        return ret

    def _gen_in_combine(self, op_info, is_mutable_attr, is_vector_mutable_attr):
        name_list = op_info.input_name_list
        type_list = op_info.input_type_list
        optional_list = op_info.input_optional_list
        assert len(name_list) == len(type_list) == len(optional_list)
        combine_op = ''
        combine_op_list = []
        for name, type, optional in zip(name_list, type_list, optional_list):
            if optional == 'false' and VECTOR_TYPE in type:
                op_name = f'{name}_combine_op'
                combine_op += COMBINE_OP_TEMPLATE.format(
                    op_name=op_name, in_name=name
                )
                combine_op_list.append(op_name)
            else:
                combine_op_list.append(None)

        if is_mutable_attr:
            name_list = op_info.mutable_attribute_name_list
            type_list = op_info.mutable_attribute_type_list
            assert len(name_list) == len(type_list)
            for name, type in zip(name_list, type_list):
                if type[0] == INTARRAY_ATTRIBUTE and is_vector_mutable_attr:
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
        assert len(input_name_list) + len(mutable_attr_list) == len(
            in_combine_op_list
        ) or len(input_name_list) == len(in_combine_op_list)
        ret = []
        if is_mutable_attr:
            name_list = input_name_list + mutable_attr_list
        else:
            name_list = input_name_list

        for input_name, combine_op in zip(name_list, in_combine_op_list):
            if combine_op is None:
                if self._is_optional_input(op_info, input_name):
                    ret.append(f'optional_{input_name}.get()')
                else:
                    ret.append(input_name)
            else:
                ret.append(f'{combine_op}.out()')
        if is_mutable_attr:
            ret += list(no_mutable_attr_list)
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
        intermediate_list = op_info.output_intermediate_list
        optional_list = op_info.output_optional_list
        assert (
            len(name_list)
            == len(type_list)
            == len(intermediate_list)
            == len(optional_list)
        )

        split_op_str = ''
        ret_list = []
        for i, (name, type, intermediate) in enumerate(
            zip(name_list, type_list, intermediate_list)
        ):
            if intermediate == 'true':
                continue
            if self._is_optinonal_output(op_info, name):
                ret_list.append(f'optional_{name}')
            elif VECTOR_TYPE in type:
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

    def _gen_one_impl(
        self, op_info, op_name, is_mutable_attr, is_vector_mutable_attr
    ):
        ret_type = self._gen_ret_type(op_info)
        in_combine, in_combine_op_list = self._gen_in_combine(
            op_info, is_mutable_attr, is_vector_mutable_attr
        )
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
            args=self._gen_api_args(
                op_info, False, is_mutable_attr, is_vector_mutable_attr
            ),
            handle_optional_inputs=self._gen_handle_optional_inputs(op_info),
            in_combine=in_combine,
            compute_op=compute_op,
            handle_optional_outputs=self._gen_handle_optional_outputs(
                op_info, op_name
            ),
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
                if self._need_skip(op_info, op_name):
                    continue
                impl_str += self._gen_one_impl(op_info, op_name, False, False)
                if len(op_info.mutable_attribute_name_list) > 0:
                    impl_str += self._gen_one_impl(
                        op_info, op_name, True, False
                    )
                    if INTARRAY_ATTRIBUTE in {
                        type[0] for type in op_info.mutable_attribute_type_list
                    }:
                        impl_str += self._gen_one_impl(
                            op_info, op_name, True, True
                        )
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
