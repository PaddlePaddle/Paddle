# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import re

import yaml
from api_base import PREFIX_TENSOR_NAME, BaseAPI

inplace_out_type_map = {
    "Tensor": "Tensor&",
    "std::vector<Tensor>": "std::vector<Tensor>&",
}

inplace_optional_out_type_map = {
    "Tensor": "paddle::optional<Tensor>&",
    "std::vector<Tensor>": "paddle::optional<std::vector<Tensor>>&",
}


class ForwardAPI(BaseAPI):
    def __init__(self, api_item_yaml):
        super().__init__(api_item_yaml)
        self.is_dygraph_api, self.intermediate_outs = self.parse_intermediate(
            api_item_yaml
        )
        # inplace_map[out_val] = in_val
        #  view_map[out_val] = in_val
        self.inplace_map, self.view_map = self.parse_inplace_and_view(
            api_item_yaml
        )

    def get_api_func_name(self):
        if self.is_dygraph_api:
            return self.api + '_intermediate'
        else:
            return self.api

    def dev2_gene_input(self, kernel_tensor_type=None, code_indent=''):
        kernel_param = self.kernel['param']
        input_name_tensor_map, input_tensor_code = super().gene_input(
            kernel_tensor_type, code_indent
        )
        (
            dev2_input_name_tensor_map,
            dev2_input_tensor_code,
            dev2_input_declaration,
        ) = super().dev2_gene_input(kernel_tensor_type, code_indent)

        def dev2_gene_dense_input(input_name, code_indent=''):
            return f"""
{code_indent}  auto dev2_{PREFIX_TENSOR_NAME}{input_name} = paddle::experimental::CopyDenseTensor({PREFIX_TENSOR_NAME}{input_name}, paddle::experimental::GetDebugDev2Type());"""

        def dev2_gene_selected_rows_input(input_name, code_indent=''):
            return f"""
{code_indent}  auto dev2_{PREFIX_TENSOR_NAME}{input_name} = paddle::experimental::CopySelectedRows({PREFIX_TENSOR_NAME}{input_name}, paddle::experimental::GetDebugDev2Type());"""

        def dev2_gene_optional_vec_dense_input(input_name, code_indent=''):
            return f"""
{code_indent}  auto dev2_{PREFIX_TENSOR_NAME}{input_name}_vec = paddle::experimental::CopyOptionalVector({PREFIX_TENSOR_NAME}{input_name}, paddle::experimental::GetDebugDev2Type());
{code_indent}  paddle::optional<std::vector<const phi::DenseTensor*>> dev2_{PREFIX_TENSOR_NAME}{input_name} = paddle::experimental::DenseTensorToConstDenseTensorPtr(dev2_{PREFIX_TENSOR_NAME}{input_name}_vec);"""

        def dev2_gene_vec_dense_input(input_name, ode_indent=''):
            return f"""
{code_indent}  auto dev2_{PREFIX_TENSOR_NAME}{input_name}_vec = paddle::experimental::CopyVector({PREFIX_TENSOR_NAME}{input_name}, paddle::experimental::GetDebugDev2Type());
{code_indent}  std::vector<const phi::DenseTensor*> dev2_{PREFIX_TENSOR_NAME}{input_name} = paddle::experimental::DenseTensorToConstDenseTensorPtr(*dev2_{PREFIX_TENSOR_NAME}{input_name}_vec, {PREFIX_TENSOR_NAME}{input_name});"""

        gene_dev2_input_func = {
            "const Tensor&": {
                "dense": dev2_gene_dense_input,
                "selected_rows": dev2_gene_selected_rows_input,
            },
            "const paddle::optional<Tensor>&": {
                "dense": dev2_gene_dense_input,
                "selected_rows": dev2_gene_selected_rows_input,
            },
            "const std::vector<Tensor>&": {"dense": dev2_gene_vec_dense_input},
            "const paddle::optional<std::vector<Tensor>>&": {
                "dense": dev2_gene_optional_vec_dense_input
            },
        }

        # generate the input that is in view list
        for i, input_name in enumerate(self.inputs['names']):
            if (
                input_name in self.view_map.values()
                and input_name not in input_name_tensor_map.keys()
            ):
                # print("kernel_param: ", kernel_param)
                api_tensor_type = self.inputs['input_info'][input_name]
                phi_tensor_type = (
                    'dense'
                    if kernel_tensor_type is None
                    else kernel_tensor_type[0][kernel_param.index(input_name)]
                )
                if phi_tensor_type == 'dense':
                    if api_tensor_type == "const std::vector<Tensor>&":
                        dev2_input_declaration += f"""
{code_indent}std::unique_ptr<std::vector<phi::DenseTensor>> dev2_{PREFIX_TENSOR_NAME}{input_name}_vec;"""
                    elif (
                        api_tensor_type
                        == "const paddle::optional<std::vector<Tensor>>&"
                    ):
                        dev2_input_declaration += f"""
{code_indent}paddle::optional<std::vector<phi::DenseTensor>> dev2_{PREFIX_TENSOR_NAME}{input_name}_vec;"""
                    dev2_input_declaration += f"""
{code_indent}  {self.dev2_input_trans_map[api_tensor_type][phi_tensor_type]} dev2_{PREFIX_TENSOR_NAME}{input_name};"""
                    dev2_input_tensor_code += gene_dev2_input_func[
                        api_tensor_type
                    ][phi_tensor_type](input_name, code_indent)
                else:
                    # do nothing
                    pass

        return (
            dev2_input_name_tensor_map,
            dev2_input_tensor_code,
            dev2_input_declaration,
        )

    def gene_input(self, kernel_tensor_type=None, code_indent=''):
        kernel_param = self.kernel['param']
        input_name_tensor_map, input_tensor_code = super().gene_input(
            kernel_tensor_type, code_indent
        )

        # generate the input that is in view list
        for i, input_name in enumerate(self.inputs['names']):
            if (
                input_name in self.view_map.values()
                and input_name not in input_name_tensor_map.keys()
            ):
                if (
                    kernel_tensor_type is None
                    or kernel_tensor_type[0][kernel_param.index(input_name)]
                    == 'dense'
                ):
                    trans_flag = self.gene_trans_flag(input_name)
                    input_tensor_code = (
                        input_tensor_code
                        + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name} = PrepareData({input_name}, kernel.InputAt(0), {trans_flag});"""
                    )
                else:
                    # do nothing
                    pass

        return input_name_tensor_map, input_tensor_code

    def parse_intermediate(self, api_item_yaml):
        if 'intermediate' in api_item_yaml:
            intermediate_outs = [
                item.strip()
                for item in api_item_yaml['intermediate'].split(',')
            ]
            return True, intermediate_outs
        else:
            return False, []

    def parse_inplace_and_view(self, api_item_yaml):
        inplace_map, view_map = {}, {}
        for mode in ['inplace', 'view']:
            if mode in api_item_yaml:
                if mode == 'inplace':
                    inplace_map = {}
                else:
                    view_map = {}
                in_out_mapping_list = api_item_yaml[mode].split(',')
                for item in in_out_mapping_list:
                    result = re.search(r"(?P<in>\w+)\s*->\s*(?P<out>\w+)", item)
                    in_val = result.group('in')
                    out_val = result.group('out')
                    assert (
                        in_val in self.inputs['names']
                    ), f"{self.api} : {mode} input error: the input var name('{in_val}') is not found in the input args of {self.api}."
                    assert (
                        out_val in self.outputs['names']
                    ), f"{self.api} : {mode} output error: the output var name('{out_val}') is not found in the output args of {self.api}."

                    if mode == 'inplace':
                        inplace_map[out_val] = in_val
                    else:
                        view_map[out_val] = in_val

        return inplace_map, view_map

    def get_return_type_with_intermediate(self, inplace_flag=False):
        out_type_list = []
        for i, out_type in enumerate(self.outputs['types']):
            out_name = self.outputs['names'][i].split('@')[0]
            if inplace_flag and out_name in self.inplace_map:
                if self.inplace_map[out_name] in self.optional_vars:
                    out_type_list.append(
                        inplace_optional_out_type_map[out_type]
                    )
                else:
                    out_type_list.append(inplace_out_type_map[out_type])
            else:
                out_type_list.append(out_type)

        if len(out_type_list) == 1:
            return out_type_list[0]
        else:
            return "std::tuple<" + ", ".join(out_type_list) + ">"

    def get_return_type(self, inplace_flag=False):
        out_type_list = []
        for i, out_type in enumerate(self.outputs['types']):
            out_name = self.outputs['names'][i].split('@')[0]
            if inplace_flag and out_name in self.inplace_map:
                if self.inplace_map[out_name] in self.optional_vars:
                    out_type_list.append(
                        inplace_optional_out_type_map[out_type]
                    )
                else:
                    out_type_list.append(inplace_out_type_map[out_type])
            elif self.is_dygraph_api or out_name not in self.intermediate_outs:
                out_type_list.append(out_type)

        if len(out_type_list) == 1:
            return out_type_list[0]
        else:
            return "std::tuple<" + ", ".join(out_type_list) + ">"

    def gene_return_code(self):
        if self.is_dygraph_api or len(self.intermediate_outs) == 0:
            return "return api_output;"
        else:
            return_out_list = []
            for i, name in enumerate(self.outputs['names']):
                if name.split('@')[0] not in self.intermediate_outs:
                    return_out_list.append(i)
            if len(return_out_list) == 1:
                return f"return std::get<{return_out_list[0]}>(api_output);"
            else:
                selected_code = [
                    f"std::get<{i}>(api_output)" for i in return_out_list
                ]
            return 'return std::make_tuple(' + ", ".join(selected_code) + ');'

    # '''
    def dev2_gene_output(
        self,
        out_dtype_list,
        out_tensor_type_list=None,
        code_indent='',
        inplace_flag=False,
    ):

        # self.dev2_output_trans_map = {
        #     "Tensor": {
        #         "dense": "phi::DenseTensor*",
        #         "selected_rows": "phi::SelectedRows*",
        #     },
        #     "std::vector<Tensor>": {"dense": "std::vector<phi::DenseTensor*>"},
        # }
        '''
        dev2_out_type_map = {
            "Tensor": {
                "dense": "phi::DenseTensor*&",
                "selected_rows": "phi::SelectedRows*&",
            },
            "std::vector<Tensor>": {"dense": "std::vector<phi::DenseTensor*>&"},
        }
        dev2_inplace_out_type_map = {
            "Tensor":{
                "dense": "std::shared_ptr<phi::DenseTensor>&",
                "selected_rows": "std::shared_ptr<phi::SelectedRows>&",
            },
            "std::vector<Tensor>": {"dense": "std::unique_ptr<std::vector<phi::DenseTensor>>&"},
        }

        # optional 类型的输出一定都是inplace的
        dev2_inplace_optional_out_type_map = {
            "Tensor": {
                "dense": "paddle::optional<phi::DenseTensor>&",
                "selected_rows": "paddle::optional<phi::SelectedRows>&",
            },
            "std::vector<Tensor>": {
                "dense": "paddle::optional<std::vector<phi::DenseTensor>>&"
            },
        }
        def dev2_get_return_type_with_intermediate(out_tensor_type_list, inplace_flag=False):
            out_type_list = []
            input_names = self.inputs['names']
            attr_names = self.attrs['names']
            kernel_param = self.kernel['param']
            if kernel_param is None:
                kernel_param = input_names + attr_names
            for i, out_type in enumerate(self.outputs['types']):
                out_name = self.outputs['names'][i].split('@')[0]
                dense_or_select = 'dense' if out_tensor_type_list is None or out_tensor_type_list[i] == 'dense' else 'selected_rows'
                if inplace_flag and out_name in self.inplace_map  and self.inplace_map[out_name] in kernel_param:
                    if self.inplace_map[out_name] in self.optional_vars:
                        out_type_list.append(
                            dev2_inplace_optional_out_type_map[out_type][dense_or_select]
                        )
                    else:
                        out_type_list.append(dev2_inplace_out_type_map[out_type][dense_or_select])
                else:
                    out_type_list.append(dev2_out_type_map[out_type][dense_or_select])

            if len(out_type_list) == 1:
                return out_type_list[0]
            else:
                return "std::tuple<" + ", ".join(out_type_list) + ">"
        '''

        # copy output
        dev2_kernel_output = []
        dev2_output_names = []
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names
        dev2_output_create = ""
        dev2_output_declaration = ""
        return_type = self.get_return_type_with_intermediate(inplace_flag)
        # dev2_return_type = dev2_get_return_type_with_intermediate(out_tensor_type_list, inplace_flag)

        #  输出只有一个时，输出没有paddle::optional类型, 都一个输出了你再optional就没输出了
        if len(out_dtype_list) == 1:
            dev2_kernel_output.append('dev2_kernel_out')
            dev2_output_names.append('dev2_kernel_out')
            # dev2_get_out_code = ''
            inplace_assign = (
                f"dev2_{PREFIX_TENSOR_NAME}{self.inplace_map[self.outputs['names'][0]]}"
                if inplace_flag
                and self.outputs['names'][0] in self.inplace_map
                and self.inplace_map[self.outputs['names'][0]] in kernel_param
                else "kernel_out"
            )

            set_out_func = (
                'CopyDenseTensor'
                if out_tensor_type_list is None
                or out_tensor_type_list[0] == 'dense'
                else 'CopySelectedRows'
            )

            dev2_kernel_out_type = (
                'phi::DenseTensor*'
                if out_tensor_type_list is None
                or out_tensor_type_list[0] == 'dense'
                else 'phi::SelectedRows*'
            )

            #             dev2_output_create = f"""
            # {code_indent}  auto& dev2_api_output_copy_src = {inplace_assign};"""

            if return_type == 'std::vector<Tensor>':
                dev2_output_declaration += f"""
{code_indent}std::unique_ptr<std::vector<phi::DenseTensor>> dev2_kernel_out_vec;
{code_indent}std::vector<phi::DenseTensor*> dev2_kernel_out;"""

                set_out_func = "CopyVector"
                if inplace_assign != "kernel_out":
                    # inplace_assign += "_vec"
                    dev2_output_create = (
                        dev2_output_create
                        + f"""
{code_indent}  dev2_kernel_out = paddle::experimental::DenseTensorToDenseTensorPtr({inplace_assign}_vec.get(), {inplace_assign});"""
                    )
                else:
                    dev2_output_create = (
                        dev2_output_create
                        + f"""
{code_indent}  dev2_kernel_out_vec = paddle::experimental::{set_out_func}({inplace_assign}, paddle::experimental::GetDebugDev2Type());
{code_indent}  dev2_kernel_out = paddle::experimental::DenseTensorToDenseTensorPtr(dev2_kernel_out_vec.get(), {inplace_assign});"""
                        # {code_indent}  std::vector<phi::DenseTensor> dev2_kernel_out_vec;
                        # {code_indent}  dev2_kernel_out = paddle::experimental::CopyVector({inplace_assign}, paddle::experimental::GetDebugDev2Type(), *dev2_kernel_out_vec);"""
                    )
            else:
                dev2_output_declaration += f"""
{code_indent}std::shared_ptr<{dev2_kernel_out_type[:-1]}> dev2_kernel_out_smart_ptr;
{code_indent}{dev2_kernel_out_type} dev2_kernel_out = nullptr;"""
                if inplace_assign != "kernel_out":
                    # dev2_get_out_code = ".get()"
                    dev2_output_create = (
                        dev2_output_create
                        + f"""
{code_indent}  dev2_kernel_out = {inplace_assign} ? {inplace_assign}.get() : nullptr;"""
                    )
                else:
                    dev2_output_create = (
                        dev2_output_create
                        + f"""
{code_indent}  dev2_kernel_out_smart_ptr = paddle::experimental::{set_out_func}({inplace_assign}, paddle::experimental::GetDebugDev2Type());
{code_indent}  dev2_kernel_out = dev2_kernel_out_smart_ptr ? dev2_kernel_out_smart_ptr.get() : nullptr;"""
                    )
            # 当输出只有一个且返回类型且不是inplace的算子但self.outputs['names'][0]在self.view_map中
            # 这种情况暂时没有
            if (
                not inplace_flag
                and self.view_map is not None
                and self.outputs['names'][0] in self.view_map
            ):
                dev2_output_create = (
                    dev2_output_create
                    + f"""
{code_indent}  dev2_kernel_out->ShareBufferWith(*dev2_{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]});
{code_indent}  dev2_kernel_out->ShareInplaceVersionCounterWith(*dev2_{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]});
{code_indent}  VLOG(3) << "Perform View between Output and Input Tensor of Device 2, share allocation and inplace version.";"""
                )
        elif len(out_dtype_list) > 1:
            #             dev2_output_create = f"""
            # {code_indent}  auto dev2_api_output_copy_src = std::make_tuple("""
            '''
                        dev2_output_create = f"""
            {code_indent}  {dev2_return_type} dev2_api_output_copy_src{{"""
                        for i, out_name in enumerate(self.outputs['names']):
                            # 如果output是inplace的话就用输入的dense
                            # if inplace_flag and out_name in self.inplace_map and self.inplace_map[out_name] in kernel_param:
                            if inplace_flag and out_name in self.inplace_map and self.inplace_map[out_name] in kernel_param:
                                if out_dtype_list[i] == 'std::vector<Tensor>':
                                    dev2_output_create += f"dev2_{PREFIX_TENSOR_NAME}{self.inplace_map[self.outputs['names'][i]]}_vec" + ', '
                                else:
                                    dev2_output_create += f"dev2_{PREFIX_TENSOR_NAME}{self.inplace_map[self.outputs['names'][i]]}" + ', '
                            # 如果output不是inplace的话就新建一个
                            else:
                                dev2_output_create += f'kernel_out_{i}, '
                        # 去掉最后的", "
                        # dev2_output_create = dev2_output_create[:-2] + ');'
                        dev2_output_create = dev2_output_create[:-2] + '};'
            '''
            assert len(self.outputs['names']) == len(
                out_dtype_list
            ), "the len(self.outputs['names']) of kernel: {} is {} , but the len(out_dtype_list) is {}".format(
                self.kernel, len(self.outputs['names']), len(out_dtype_list)
            )
            for i in range(len(out_dtype_list)):
                dev2_kernel_output.append(f'dev2_kernel_out_{i}')
                dev2_output_names.append(f'dev2_kernel_out_{i}')
                inplace_assign = (
                    f"dev2_{PREFIX_TENSOR_NAME}{self.inplace_map[self.outputs['names'][i]]}"
                    if inplace_flag
                    and self.outputs['names'][i] in self.inplace_map
                    and self.inplace_map[self.outputs['names'][i]]
                    in kernel_param
                    else f"kernel_out_{i}"
                )

                # dev2_get_out_code = f"std::get<{i}>(dev2_api_output_copy_src)"
                # if (
                #     self.outputs['names'][i] in self.inplace_map
                #     and self.inplace_map[self.outputs['names'][i]]
                #     in self.optional_vars
                # ):
                #     dev2_get_out_code = f"std::get<{i}>(dev2_api_output_copy_src).get_ptr()"

                dev2_get_ptr = ""
                if (
                    inplace_flag
                    and self.outputs['names'][i] in self.inplace_map
                    and self.inplace_map[self.outputs['names'][i]]
                    in kernel_param
                    and self.inplace_map[self.outputs['names'][i]]
                    not in self.optional_vars
                ):
                    dev2_get_ptr = ".get()"

                set_out_func = (
                    'CopyDenseTensor'
                    if out_tensor_type_list is None
                    or out_tensor_type_list[i] == 'dense'
                    else 'CopySelectedRows'
                )

                dev2_kernel_out_type = (
                    'phi::DenseTensor*'
                    if out_tensor_type_list is None
                    or out_tensor_type_list[0] == 'dense'
                    else 'phi::SelectedRows*'
                )

                if out_dtype_list[i] == 'std::vector<Tensor>':
                    dev2_output_declaration += f"""
{code_indent}std::unique_ptr<std::vector<phi::DenseTensor>> dev2_kernel_out_{i}_vec;
{code_indent}std::vector<phi::DenseTensor*> dev2_kernel_out_{i};"""

                    set_out_func = "CopyVector"
                    dev2_get_ptr = ".get()"
                    if self.outputs['names'][i] in self.inplace_map:
                        # inplace只有两种情况：
                        # paddle::optional<std::vector<phi::DenseTensor>> 这个直接传
                        # std::unique_ptr<std::vector<phi::DenseTensor>>  这个需要.get()
                        if (
                            self.inplace_map[self.outputs['names'][i]]
                            in self.optional_vars
                        ):
                            # set_out_func="CopyOptionalVector"
                            # dev2_get_out_code = f"std::get<{i}>(dev2_api_output_copy_src)"
                            dev2_get_ptr = ""
                            dev2_output_create = (
                                dev2_output_create
                                + f"""
{code_indent}  dev2_kernel_out_{i} = paddle::experimental::DenseTensorToDenseTensorPtr({inplace_assign}_vec{dev2_get_ptr});"""
                            )
                        else:
                            dev2_output_create = (
                                dev2_output_create
                                + f"""
{code_indent}  dev2_kernel_out_{i} = paddle::experimental::DenseTensorToDenseTensorPtr({inplace_assign}_vec{dev2_get_ptr}, {inplace_assign});"""
                            )
                    else:
                        dev2_output_create = (
                            dev2_output_create
                            + f"""
{code_indent}  dev2_kernel_out_{i}_vec = paddle::experimental::{set_out_func}({inplace_assign}, paddle::experimental::GetDebugDev2Type());
{code_indent}  dev2_kernel_out_{i} = paddle::experimental::DenseTensorToDenseTensorPtr(dev2_kernel_out_{i}_vec{dev2_get_ptr}, {inplace_assign});"""
                        )
                else:
                    dev2_output_declaration += f"""
{code_indent}std::shared_ptr<{dev2_kernel_out_type[:-1]}> dev2_kernel_out_{i}_smart_ptr;
{code_indent}{dev2_kernel_out_type} dev2_kernel_out_{i} = nullptr;"""

                    if (
                        inplace_flag
                        and self.outputs['names'][i] in self.inplace_map
                        and self.inplace_map[self.outputs['names'][i]]
                        in kernel_param
                    ):
                        dev2_get_ptr = ".get()"
                        if (
                            self.inplace_map[self.outputs['names'][i]]
                            in self.optional_vars
                        ):
                            dev2_get_ptr = ".get_ptr()"
                        dev2_output_create = (
                            dev2_output_create
                            + f"""
{code_indent}  dev2_kernel_out_{i} = {inplace_assign} ? {inplace_assign}{dev2_get_ptr} : nullptr;"""
                        )
                    else:
                        dev2_output_create = (
                            dev2_output_create
                            + f"""
{code_indent}  dev2_kernel_out_{i}_smart_ptr = paddle::experimental::{set_out_func}({inplace_assign}, paddle::experimental::GetDebugDev2Type());
{code_indent}  dev2_kernel_out_{i} = dev2_kernel_out_{i}_smart_ptr ? dev2_kernel_out_{i}_smart_ptr.get() : nullptr;"""
                        )

                # 算子没有inplace标记，但是有view_map标记
                if (
                    not inplace_flag
                    and self.view_map is not None
                    and self.outputs['names'][i] in self.view_map
                ):
                    if out_dtype_list[i] == 'Tensor':
                        dev2_output_create = (
                            dev2_output_create
                            + f"""
{code_indent}  dev2_kernel_out_{i}->ShareBufferWith(*dev2_{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]});
{code_indent}  dev2_kernel_out_{i}->ShareInplaceVersionCounterWith(*dev2_{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]});
{code_indent}  VLOG(3) << "Perform View between Output and Input Tensor of Device 2, share allocation and inplace version.";"""
                        )
                    else:
                        raise ValueError(
                            "{} : Output error: only support Tensor type when use view in yaml. But get {}".format(
                                self.api, out_dtype_list[i]
                            )
                        )
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api
                )
            )

        return (
            dev2_kernel_output,
            dev2_output_names,
            dev2_output_create,
            dev2_output_declaration,
        )

    def gene_output(
        self,
        out_dtype_list,
        out_tensor_type_list=None,
        code_indent='',
        inplace_flag=False,
    ):
        kernel_output = []
        output_names = []
        output_create = ""
        return_type = self.get_return_type_with_intermediate(inplace_flag)

        #  输出只有一个
        if len(out_dtype_list) == 1:
            kernel_output.append('kernel_out')
            output_names.append('kernel_out')
            inplace_assign = (
                " = " + self.inplace_map[self.outputs['names'][0]]
                if inplace_flag and self.outputs['names'][0] in self.inplace_map
                else ""
            )
            # PADDLE_API Tensor& ceil_(Tensor& x) :  Tensor& api_output = x;
            output_create = f"""
{code_indent}  {return_type} api_output{inplace_assign};"""
            set_out_func = (
                'SetKernelOutput'
                if out_tensor_type_list is None
                or out_tensor_type_list[0] == 'dense'
                else 'SetSelectedRowsKernelOutput'
            )
            # 输出只有一个
            # 如果return_type == 'std::vector<Tensor>'， 且out_tensor_type_list要么为None 要么输出类型为 == 'dense'
            # 调用的是SetKernelOutput(input.size(), &api_output);
            # 当输出只有一个且返回类型为std::vector<Tensor> 目前没有inplace的算子
            if return_type == 'std::vector<Tensor>':
                assert (
                    self.outputs['out_size_expr'][0] is not None
                ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                output_create = (
                    output_create
                    + f"""
{code_indent}  auto kernel_out = {set_out_func}({self.outputs['out_size_expr'][0]}, &api_output);"""
                )
                # print("self.kernel: ", self.kernel)
                # print("output_create: ", output_create)
            # 当输出只有一个且返回类型不为std::vector<Tensor> 且是inplace的算子：
            else:
                output_create = (
                    output_create
                    + f"""
{code_indent}  auto kernel_out = {set_out_func}(&api_output);"""
                )

                # if inplace_assign != "":
                #     print("+++")
                #     print("self.kernel: ", self.kernel)
                #     print("output_create: ", output_create, "\n")
                #     print("---")

                # if inplace_assign == "":
                #     print("+++")
                #     print("self.kernel: ", self.kernel)
                #     print("output_create: ", output_create, "\n")
                #     print("---")

            # 当输出只有一个且返回类型且不是inplace的算子但self.outputs['names'][0]在self.view_map中
            # 这种情况暂时没有
            if (
                not inplace_flag
                and self.view_map is not None
                and self.outputs['names'][0] in self.view_map
            ):
                output_create = (
                    output_create
                    + f"""
{code_indent}  kernel_out->ShareBufferWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]});
{code_indent}  kernel_out->ShareInplaceVersionCounterWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]});
{code_indent}  VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";"""
                )
                # print("+++")
                # print("self.kernel: ", self.kernel)
                # print("output_create: ", output_create, "\n")
                # print("---")

        #  输出有两个及以上
        elif len(out_dtype_list) > 1:
            print_input_dic = {}
            # print_flag = ["dense/select/vector", inplace, optional, view]
            output_create = f"""
{code_indent}  {return_type} api_output;"""

            for out_name in self.outputs['names']:
                if inplace_flag and out_name in self.inplace_map:
                    print_flag = [None, True, None, False]
                else:
                    print_flag = [None, False, None, False]
                print_input_dic[out_name] = print_flag

            if inplace_flag:
                output_create = f"""
{code_indent}  {return_type} api_output{{"""

                for out_name in self.outputs['names']:
                    # 如果output是inplace的话就用输入的dense
                    if out_name in self.inplace_map:
                        output_create += self.inplace_map[out_name] + ', '
                    # 如果output不是inplace的话就新建一个
                    else:
                        output_create += 'Tensor(), '
                # 去掉最后的", "
                output_create = output_create[:-2] + '};'

            assert len(self.outputs['names']) == len(
                out_dtype_list
            ), "the len(self.outputs['names']) of kernel: {} is {} , but the len(out_dtype_list) is {}".format(
                self.kernel, len(self.outputs['names']), len(out_dtype_list)
            )
            for i in range(len(out_dtype_list)):
                kernel_output.append(f'kernel_out_{i}')
                output_names.append(f'kernel_out_{i}')
                set_out_func = (
                    'SetKernelOutput'
                    if out_tensor_type_list is None
                    or out_tensor_type_list[i] == 'dense'
                    else 'SetSelectedRowsKernelOutput'
                )
                print_input_dic[self.outputs['names'][i]][0] = (
                    "dense"
                    if out_tensor_type_list is None
                    or out_tensor_type_list[i] == 'dense'
                    else "select"
                )
                get_out_code = f"&std::get<{i}>(api_output)"
                if (
                    self.outputs['names'][i] in self.inplace_map
                    and self.inplace_map[self.outputs['names'][i]]
                    in self.optional_vars
                ):
                    get_out_code = f"std::get<{i}>(api_output).get_ptr()"
                    print_input_dic[self.outputs['names'][i]][2] = True
                else:
                    print_input_dic[self.outputs['names'][i]][2] = False

                if out_dtype_list[i] == 'std::vector<Tensor>':
                    print_input_dic[self.outputs['names'][i]][0] = "vector"
                    assert (
                        self.outputs['out_size_expr'][i] is not None
                    ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                    # Special case for inplace vector and inplace optional<vector>
                    if self.outputs['names'][i] in self.inplace_map:
                        print_input_dic[self.outputs['names'][i]][1] = True
                        set_out_func = "SetInplaceVectorKernelOutput"
                        if (
                            self.inplace_map[self.outputs['names'][i]]
                            in self.optional_vars
                        ):
                            set_out_func = (
                                "SetInplaceOptionalVectorKernelOutput"
                            )
                            get_out_code = f"std::get<{i}>(api_output)"
                            print_input_dic[self.outputs['names'][i]][2] = True
                        else:
                            print_input_dic[self.outputs['names'][i]][2] = False
                    else:
                        print_input_dic[self.outputs['names'][i]][1] = False
                    output_create = (
                        output_create
                        + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}({self.outputs['out_size_expr'][i]}, {get_out_code});"""
                    )

                else:
                    # 输出不为vetor时，没有optional的情况
                    # print_input_dic[self.outputs['names'][i]][2] = False
                    output_create = (
                        output_create
                        + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}({get_out_code});"""
                    )

                # 算子没有inplace标记，但是有view_map标记
                if (
                    not inplace_flag
                    and self.view_map is not None
                    and self.outputs['names'][i] in self.view_map
                ):
                    print_input_dic[self.outputs['names'][i]][3] = True
                    if out_dtype_list[i] == 'Tensor':
                        output_create = (
                            output_create
                            + f"""
    {code_indent}  kernel_out_{i}->ShareBufferWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]});
    {code_indent}  kernel_out_{i}->ShareInplaceVersionCounterWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]});
    {code_indent}  VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";"""
                        )
                    else:
                        raise ValueError(
                            "{} : Output error: only support Tensor type when use view in yaml. But get {}".format(
                                self.api, out_dtype_list[i]
                            )
                        )
            # for key, value in print_input_dic.items():
            #     if value[0] == "select" and not value[2]:
            #         print("+++")
            #         print("self.kernel: ", self.kernel)
            #         print("output_create: ", output_create,)
            #         print(print_input_dic, "\n")
            #         print("---")
            #         break
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api
                )
            )

        return kernel_output, output_names, output_create

    def reset_view_after_fallback(
        self, out_dtype_list, code_indent='', inplace_flag=False
    ):
        remap_code = ''

        if len(out_dtype_list) == 1:
            if (
                not inplace_flag
                and self.view_map is not None
                and self.outputs['names'][0] in self.view_map
            ):
                remap_code += f"""
{code_indent}    phi::DenseTensor * {self.view_map[self.outputs['names'][0]]}_remap = static_cast<phi::DenseTensor*>({self.view_map[self.outputs['names'][0]]}.impl().get());
{code_indent}    {self.view_map[self.outputs['names'][0]]}_remap->ShareBufferWith(*kernel_out);
{code_indent}    kernel_out->ShareInplaceVersionCounterWith(*{self.view_map[self.outputs['names'][0]]}_remap);
"""
        elif len(out_dtype_list) > 1:
            for i in range(len(out_dtype_list)):
                if (
                    not inplace_flag
                    and self.view_map is not None
                    and self.outputs['names'][i] in self.view_map
                ):
                    remap_code += f"""
{code_indent}    phi::DenseTensor * {self.view_map[self.outputs['names'][i]]}_remap = static_cast<phi::DenseTensor*>({self.view_map[self.outputs['names'][i]]}.impl().get());
{code_indent}    {self.view_map[self.outputs['names'][i]]}_remap->ShareBufferWith(*kernel_out_{i});
{code_indent}    kernel_out_{i}->ShareInplaceVersionCounterWith(*{self.view_map[self.outputs['names'][i]]}_remap);
"""
        return remap_code


def header_include():
    return """
#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/api_registry.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/include/tensor_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/ternary.h"

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/api/profiler/supplement_tracing.h"
#include "paddle/phi/api/lib/debug_op.h"

DECLARE_bool(conv2d_disable_cudnn);
DECLARE_int32(low_precision_op_list);
"""


def api_namespace():
    return (
        """
namespace paddle {
namespace experimental {

""",
        """

}  // namespace experimental
}  // namespace paddle
""",
    )


def declare_extension_api():
    return """
namespace paddle {
PD_DECLARE_API(from_blob);
}  // namespace paddle
"""


def generate_api(
    api_yaml_path, is_fused_ops_yaml, header_file_path, source_file_path
):
    apis = []

    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)

    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = (
        "paddle/phi/api/include/fused_api.h"
        if is_fused_ops_yaml is True
        else "paddle/phi/api/include/api.h"
    )
    # not all fused ops supoort dygraph
    if is_fused_ops_yaml is True:
        new_apis = [
            api
            for api in apis
            if "support_dygraph_mode" in api
            and api["support_dygraph_mode"] is True
        ]
        apis = new_apis

    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for api in apis:
        # print(api)
        foward_api = ForwardAPI(api)
        if foward_api.is_dygraph_api:
            # 对于api.cpp中的api，无论是不是intermediate都要生成一个接口，
            # 不同的是yaml中带intermediate的api，需要再在intermediate中多生成一个带_intermediate接口
            foward_api.is_dygraph_api = False

        header_file.write(foward_api.gene_api_declaration())
        source_file.write(foward_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    source_file.write(declare_extension_api())

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files'
    )
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        nargs='+',
        default=['paddle/phi/api/yaml/ops.yaml'],
    )

    parser.add_argument(
        '--is_fused_ops_yaml',
        help='flag of fused ops yaml',
        action='store_true',
    )

    parser.add_argument(
        '--api_header_path',
        help='output of generated api header code file',
        default='paddle/phi/api/include/api.h',
    )

    parser.add_argument(
        '--api_source_path',
        help='output of generated api source code file',
        default='paddle/phi/api/lib/api.cc',
    )

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    is_fused_ops_yaml = options.is_fused_ops_yaml
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path

    generate_api(
        api_yaml_path, is_fused_ops_yaml, header_file_path, source_file_path
    )


if __name__ == '__main__':
    main()
