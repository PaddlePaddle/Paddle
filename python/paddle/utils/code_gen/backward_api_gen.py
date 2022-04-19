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

import os
import yaml
import argparse
import re

from api_base import BaseAPI, PREFIX_TENSOR_NAME


class BackwardAPI(BaseAPI):
    def __init__(self, backward_item_yaml):
        super(BackwardAPI, self).__init__(backward_item_yaml)
        self.check_args(backward_item_yaml['forward'])
        self.no_need_buffer = self.parse_no_need_buffer(backward_item_yaml)

    def get_api_name(self, api_item_yaml):
        return api_item_yaml['backward_api']

    def parse_forward_config(self, forward_config):
        # api_name (const Tensor& input, ... , int attr, ...) -> Tensor(out)
        result = re.search(
            r"(?P<api>[a-z][a-z0-9_]+)\s*(?P<args>\([^\)]+\))\s*->\s*(?P<outputs>.+)",
            forward_config)
        api = result.group('api')
        _, outputs, _, _ = self.parse_output(self.api, result.group('outputs'))
        outputs = [item.split('@')[0] for item in outputs]
        fw_inputs, fw_attrs, _, = self.parse_input_and_attr(
            api, result.group('args'))

        return api, fw_inputs, fw_attrs, outputs

    def parse_no_need_buffer(self, api_item_yaml):
        no_need_buffer = []
        if 'no_need_buffer' in api_item_yaml:
            no_need_buffer = [
                item.strip()
                for item in api_item_yaml['no_need_buffer'].split(',')
            ]
        return no_need_buffer

    def check_args(self, forward_config):
        # parse the forward and backward config
        _, fw_inputs, fw_attrs, fw_outputs = self.parse_forward_config(
            forward_config)

        # check the inputs of backward
        for input in self.inputs['names']:
            if input not in fw_inputs['names'] and input not in fw_outputs:
                if input.endswith('_grad'):
                    original_name = input[:-5]
                    assert original_name in fw_outputs, \
                        f"{self.api} : Input Tensor error: the input tensor({input}) of backward should be an input or output or grad of output in forward api. \
                         Please check the forward of {self.api} in yaml."

        # check the attributes of backward
        for attr in self.attrs['names']:
            assert (attr in fw_attrs['names'] and self.attrs['attr_info'][attr][0] == fw_attrs['attr_info'][attr][0]) or \
                 self.attrs['attr_info'][attr][1] is not None, \
                f"{self.api} : Attribute error: The attribute({attr}) of backward isn't consistent with forward api or doesn't have default value. \
                 Please check the args of {self.api} in yaml."

        # check the output of backward
        assert len(self.outputs['types']) <= len(fw_inputs['names']), \
            f"{self.api} : Output error: The number of outputs should be less then the number of inputs of forward api. \
             Please check the output of {self.api} in yaml."

    def gene_kernel_backend_select(self):
        all_no_need_buffer = True
        for in_name in self.inputs['names']:
            if in_name not in self.no_need_buffer:
                all_no_need_buffer = False

        if all_no_need_buffer:
            return """
  kernel_backend = ParseBackend(egr::Controller::Instance().GetExpectedPlace());
"""
        else:
            return super().gene_kernel_backend_select()

    def get_return_type(self, out_type_list):
        return out_type_list[0] if len(
            out_type_list) == 1 else "std::vector<std::vector<Tensor>>"

    def gene_output(self,
                    output_type_list,
                    set_out_func,
                    code_indent,
                    inplace_flag=False):
        kernel_output = ""
        output_names = []
        output_create = ""

        if len(output_type_list) == 1:
            kernel_output = 'kernel_out'
            output_names.append('kernel_out')
            inplace_assign = " = " + self.inplace_map[self.outputs['names'][
                0]] if inplace_flag and self.inplace_map is not None and self.outputs[
                    'names'][0] in self.inplace_map else ""
            output_create = f"""
{code_indent}  {self.outputs['return_type']} api_output{inplace_assign};"""

            if output_type_list[0] == 'std::vector<Tensor>':
                assert self.outputs['out_size_expr'] is not None, \
                     f"{api_name}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                output_create = output_create + f"""
{code_indent}  auto kernel_out = {set_out_func}({self.outputs['out_size_expr']}, kernel_backend, &api_output);"""

            else:
                output_create = output_create + f"""
{code_indent}  auto kernel_out = {set_out_func}(kernel_backend, &api_output);"""

            if not inplace_flag and self.view_map is not None and self.outputs[
                    'names'][0] in self.view_map:
                output_create = output_create + f"""
{code_indent}  VLOG(10) << "{self.view_map[self.outputs['names'][0]]} use_count: " << {PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]}.use_count();
{code_indent}  //if ({PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]}.use_count() == 1 && {PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]}->initialized()) {{
{code_indent}    kernel_out->ShareBufferWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]});
{code_indent}    kernel_out->ShareInplaceVersionCounterWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]});
{code_indent}    VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
{code_indent}  //}}"""

        elif len(output_type_list) > 1:
            output_create = f"""
{code_indent}  {self.outputs['return_type']} api_output({len(output_type_list)});"""

            for i, out_type_item in enumerate(output_type_list):
                kernel_output = kernel_output + f'kernel_out_{i}, '
                output_names.append(f'kernel_out_{i}')
                if out_type_item == 'Tensor':
                    if inplace_flag and self.inplace_map is not None and self.outputs[
                            'names'][i] in self.inplace_map:
                        output_create = output_create + f"""
{code_indent}  api_output[{i}].emplace_back({self.inplace_map[self.outputs['names'][i]]});"""

                    else:
                        output_create = output_create + f"""
{code_indent}  api_output[{i}].emplace_back();"""

                    output_create = output_create + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}(kernel_backend, &api_output[{i}][0]);"""

                else:
                    get_out_code = f'&api_output[{i}]'
                    if inplace_flag and self.inplace_map is not None and self.outputs[
                            'names'][i] in self.inplace_map:
                        output_create = output_create + f"""
{code_indent}  api_output[{i}] = {self.inplace_map[self.outputs['names'][i]]};"""

                    assert self.outputs['out_size_expr'][i] is not None, \
                        f"{api_name}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                    output_create = output_create + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}({self.outputs['out_size_expr'][i]}, kernel_backend, &api_output[{i}]);"""

                if not inplace_flag and self.view_map is not None and self.outputs[
                        'names'][i] in self.view_map:
                    output_create = output_create + f"""
{code_indent}  VLOG(10) << "{self.view_map[self.outputs['names'][i]]} use_count: " << {PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]}.use_count();
{code_indent}  //if ({PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]}.use_count() == 1 && {PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]}->initialized()) {{
{code_indent}    kernel_out_{i}->ShareBufferWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]});
{code_indent}    kernel_out_{i}->ShareInplaceVersionCounterWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]});
{code_indent}    VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
{code_indent}  //}}"""

            kernel_output = kernel_output[:-2]
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api))

        return kernel_output, output_names, output_create


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

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

DECLARE_bool(conv2d_disable_cudnn);
"""


def backward_api_namespace():
    return ("""
namespace paddle {
namespace experimental {

""", """

}  // namespace experimental
}  // namespace paddle
""")


def generate_backward_api(backward_yaml_path, header_file_path,
                          source_file_path):

    with open(backward_yaml_path, 'r') as f:
        bw_apis = yaml.load(f, Loader=yaml.FullLoader)
    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = backward_api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = "paddle/phi/api/backward/backward_api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for bw_api in bw_apis:
        bw_api = BackwardAPI(bw_api)
        header_file.write(bw_api.gene_api_declaration())
        if bw_api.api == 'reshape_grad':
            source_file.write("""
PADDLE_API Tensor reshape_grad(const Tensor& xshape, const Tensor& out_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(out_grad);

  kernel_layout = ParseLayout(out_grad);

  kernel_data_type = ParseDataType(out_grad);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(xshape, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "reshape_grad API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "reshape_grad", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "reshape_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto input_xshape = TensorToDenseTensor(xshape);
  VLOG(10) << "out_grad use_count1: " << out_grad.impl().use_count();
  auto input_out_grad = PrepareData(out_grad, kernel.InputAt(1), {});
  VLOG(10) << "out_grad use_count2: " << input_out_grad.use_count();

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  VLOG(10) << "out_grad use_count3: " << input_out_grad.use_count();
  //if (input_out_grad.use_count() == 1 && input_out_grad->initialized()) {
    kernel_out->ShareBufferWith(*input_out_grad);
    kernel_out->ShareInplaceVersionCounterWith(*input_out_grad);
    VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  //}
  phi::MetaTensor meta_out(kernel_out);

  phi::KernelWithXShapeInferMeta(MakeMetaTensor(*input_xshape), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("reshape_grad compute", paddle::platform::TracerEventType::Operator, 1);
    (*kernel_fn)(*dev_ctx, *input_out_grad, kernel_out);
  }

  return api_output;
}
""")
        elif bw_api.api == 'cross_entropy_with_softmax_grad':
            source_file.write("""
PADDLE_API Tensor cross_entropy_with_softmax_grad(const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(softmax);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(label, softmax, loss_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "cross_entropy_with_softmax_grad API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "cross_entropy_with_softmax_grad", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "cross_entropy_with_softmax_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto input_label = PrepareData(label, kernel.InputAt(0), {});
  VLOG(10) << "yoki: input_softmax use_count1: " << softmax.impl().use_count();
  auto input_softmax = PrepareData(softmax, kernel.InputAt(1), {});
  VLOG(10) << "yoki: input_softmax use_count2: " << input_softmax.use_count();
  auto input_loss_grad = PrepareData(loss_grad, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  VLOG(10) << "input_softmax use_count3: " << input_softmax.use_count();
  //if (input_softmax.use_count() == 1 && input_softmax->initialized()) {
    kernel_out->ShareBufferWith(*input_softmax);
    kernel_out->ShareInplaceVersionCounterWith(*input_softmax);
    VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  //}
  phi::MetaTensor meta_out(kernel_out);

  phi::CrossEntropyWithSoftmaxGradInferMeta(MakeMetaTensor(*input_label), MakeMetaTensor(*input_softmax), MakeMetaTensor(*input_loss_grad), soft_label, use_softmax, numeric_stable_mode, ignore_index, axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, bool, bool, bool, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("cross_entropy_with_softmax_grad compute", paddle::platform::TracerEventType::Operator, 1);
    (*kernel_fn)(*dev_ctx, *input_label, *input_softmax, *input_loss_grad, soft_label, use_softmax, numeric_stable_mode, ignore_index, axis, kernel_out);
  }

  return api_output;
}
""")
        else:
            source_file.write(bw_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ backward API files')
    parser.add_argument(
        '--backward_yaml_path',
        help='path to backward yaml file',
        default='python/paddle/utils/code_gen/backward.yaml')
    parser.add_argument(
        '--backward_header_path',
        help='output of generated backward header code file',
        default='paddle/phi/api/backward/backward_api.h')

    parser.add_argument(
        '--backward_source_path',
        help='output of generated backward source code file',
        default='paddle/phi/api/lib/backward_api.cc')

    options = parser.parse_args()

    backward_yaml_path = options.backward_yaml_path
    header_file_path = options.backward_header_path
    source_file_path = options.backward_source_path

    generate_backward_api(backward_yaml_path, header_file_path,
                          source_file_path)


if __name__ == '__main__':
    main()
