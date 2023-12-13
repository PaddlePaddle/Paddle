// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/conv_kernel.h"

#include "paddle/phi/core/compat/get_kerneltype_forvar_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"
#include "paddle/phi/kernels/onednn/conv_function.h"

namespace phi {

template <typename T, typename Context>
void ConvKernel(const Context& dev_ctx,
                const DenseTensor& input,
                const DenseTensor& filter,
                const std::vector<int>& strides,
                const std::vector<int>& paddings,
                const std::string& padding_algorithm,
                const std::vector<int>& dilations,
                int groups,
                const std::string& data_format,
                DenseTensor* out) {
  auto desc = input.mem_desc();
  auto input_local = input;
  if (data_format == "NHWC" && desc.get_dims().size() == 4 &&
      desc == dnnl::memory::desc(desc.get_dims(),
                                 desc.get_data_type(),
                                 dnnl::memory::format_tag::nhwc)) {
    std::vector<int64_t> dims = desc.get_dims();
    std::rotate(dims.begin() + 1, dims.end() - 1, dims.end());
    input_local.set_mem_desc(dnnl::memory::desc(
        dims, desc.get_data_type(), dnnl::memory::format_tag::nhwc));
    phi::OneDNNContext::tls().set_cur_paddle_data_layout(DataLayout::kNHWC);
  } else if (data_format == "NDHWC" && desc.get_dims().size() == 5 &&
             desc == dnnl::memory::desc(desc.get_dims(),
                                        desc.get_data_type(),
                                        dnnl::memory::format_tag::nhwc)) {
    std::vector<int64_t> dims = desc.get_dims();
    std::rotate(dims.begin() + 1, dims.end() - 1, dims.end());
    input_local.set_mem_desc(dnnl::memory::desc(
        dims, desc.get_data_type(), dnnl::memory::format_tag::nhwc));
    phi::OneDNNContext::tls().set_cur_paddle_data_layout(DataLayout::kNDHWC);
  }

  bool is_test = dev_ctx.HasDnnAttr("is_test")
                     ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                     : false;
  bool is_BFLOAT16 =
      dev_ctx.HasDnnAttr("mkldnn_data_type")
          ? PADDLE_GET_CONST(std::string,
                             dev_ctx.GetDnnAttr("mkldnn_data_type")) ==
                "bfloat16"
          : false;
  bool force_fp32_output =
      dev_ctx.HasDnnAttr("force_fp32_output")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
          : false;

  ConvOnednn<T>(dev_ctx,
                &input,
                &filter,
                nullptr,
                nullptr,
                strides,
                paddings,
                padding_algorithm,
                dilations,
                groups,
                data_format,
                is_test,
                is_BFLOAT16,
                "",
                false,
                force_fp32_output,
                out);
}

template <typename T, typename Context>
void DepthwiseConvKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& filter,
                         const std::vector<int>& strides,
                         const std::vector<int>& paddings,
                         const std::string& padding_algorithm,
                         int groups,
                         const std::vector<int>& dilations,
                         const std::string& data_format,
                         DenseTensor* out) {
  ConvKernel<T, Context>(dev_ctx,
                         input,
                         filter,
                         strides,
                         paddings,
                         padding_algorithm,
                         dilations,
                         groups,
                         data_format,
                         out);
}

template <typename T, typename Context>
void Conv3DKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string& padding_algorithm,
                  int groups,
                  const std::vector<int>& dilations,
                  const std::string& data_format,
                  DenseTensor* out) {
  ConvKernel<T, Context>(dev_ctx,
                         input,
                         filter,
                         strides,
                         paddings,
                         padding_algorithm,
                         dilations,
                         groups,
                         data_format,
                         out);
}

KernelKey ConvGetKernelTypeForVar(const GetKernelTypeForVarContext* ctx) {
  const DenseTensor& tensor = ctx->GetTensor();
  const KernelKey& expected_kernel_type = ctx->GetKernelKey();
  return phi::KernelKey(
      tensor.place(), tensor.layout(), expected_kernel_type.dtype());
}

}  // namespace phi

PD_REGISTER_KERNEL(conv2d,
                   OneDNN,
                   ONEDNN,
                   phi::ConvKernel,
                   float,
                   phi::dtype::bfloat16,
                   uint8_t,
                   int8_t) {
  kernel->get_kerneltype_forvar_fn_ = phi::ConvGetKernelTypeForVar;
}

PD_REGISTER_KERNEL(depthwise_conv2d,
                   OneDNN,
                   ONEDNN,
                   phi::DepthwiseConvKernel,
                   float,
                   phi::dtype::bfloat16,
                   uint8_t,
                   int8_t) {
  kernel->get_kerneltype_forvar_fn_ = phi::ConvGetKernelTypeForVar;
}

PD_REGISTER_KERNEL(conv3d, OneDNN, ONEDNN, phi::Conv3DKernel, float) {
  kernel->get_kerneltype_forvar_fn_ = phi::ConvGetKernelTypeForVar;
}
