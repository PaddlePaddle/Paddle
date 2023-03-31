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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/onednn/conv_function.h"

namespace phi {

template <typename T, typename Context>
void FusedConv2DKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& filter,
                       const paddle::optional<DenseTensor>& bias,
                       const paddle::optional<DenseTensor>& residual_param,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::string& padding_algorithm,
                       const std::vector<int>& dilations,
                       int groups,
                       const std::string& data_format,
                       const std::string& mkldnn_data_type,
                       const std::string& fuse_activation,
                       bool fuse_residual_conn,
                       bool force_fp32_output,
                       DenseTensor* out) {
  bool is_BFLOAT16 = mkldnn_data_type == "bfloat16";

  ConvOnednn<T>(dev_ctx,
                &input,
                &filter,
                bias.get_ptr(),
                residual_param.get_ptr(),
                strides,
                paddings,
                padding_algorithm,
                dilations,
                groups,
                data_format,
                true,
                is_BFLOAT16,
                fuse_activation,
                fuse_residual_conn,
                force_fp32_output,
                out);
}

template <typename T, typename Context>
void FusedDepthwiseConv2DKernel(
    const Context& dev_ctx,
    const DenseTensor& input,
    const DenseTensor& filter,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& residual_param,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format,
    const std::string& mkldnn_data_type,
    const std::string& fuse_activation,
    bool fuse_residual_conn,
    bool force_fp32_output,
    DenseTensor* out) {
  bool is_BFLOAT16 = mkldnn_data_type == "bfloat16";

  ConvOnednn<T>(dev_ctx,
                &input,
                &filter,
                bias.get_ptr(),
                residual_param.get_ptr(),
                strides,
                paddings,
                padding_algorithm,
                dilations,
                groups,
                data_format,
                true,
                is_BFLOAT16,
                fuse_activation,
                fuse_residual_conn,
                force_fp32_output,
                out);
}

template <typename T, typename Context>
void FusedConv3DKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& filter,
                       const paddle::optional<DenseTensor>& bias,
                       const paddle::optional<DenseTensor>& residual_param,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::string& padding_algorithm,
                       const std::vector<int>& dilations,
                       int groups,
                       const std::string& data_format,
                       const std::string& mkldnn_data_type,
                       const std::string& fuse_activation,
                       bool fuse_residual_conn,
                       bool force_fp32_output,
                       DenseTensor* out) {
  bool is_BFLOAT16 = mkldnn_data_type == "bfloat16";

  ConvOnednn<T>(dev_ctx,
                &input,
                &filter,
                bias.get_ptr(),
                residual_param.get_ptr(),
                strides,
                paddings,
                padding_algorithm,
                dilations,
                groups,
                data_format,
                true,
                is_BFLOAT16,
                fuse_activation,
                fuse_residual_conn,
                force_fp32_output,
                out);
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_conv2d,
                   OneDNN,
                   ONEDNN,
                   phi::FusedConv2DKernel,
                   float,
                   phi::dtype::bfloat16,
                   uint8_t,
                   int8_t) {}

PD_REGISTER_KERNEL(
    fused_conv3d, OneDNN, ONEDNN, phi::FusedConv3DKernel, float) {}
