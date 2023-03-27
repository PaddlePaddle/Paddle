// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void ConvXPUKernel(const Context& ctx,
                   const DenseTensor& Input,
                   const paddle::optional<DenseTensor>& InputMax,
                   const DenseTensor& Filter,
                   const DenseTensor& FilterMax,
                   const paddle::optional<DenseTensor>& Bias,
                   const paddle::optional<DenseTensor>& Branch,
                   const std::vector<int>& paddings,
                   const std::vector<int>& dilations,
                   const std::vector<int>& strides,
                   const std::string& padding_algorithm,
                   int groups,
                   bool has_bias,
                   bool has_branch,
                   int act_type,
                   float act_param,
                   DenseTensor* Output,
                   DenseTensor* OutputMax) {
  LOG(INFO) << "-----------start running conv_xpu kernel.";
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto input_dims = Input.dims();
  auto filter_dims = Filter.dims();
  // update paddings and dilations accoring to padding_algorithm
  std::vector<int> paddings_vec = paddings;
  std::vector<int> dilations_vec = dilations;
  DDim filter_data_dims = phi::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(&paddings_vec,
                                &dilations_vec,
                                padding_algorithm,
                                input_dims,
                                strides,
                                ksize);

  int batch = static_cast<int>(input_dims[0]);
  int in_c = static_cast<int>(input_dims[1]);
  int in_h = static_cast<int>(input_dims[2]);
  int in_w = static_cast<int>(input_dims[3]);
  int out_c = static_cast<int>(filter_dims[0]);
  int win_h = static_cast<int>(filter_dims[2]);
  int win_w = static_cast<int>(filter_dims[3]);

  auto* input_data = reinterpret_cast<const XPUType*>(Input.data<T>());
  const float* input_max_data = InputMax.get_ptr() == nullptr
                                    ? nullptr
                                    : InputMax.get_ptr()->data<float>();
  auto* branch_data =
      Branch.get_ptr() == nullptr
          ? nullptr
          : reinterpret_cast<const XPUType*>(Branch.get_ptr()->data<T>());
  const float* bias_data =
      Bias.get_ptr() == nullptr ? nullptr : Bias.get_ptr()->data<float>();
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(Output));

  xpu::Activation_t act(static_cast<xpu::Activation_t::act_enum>(act_type));
  if (act_type == 5) {
    act.leaky_alpha = act_param;
  } else if (act_type == 15) {
    act.hard_sigmoid_slope = act_param;
  }
  LOG(INFO) << "-------------start running xpu::conv2d_fusion";
  int r =
      xpu::conv2d_fusion<XPUType, int16_t, XPUType, int16_t>(  // TX/TW/TY/TGEMM
          ctx.x_context(),                                     // ctx
          input_data,                                          // input
          Filter.data<int16_t>(),                              // filter
          out_data,                                            // output
          batch,                                               // batch
          in_c,                                                // in_c
          in_h,                                                // in_h
          in_w,                                                // in_w
          out_c,                                               // out_c
          std::vector<int>{win_h, win_w},                      // k_size
          strides,                                             // strides
          paddings_vec,                                        // paddings
          dilations_vec,                                       // dilations
          groups,                                              // group
          input_max_data,                                      // in_maxptr
          FilterMax.data<float>(),                             // filter_maxptr
          ctx.template Alloc<float>(OutputMax),                // out_maxptr
          true,                                                // ?
          bias_data,                                           // bias
          branch_data,                                         // branch
          act,                                                 // act
          nullptr);                                            // branch_maxptr
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv_xpu");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(conv_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::ConvXPUKernel,
                   float,
                   phi::dtype::float16) {}
