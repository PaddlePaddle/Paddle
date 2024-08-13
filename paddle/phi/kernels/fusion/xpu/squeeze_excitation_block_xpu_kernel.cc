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

#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename TW, typename Context>
void SqueezeExcitationKernelImpl(const Context& ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& filter,
                                 const DenseTensor& filter_max,
                                 const paddle::optional<DenseTensor>& bias,
                                 const paddle::optional<DenseTensor>& branch,
                                 const std::vector<int>& act_type,
                                 const std::vector<float>& act_param,
                                 const std::vector<int>& filter_dims,
                                 DenseTensor* out) {
  using XPUTypeX = typename XPUTypeTrait<T>::Type;
  using XPUTypeW = typename XPUTypeTrait<TW>::Type;

  auto* weight1_ptr = filter.data<TW>();
  auto weight_len = filter.numel();
  auto weight1_len = weight_len / 2;
  auto* weight2_ptr = weight1_ptr + weight1_len;

  auto input_dims = x.dims();

  int batch = static_cast<int>(input_dims[0]);
  int channel = static_cast<int>(input_dims[1]);
  int h = static_cast<int>(input_dims[2]);
  int w = static_cast<int>(input_dims[3]);
  auto* input_data = reinterpret_cast<const XPUTypeX*>(x.data<T>());
  const XPUTypeX* branch_data = nullptr;
  auto* branch_tensor = branch.get_ptr();
  if (branch_tensor != nullptr) {
    branch_data = reinterpret_cast<const XPUTypeX*>(branch_tensor->data<T>());
  }
  const float* bias1_ptr =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  const float* bias2_ptr = (bias1_ptr != nullptr)
                               ? (bias1_ptr + filter_dims[1] / filter_dims[0])
                               : nullptr;
  int max_ptr_size = 6;
  const float* w1_maxptr = filter_max.data<float>();
  const float* w2_maxptr = w1_maxptr + max_ptr_size;
  auto* out_data = reinterpret_cast<XPUTypeX*>(ctx.template Alloc<T>(out));

  std::vector<xpu::Activation_t> act;
  for (size_t i = 0; i < 3; i++) {
    xpu::Activation_t cur_act = (xpu::Activation_t::act_enum)act_type[i];
    if (act_type[i] == 5) {
      cur_act.leaky_alpha = act_param[i];
    } else if (act_type[i] == 15) {
      cur_act.hard_sigmoid_slope = act_param[i];
    }
    act.push_back(cur_act);
  }
  int r = xpu::squeeze_excitation_block<XPUTypeX, XPUTypeW, XPUTypeW>(
      /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
      /* const T* x */ input_data,
      /* const TW* weight1 */ reinterpret_cast<const XPUTypeW*>(weight1_ptr),
      /* const TW* weight2 */ reinterpret_cast<const XPUTypeW*>(weight2_ptr),
      /* T* y */ out_data,
      /* int64_t n x */ batch,
      /* int64_t c x */ channel,
      /* int64_t h */ h,
      /* int64_t w */ w,
      /* int64_t r */ filter_dims[0],
      /* const float* w1_maxptr */ reinterpret_cast<const float*>(w1_maxptr),
      /* const float* w2_maxptr */ reinterpret_cast<const float*>(w2_maxptr),
      /* const float* bias1 x */ bias1_ptr,
      /* const float* bias2 */ bias2_ptr,
      /* const T* branch */ branch_data,
      /* const Activation_t& excitation_act1 */ act[0],
      /* const Activation_t& excitation_act2 */ act[1],
      /* const Activation_t& block_act */ act[2]);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "squeeze_excitation_block");
}

#define SQUEEZE_EXCITATION_KERNEL_IMPL(t_dtype_, tw_dtype_)              \
  SqueezeExcitationKernelImpl<t_dtype_, tw_dtype_, Context>(ctx,         \
                                                            x,           \
                                                            filter,      \
                                                            filter_max,  \
                                                            bias,        \
                                                            branch,      \
                                                            act_type,    \
                                                            act_param,   \
                                                            filter_dims, \
                                                            out);

template <typename T, typename Context>
void SqueezeExcitationKernel(const Context& ctx,
                             const DenseTensor& x,
                             const DenseTensor& filter,
                             const DenseTensor& filter_max,
                             const paddle::optional<DenseTensor>& bias,
                             const paddle::optional<DenseTensor>& branch,
                             const std::vector<int>& act_type,
                             const std::vector<float>& act_param,
                             const std::vector<int>& filter_dims,
                             DenseTensor* out) {
  if (x.dtype() == DataType::FLOAT16 && filter.dtype() == DataType::INT16) {
    // float16 kernel
    SQUEEZE_EXCITATION_KERNEL_IMPL(phi::dtype::float16, int16_t);
  } else if (x.dtype() == DataType::FLOAT32 &&
             filter.dtype() == DataType::INT16) {
    // float32 kernel
    SQUEEZE_EXCITATION_KERNEL_IMPL(float, int16_t);
  } else {
    PADDLE_ENFORCE(true,
                   errors::InvalidArgument("Not support x_dtype is %s ",
                                           DataTypeToString(x.dtype())));
  }
  return;
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(squeeze_excitation_block,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::SqueezeExcitationKernel,
                   phi::dtype::float16,
                   float) {}
