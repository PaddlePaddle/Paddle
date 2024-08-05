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

#include "paddle/phi/kernels/nll_loss_grad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void NllLossGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& label,
                       const paddle::optional<DenseTensor>& weight,
                       const DenseTensor& total_weight,
                       const DenseTensor& d_out,
                       int64_t ignore_index,
                       const std::string& reduction,
                       DenseTensor* d_x) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto& label_type = label.dtype();
  bool label_type_match =
      label_type == phi::DataType::INT32 || label_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(label_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(Label) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        label_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  auto d_out_data = d_out.data<XPUType>();
  auto d_x_data = dev_ctx.template Alloc<XPUType>(d_x);

  auto d_x_dims = d_x->dims();
  std::vector<int64_t> d_x_shape = common::vectorize<int64_t>(d_x_dims);

  auto weight_data =
      weight.get_ptr() ? weight.get_ptr()->data<float>() : nullptr;

  int64_t reduction_id = 0;
  if (reduction == "none") {
    reduction_id = 0;
  } else if (reduction == "mean") {
    reduction_id = 1;
  } else if (reduction == "sum") {
    reduction_id = 2;
  }

  auto total_weight_data = total_weight.data<XPUType>();

  int r;
  if (label_type == phi::DataType::INT32) {
    const int* label_data = label.data<int>();
    r = xpu::nll_loss_grad(dev_ctx.x_context(),
                           d_out_data,
                           d_x_data,
                           d_x_shape,
                           label_data,
                           weight_data,
                           reduction_id,
                           ignore_index,
                           total_weight_data);
  } else if (label_type == phi::DataType::INT64) {
    const int64_t* label_data = label.data<int64_t>();
    r = xpu::nll_loss_grad(dev_ctx.x_context(),
                           d_out_data,
                           d_x_data,
                           d_x_shape,
                           label_data,
                           weight_data,
                           reduction_id,
                           ignore_index,
                           total_weight_data);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "nll_loss_grad");
}

}  // namespace phi

// TODO(xiongkun): add the non-raw kernel register here.
PD_REGISTER_KERNEL(
    nll_loss_grad, XPU, ALL_LAYOUT, phi::NllLossGradKernel, float) {}
