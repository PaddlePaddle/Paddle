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

#include "glog/logging.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {
namespace fusion {

static phi::DDim BroadCastInferShape(const DDim x_dims,
                                     const DDim y_dims,
                                     int axis) {
  std::vector<int> out_dims_array(x_dims.size(), -1);
  if (x_dims != y_dims) {
    int max_dim = std::max(x_dims.size(), y_dims.size());
    if (x_dims.size() == y_dims.size()) {
      PADDLE_ENFORCE_EQ((axis == -1) || (axis == 0),
                        true,
                        common::errors::InvalidArgument(
                            "axis should be -1 or 0 while the dimension of "
                            "tensor X (%s) is equal to the dimension of "
                            "tensor Y (%s), but received axis: %s",
                            x_dims.size(),
                            y_dims.size(),
                            axis));
    }
    PADDLE_ENFORCE_EQ((axis >= (-1 * max_dim)) && (axis < max_dim),
                      true,
                      common::errors::InvalidArgument(
                          "The axis range must be [%s, %s), but axis is %s. "
                          "Please set the axis again.",
                          -1 * max_dim,
                          max_dim,
                          axis));
    axis = (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1)
                     : axis);
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    out_dims_array.resize(max_dim);
    phi::funcs::GetBroadcastDimsArrays(x_dims,
                                       y_dims,
                                       x_dims_array.data(),
                                       y_dims_array.data(),
                                       out_dims_array.data(),
                                       max_dim,
                                       axis);

    return common::make_ddim(out_dims_array);
  }
  return x_dims;
}

template <typename T, typename Context>
void AddLayernormXPUKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           const DenseTensor& scale,
                           const DenseTensor& bias,
                           int begin_norm_axis,
                           float epsilon,
                           DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  const float* scale_data = scale.data<float>();
  const float* bias_data = bias.data<float>();

  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto out_dims = BroadCastInferShape(x_dims, y_dims, -1);
  auto layer_norm_x_mat_dims = common::flatten_to_2d(out_dims, begin_norm_axis);
  int64_t m = layer_norm_x_mat_dims[0];
  int64_t n = layer_norm_x_mat_dims[1];

  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));

  int r = xpu::add_layer_norm_fusion<XPUType>(  // T
      /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
      /* const T* x */ x_data,
      /* const T* y */ y_data,
      /* T* z */ out_data,
      /* int64_t m */ m,
      /* int64_t n */ n,
      /* float epsilon */ epsilon,
      /* const float* scale */ scale_data,
      /* const float* bias */ bias_data,
      /* float* mean */ nullptr,
      /* float* variance */ nullptr,
      /* T* z_add */ nullptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "add_layer_norm_fusion");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(add_layernorm_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::AddLayernormXPUKernel,
                   float,
                   phi::dtype::float16) {}
