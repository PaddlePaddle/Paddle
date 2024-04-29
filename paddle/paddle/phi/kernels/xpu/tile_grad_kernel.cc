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

#include "paddle/phi/kernels/tile_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TileGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const IntArray& repeat_times,
                    DenseTensor* x_grad) {
  auto x_dims = x.dims();
  auto vec_x_dims = common::vectorize<int>(x_dims);
  auto repeat_times_data = repeat_times.GetData();
  if (repeat_times_data.size() < vec_x_dims.size()) {
    int diff = vec_x_dims.size() - repeat_times_data.size();
    repeat_times_data.insert(repeat_times_data.begin(), diff, 1);
  } else {
    int diff = repeat_times_data.size() - vec_x_dims.size();
    vec_x_dims.insert(vec_x_dims.begin(), diff, 1);
  }
  // 1. reshape_dims_vec is the broadcast parameter.
  // 2. reduce_dims_vec is the dimension parameter to compute gradients. For
  //    each dimension expanded, the gradients should be summed to original
  //    size.
  std::vector<int> reshape_dims_vec;
  std::vector<int> reduce_dims_vec;
  for (size_t i = 0; i < repeat_times_data.size(); ++i) {
    reduce_dims_vec.push_back(reshape_dims_vec.size());
    reshape_dims_vec.push_back(repeat_times_data[i]);
    reshape_dims_vec.push_back(vec_x_dims[i]);
  }

  dev_ctx.template Alloc<T>(x_grad);

  int dims = reduce_dims_vec.size();

  bool just_copy = true;
  for (size_t i = 0; i < repeat_times_data.size(); i++) {
    if (repeat_times_data[i] != 1) {
      just_copy = false;
      break;
    }
  }
  // no need reduce, just copy
  if (just_copy) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    // TensorCopy may change the dims of dx
    x_grad->Resize(x_dims);
  } else {
    PADDLE_ENFORCE_GE(dims,
                      1,
                      errors::InvalidArgument(
                          "The rank of the input 'Out@GRAD' for tile_grad op "
                          "must be greater than or equal to 1, but "
                          "the value received is %d.",
                          dims));
    PADDLE_ENFORCE_LE(dims,
                      MAX_RANK_SUPPORTED,
                      errors::InvalidArgument(
                          "The rank of the input 'Out@GRAD' for tile_grad op "
                          "must be less than or equal "
                          "to %d, but the value received is %d.",
                          MAX_RANK_SUPPORTED,
                          dims));

    using XPUType = typename XPUTypeTrait<T>::Type;
    // int reduce_sum(Context* ctx, const T* x, T* y, const std::vector<int>&
    // xshape, const std::vector<int>& rdims)
    const auto* out_data = reinterpret_cast<const XPUType*>(out_grad.data<T>());
    auto* x_grad_data = reinterpret_cast<XPUType*>(x_grad->data<T>());
    int r = xpu::reduce_sum<XPUType>(dev_ctx.x_context(),
                                     out_data,
                                     x_grad_data,
                                     reshape_dims_vec,
                                     reduce_dims_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(tile_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::TileGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
