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

#include "paddle/phi/kernels/set_value_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {

template <typename T, typename Context>
void SetValueGradKernel(const Context& dev_ctx,
                        const DenseTensor& out_grad,
                        const IntArray& starts,
                        const IntArray& ends,
                        const IntArray& steps,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& decrease_axes,
                        const std::vector<int64_t>& none_axes,
                        DenseTensor* x_grad,
                        DenseTensor* value_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  x_grad->Resize(out_grad.dims());
  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<T>(value_grad);

  const XPUType* dy_data = reinterpret_cast<const XPUType*>(out_grad.data<T>());
  XPUType* dx_data = reinterpret_cast<XPUType*>(x_grad->data<T>());
  XPUType* dv_data = reinterpret_cast<XPUType*>(value_grad->data<T>());

  std::vector<int64_t> starts_vec = starts.GetData();
  std::vector<int64_t> ends_vec = ends.GetData();
  std::vector<int64_t> steps_vec = steps.GetData();

  auto dy_dims = out_grad.dims();
  std::vector<int> dy_shape;
  for (int i = 0; i < dy_dims.size(); ++i) {
    dy_shape.push_back(dy_dims[i]);
  }

  auto dv_dims = value_grad->dims();
  std::vector<int> dv_shape;
  for (int i = 0; i < dv_dims.size(); ++i) {
    dv_shape.push_back(dv_dims[i]);
  }

  auto dx_dims = x_grad->dims();
  std::vector<int> dx_shape;
  for (int i = 0; i < dx_dims.size(); ++i) {
    dx_shape.push_back(dx_dims[i]);
  }

  std::vector<int> starts_vec_int32;
  for (size_t i = 0; i < starts_vec.size(); ++i) {
    starts_vec_int32.push_back(starts_vec[i]);
  }

  std::vector<int> ends_vec_int32;
  for (size_t i = 0; i < ends_vec.size(); ++i) {
    ends_vec_int32.push_back(ends_vec[i]);
  }

  std::vector<int> steps_vec_int32;
  for (size_t i = 0; i < steps_vec.size(); ++i) {
    steps_vec_int32.push_back(steps_vec[i]);
  }

  std::vector<int> axes_int32;
  for (size_t i = 0; i < axes.size(); ++i) {
    axes_int32.push_back(axes[i]);
  }

  std::vector<int> decrease_axes_int32;
  for (size_t i = 0; i < decrease_axes.size(); ++i) {
    decrease_axes_int32.push_back(decrease_axes[i]);
  }

  std::vector<int> none_axes_int32;
  for (size_t i = 0; i < none_axes.size(); ++i) {
    none_axes_int32.push_back(none_axes[i]);
  }

  int r = xpu::set_value_grad(dev_ctx.x_context(),
                              dy_data,
                              dx_data,
                              dv_data,
                              dy_shape,
                              dv_shape,
                              starts_vec_int32,
                              ends_vec_int32,
                              steps_vec_int32,
                              axes_int32,
                              decrease_axes_int32,
                              none_axes_int32);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "set_value_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(set_value_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SetValueGradKernel,
                   float,
                   phi::dtype::float16) {}
