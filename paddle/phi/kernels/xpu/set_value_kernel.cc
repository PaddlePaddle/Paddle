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

#include "paddle/phi/kernels/set_value_kernel.h"

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
void SetTensorValueKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& value,
                          const IntArray& starts,
                          const IntArray& ends,
                          const IntArray& steps,
                          const std::vector<int64_t>& axes,
                          const std::vector<int64_t>& decrease_axes,
                          const std::vector<int64_t>& none_axes,
                          DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  const XPUType* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* v_data = reinterpret_cast<const XPUType*>(value.data<T>());
  XPUType* y_data = reinterpret_cast<XPUType*>(out->data<T>());

  std::vector<int64_t> starts_vec = starts.GetData();
  std::vector<int64_t> ends_vec = ends.GetData();
  std::vector<int64_t> steps_vec = steps.GetData();

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

  auto x_dims = x.dims();
  std::vector<int> x_shape;
  for (int i = 0; i < x_dims.size(); ++i) {
    x_shape.push_back(x_dims[i]);
  }

  auto v_dims = value.dims();
  std::vector<int> v_shape;
  for (int i = 0; i < v_dims.size(); ++i) {
    v_shape.push_back(v_dims[i]);
  }

  int r = xpu::set_value(dev_ctx.x_context(),
                         x_data,
                         v_data,
                         y_data,
                         x_shape,
                         v_shape,
                         starts_vec_int32,
                         ends_vec_int32,
                         steps_vec_int32,
                         axes_int32,
                         decrease_axes_int32,
                         none_axes_int32);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "set_value");
}

template <typename T, typename Context>
void SetValueKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const IntArray& starts,
                    const IntArray& ends,
                    const IntArray& steps,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& decrease_axes,
                    const std::vector<int64_t>& none_axes,
                    const std::vector<int64_t>& shape,
                    const std::vector<Scalar>& values,
                    DenseTensor* out) {
  std::vector<T> assgin_values;
  assgin_values.reserve(values.size());
  for (const auto& val : values) {
    assgin_values.push_back(val.to<T>());
  }
  DenseTensor value_tensor = Empty<T>(dev_ctx, shape);
  paddle::framework::TensorFromVector(assgin_values, dev_ctx, &value_tensor);
  value_tensor.Resize(phi::make_ddim(shape));

  SetTensorValueKernel<T, Context>(dev_ctx,
                                   x,
                                   value_tensor,
                                   starts,
                                   ends,
                                   steps,
                                   axes,
                                   decrease_axes,
                                   none_axes,
                                   out);
}

}  // namespace phi

PD_REGISTER_KERNEL(set_value,
                   XPU,
                   ALL_LAYOUT,
                   phi::SetValueKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(set_value_with_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::SetTensorValueKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t) {}
