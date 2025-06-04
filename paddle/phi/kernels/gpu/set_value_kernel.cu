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
#include <chrono>
#include <iostream>
#include <type_traits>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"
#include "paddle/phi/kernels/impl/set_value_kernel_impl.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"
namespace phi {
template <typename T, typename Context>
void SetTensorValueKernelV2(const Context& dev_ctx,
                            const DenseTensor& in,
                            const DenseTensor& value,
                            const IntArray& starts,
                            const IntArray& ends,
                            const IntArray& steps,
                            const std::vector<int64_t>& axes,
                            const std::vector<int64_t>& decrease_axes,
                            const std::vector<int64_t>& none_axes,
                            DenseTensor* out) {
  auto in_dims = in.dims();
  auto meta = in.meta();
  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();
  phi::funcs::CheckAndUpdateSliceAttrs(
      in_dims, axes, &starts_local, &ends_local, &steps_local);

  std::vector<int64_t> output_dims = common::vectorize<int64_t>(in.dims());
  std::vector<int64_t> output_stride = common::vectorize<int64_t>(in.strides());
  int64_t output_offset = static_cast<int64_t>(in.offset());
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis_size = in.dims()[axes[i]];
    if (axis_size < 0) {
      continue;
    }

    int64_t step_size = std::abs(steps_local[i]);

    auto out_dim =
        (std::abs(ends_local[i] - starts_local[i]) + step_size - 1) / step_size;
    output_offset += static_cast<int64_t>(
        starts_local[i] * output_stride[axes[i]] * SizeOf(out->dtype()));
    output_dims[axes[i]] = out_dim;
    output_stride[axes[i]] *= steps_local[i];
  }

  // generate new shape
  std::vector<int64_t> new_out_shape;
  std::vector<int64_t> new_out_stride;
  funcs::GetDecreasedDimsAndStrides(output_dims,
                                    output_stride,
                                    decrease_axes,
                                    none_axes,
                                    &new_out_shape,
                                    &new_out_stride);

  if (product(phi::make_ddim(new_out_shape)) <= 0) {
    out->ResetHolder(in.Holder());
    out->ShareInplaceVersionCounterWith(in);
    return;
  }

  CheckIsDimsMatch(phi::make_ddim(new_out_shape), value.dims());
  auto value_dims = phi::vectorize<int64_t>(value.dims());
  DenseTensor value_tensor = Empty<T>(dev_ctx, IntArray{value_dims});
  value_tensor = value;
  auto it = value_dims.begin();
  while (it != value_dims.end() && *it == 1) {
    it = value_dims.erase(it);
  }
  if (value_dims.empty()) value_dims.push_back(1);
  value_tensor.Resize(phi::make_ddim(value_dims));

  if (new_out_shape.empty()) new_out_shape.push_back(1);

  DenseTensor expand_tensor = Empty<T>(dev_ctx, IntArray{new_out_shape});
  ExpandKernel<T, Context>(
      dev_ctx, value_tensor, IntArray{new_out_shape}, &expand_tensor);

  out->ResetHolder(in.Holder());
  out->ShareInplaceVersionCounterWith(in);
  StridedCopyKernel<T, Context>(dev_ctx,
                                expand_tensor,
                                new_out_shape,
                                new_out_stride,
                                output_offset,
                                out);
  out->set_meta(meta);
}

template <typename T, typename Context>
void SetValueKernelV2(const Context& dev_ctx,
                      const DenseTensor& in,
                      const IntArray& starts,
                      const IntArray& ends,
                      const IntArray& steps,
                      const std::vector<int64_t>& axes,
                      const std::vector<int64_t>& decrease_axes,
                      const std::vector<int64_t>& none_axes,
                      const std::vector<int64_t>& shape,
                      const std::vector<Scalar>& values,
                      DenseTensor* out) {
  // auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<T> assign_values;
  assign_values.reserve(values.size());
  for (const auto& val : values) {
    assign_values.push_back(val.to<T>());
  }

  bool is_full_set_one_value = false;
  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();
  if (starts_local.empty() && ends_local.empty() && steps_local.empty() &&
      shape.size() == 1 && shape[0] == 1 && assign_values.size() == 1) {
    is_full_set_one_value = true;
  }

  if (is_full_set_one_value && std::is_same<T, float>::value) {
    dev_ctx.template Alloc<T>(out);
    phi::funcs::set_constant(
        dev_ctx, out, static_cast<float>(assign_values[0]));
    return;
  }

  DenseTensor value_tensor = Empty<T>(dev_ctx, shape);
  phi::TensorFromVector(assign_values, dev_ctx, &value_tensor);
  value_tensor.Resize(common::make_ddim(shape));

  SetTensorValueKernelV2<T, Context>(dev_ctx,
                                     in,
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
                   GPU,
                   ALL_LAYOUT,
                   phi::SetValueKernelV2,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(set_value_with_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::SetTensorValueKernelV2,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
