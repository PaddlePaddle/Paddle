// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_get_grad_kernel.h"

#include <array>
#include <cstring>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"

namespace phi {

template <typename T>
void index_get_grad_kernel_impl(const int64_t N,
                                const T* out_grad_data,
                                const int64_t** indices,
                                const DDim& stride,
                                const DDim& shape,
                                T* x_grad_data) {
  for (int64_t idx = 0; idx < N; ++idx) {
    int64_t offset = 0;
    for (int i = 0; i < shape.size(); ++i) {
      int64_t cur_idx = static_cast<int64_t>(*(indices[i] + idx));
      if (cur_idx < 0) cur_idx += shape[i];
      offset += stride[i] * cur_idx;
    }
    x_grad_data[offset] += out_grad_data[idx];
  }
}

template <typename T, typename Context>
void IndexGetGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<const DenseTensor*>& indices,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad) {
  if (out_grad.numel() == 0) {
    // Zero-fill x_grad
    auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
    memset(x_grad_data, 0, x_grad->numel() * sizeof(T));
    return;
  }

  PADDLE_ENFORCE_EQ(
      indices.empty(),
      false,
      common::errors::InvalidArgument("Indices cannot be empty."));
  PADDLE_ENFORCE_LE(x.dims().size(),
                    6,
                    common::errors::InvalidArgument(
                        "Dims of input tensor should be less than 7."));

  std::vector<DenseTensor> temp_args;
  std::vector<const DenseTensor*> int_indices =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &temp_args);
  if (int_indices.empty()) {
    // All-false bool indices → no gradients to scatter
    auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
    memset(x_grad_data, 0, x_grad->numel() * sizeof(T));
    return;
  }

  auto bd_dim = funcs::BroadCastTensorsDims(int_indices);
  std::vector<int64_t> res_dim_v(vectorize(bd_dim));
  std::vector<const DenseTensor*> res_indices(x.dims().size(), nullptr);
  std::vector<DenseTensor> tmp_res_indices;
  std::vector<DenseTensor> range_tensors;

  for (int i = static_cast<int>(int_indices.size()); i < x.dims().size(); ++i) {
    range_tensors.emplace_back(funcs::GetRangeTensor<int64_t, Context>(
        dev_ctx, x.dims()[i], DataType::INT64));
  }

  funcs::DealWithIndices<T, Context>(dev_ctx,
                                     x,
                                     int_indices,
                                     &res_indices,
                                     &tmp_res_indices,
                                     range_tensors,
                                     bd_dim,
                                     &res_dim_v);

  auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  memset(x_grad_data, 0, x_grad->numel() * sizeof(T));

  auto* out_grad_data = out_grad.data<T>();

  const int64_t numel = res_indices[0]->numel();
  std::array<const int64_t*, 7> pd_indices{};
  for (size_t i = 0; i < res_indices.size(); ++i) {
    if (res_indices[i]) pd_indices[i] = res_indices[i]->data<int64_t>();
  }
  auto x_stride = common::stride(x.dims());
  index_get_grad_kernel_impl<T>(
      numel, out_grad_data, pd_indices.data(), x_stride, x.dims(), x_grad_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_get_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexGetGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
