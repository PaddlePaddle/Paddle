/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LTCENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS TS" BASTS,
WTTHOUT WARRANTTES OR CONDTTTONS OF ANY KTND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/sparse_sqrt_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/sparse/common_shape.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SqrtCsrKernel(const Context& dev_ctx,
                   const SparseCsrTensor& x,
                   SparseCsrTensor* Out) {
  const DDim& x_dims = x.dims();
  const auto& x_crows = x.non_zero_crows();
  const auto& x_cols = x.non_zero_cols();
  const auto& x_values = x.non_zero_elements();
  const auto* x_crows_data = x_crows.data<int64_t>();
  const auto* x_cols_data = x_cols.data<int64_t>();
  const auto place = dev_ctx.GetPlace();

  DenseTensorMeta crows_meta(
      DataType::INT64, x.non_zero_crows().dims(), DataLayout::NCHW);
  DenseTensorMeta cols_meta(
      DataType::INT64, x.non_zero_cols().dims(), DataLayout::NCHW);
  DenseTensorMeta values_meta(
      paddle::experimental::CppTypeToDataType<T>::Type(),
      x.non_zero_elements().dims(),
      DataLayout::NCHW);

  phi::DenseTensor out_crows = phi::Empty(dev_ctx, std::move(crows_meta));
  phi::DenseTensor out_cols = phi::Empty(dev_ctx, std::move(cols_meta));
  phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));

  auto* out_crows_data = out_crows.mutable_data<int64_t>(place);
  auto* out_cols_data = out_cols.mutable_data<int64_t>(place);

  std::memcpy(
      out_crows_data, x_crows_data, sizeof(int64_t) * x_crows.dims()[0]);
  std::memcpy(out_cols_data, x_cols_data, sizeof(int64_t) * x_cols.dims()[0]);

  SqrtKernel<T, Context>(dev_ctx, x_values, &out_values);

  Out->SetMember(out_crows, out_cols, out_values, x_dims);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_sqrt_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SqrtCsrKernel,
                   float,
                   double) {}
