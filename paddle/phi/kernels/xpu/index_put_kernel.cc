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

#include "paddle/phi/kernels/index_put_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/xpu/index_put_xpu_utils.h"

namespace phi {
template <typename T, typename Context>
void IndexPutKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices,
                    const DenseTensor& value,
                    bool accumulate,
                    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      value.dtype(),
      common::errors::InvalidArgument(
          "The data type of tensor value must be same to the data type "
          "of tensor x."));
  PADDLE_ENFORCE_EQ(
      indices.empty(),
      false,
      common::errors::InvalidArgument("Indices cannot be empty."));
  const int64_t total_dims = x.dims().size();
  PADDLE_ENFORCE_LE(
      total_dims,
      6,
      errors::InvalidArgument("Dims of input tensor should be less than 7."));

  // All bool indices are converted to integers currently
  std::vector<DenseTensor> tmp_args;
  std::vector<const DenseTensor*> int_indices_v =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &tmp_args);
  if (int_indices_v.empty()) {
    if (!out->initialized()) {
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto bd_dims = funcs::BroadCastTensorsDims(int_indices_v);
  DenseTensor res_indices(DataType::INT64);
  // Broadcast and merge indices
  XPUDealWithIndices<Context>(dev_ctx, int_indices_v, bd_dims, &res_indices);
  auto index_shape = common::vectorize<int64_t>(res_indices.dims());
  auto x_shape = common::vectorize<int64_t>(x.dims());

  const T* value_data = value.data<T>();

  // Broadcast value
  auto value_shape = common::vectorize<int64_t>(value.dims());
  int64_t value_rank = bd_dims.size() + (x_shape.size() - int_indices_v.size());
  std::vector<int64_t> value_shape_bd(value_rank);
  std::copy(index_shape.begin(), index_shape.end() - 1, value_shape_bd.begin());
  std::copy(x_shape.begin() + int_indices_v.size(),
            x_shape.end(),
            value_shape_bd.begin() + index_shape.size() - 1);

  DenseTensor value_bd(value.dtype());

  if (value_shape != value_shape_bd) {
    value_bd.Resize(common::make_ddim(value_shape_bd));
    ExpandKernel<T, Context>(
        dev_ctx, value, IntArray(value_shape_bd), &value_bd);
    value_data = value_bd.data<T>();
  }

  int r = xpu::index_put<XPUType, int64_t>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x.data<T>()),
      reinterpret_cast<const XPUType*>(value_data),
      res_indices.data<int64_t>(),
      reinterpret_cast<XPUType*>(out_data),
      x_shape,
      index_shape,
      accumulate);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_put");
  if (dev_ctx.x_context()->xpu_stream) {
    dev_ctx.Wait();
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_put,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexPutKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t) {}
