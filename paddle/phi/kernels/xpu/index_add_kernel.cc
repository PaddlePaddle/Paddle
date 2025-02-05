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

#include "paddle/phi/kernels/index_add_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void IndexAddKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& index,
                    const DenseTensor& add_value,
                    int axis,
                    DenseTensor* out) {
  auto index_type = index.dtype();
  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(index_type),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  using XPUType = typename XPUTypeTrait<T>::Type;
  auto input_dim = x.dims();
  int dim = axis >= 0 ? axis : axis + input_dim.size();
  auto input_vector = common::vectorize<int64_t>(input_dim);
  int64_t numel = add_value.numel();
  if (numel == 0) return;
  ctx.template Alloc<T>(out);
  int r = 0;
  if (index_type == phi::DataType::INT64) {
    r = xpu::index_add<XPUType, int64_t>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<const XPUType*>(add_value.data<T>()),
        reinterpret_cast<XPUType*>(out->data<T>()),
        reinterpret_cast<const int64_t*>(index.data<int64_t>()),
        input_vector,
        index.numel(),
        dim,
        (XPUType)(1.0f));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_add");
  } else if (index_type == phi::DataType::INT32) {
    r = xpu::index_add<XPUType, int>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<const XPUType*>(add_value.data<T>()),
        reinterpret_cast<XPUType*>(out->data<T>()),
        reinterpret_cast<const int*>(index.data<int>()),
        input_vector,
        index.numel(),
        dim,
        (XPUType)(1.0f));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_add");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_add,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexAddKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   int64_t,
                   int32_t) {}
