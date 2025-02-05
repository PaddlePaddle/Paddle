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

#include "paddle/phi/kernels/index_select_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {

template <typename T, typename Context>
void IndexSelectKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& index,
                       int dim,
                       DenseTensor* output) {
  auto input_dim = x.dims();
  dim = dim >= 0 ? dim : dim + input_dim.size();
  const auto& index_type = index.dtype();

  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* in_data = x.data<T>();
  std::vector<int64_t> in_shape = common::vectorize<int64_t>(input_dim);
  int index_len = output->dims()[dim];
  ctx.template Alloc<T>(output);
  int r = 0;
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  int8_t* index_ptr = nullptr;  // temp xpu buffer
  int byte_times = SizeOf(index_type);
  if (index.place() == CPUPlace()) {
    index_ptr = RAII_GUARD.alloc_l3_or_gm<int8_t>(byte_times * index.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(index_ptr);
    const void* cpu_idx_data = nullptr;
    if (index_type == phi::DataType::INT64) {
      cpu_idx_data = reinterpret_cast<const void*>(index.data<int64_t>());
    } else if (index_type == phi::DataType::INT32) {
      cpu_idx_data = reinterpret_cast<const void*>(index.data<int>());
    }
    memory_utils::Copy(ctx.GetPlace(),
                       reinterpret_cast<void*>(index_ptr),
                       CPUPlace(),
                       cpu_idx_data,
                       byte_times * index.numel());
  }
  if (index_type == phi::DataType::INT64) {
    const int64_t* index_data =
        index_ptr ? reinterpret_cast<const int64_t*>(index_ptr)
                  : index.template data<int64_t>();
    r = xpu::paddle_gather<XPUType, int64_t>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(in_data),
        reinterpret_cast<const int64_t*>(index_data),
        reinterpret_cast<XPUType*>(output->data<T>()),
        in_shape,
        index_len,
        dim);
  } else {
    const int* index_data = index_ptr ? reinterpret_cast<const int*>(index_ptr)
                                      : index.template data<int>();
    r = xpu::paddle_gather<XPUType, int>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(in_data),
        reinterpret_cast<const int*>(index_data),
        reinterpret_cast<XPUType*>(output->data<T>()),
        in_shape,
        index_len,
        dim);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");
}

}  // namespace phi

PD_REGISTER_KERNEL(index_select,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexSelectKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t) {}
