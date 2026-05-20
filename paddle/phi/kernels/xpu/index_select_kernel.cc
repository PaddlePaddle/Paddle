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

#include <type_traits>
#include <vector>

namespace phi {

template <typename Context, typename IndexT>
void ValidateIndexSelectIndex(const Context& dev_ctx,
                              const DenseTensor& index,
                              int64_t axis_dim) {
  const IndexT* index_data = nullptr;
  std::vector<IndexT> index_cpu_data;
  if (index.place().GetType() == AllocationType::CPU) {
    index_data = index.data<IndexT>();
  } else {
    index_cpu_data.resize(index.numel());
    memory_utils::Copy(CPUPlace(),
                       index_cpu_data.data(),
                       dev_ctx.GetPlace(),
                       index.data<IndexT>(),
                       sizeof(IndexT) * index.numel());
    index_data = index_cpu_data.data();
  }

  for (int64_t i = 0; i < index.numel(); ++i) {
    PADDLE_ENFORCE_GE(
        index_data[i],
        -axis_dim,
        common::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= %ld and < %ld, but got %ld. Please check input "
            "value.",
            -axis_dim,
            axis_dim,
            static_cast<int64_t>(index_data[i])));
    PADDLE_ENFORCE_LT(
        index_data[i],
        axis_dim,
        common::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= %ld and < %ld, but got %ld. Please check input "
            "value.",
            -axis_dim,
            axis_dim,
            static_cast<int64_t>(index_data[i])));
  }
}

template <typename T, typename Context>
void IndexSelectKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& index,
                       int dim,
                       DenseTensor* output) {
  if (output && output->numel() == 0) {
    dev_ctx.template Alloc<T>(output);
    return;
  }
  auto input_dim = x.dims();
  dim = dim >= 0 ? dim : dim + input_dim.size();
  const auto& index_type = index.dtype();

  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        DataType::INT32,
                        DataType::INT64));
  PADDLE_ENFORCE_EQ(
      (std::is_same<T, float>::value || std::is_same<T, phi::float16>::value ||
       std::is_same<T, phi::bfloat16>::value || std::is_same<T, int>::value ||
       std::is_same<T, int64_t>::value),
      true,
      common::errors::Unimplemented(
          "XPU index_select only supports float32, float16, bfloat16, int32 "
          "and int64 input tensors for the XDNN kernel, but got %s.",
          x.dtype()));
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* in_data = x.data<T>();
  std::vector<int64_t> in_shape = vectorize<int64_t>(input_dim);
  int64_t index_len = output->dims()[dim];
  dev_ctx.template Alloc<T>(output);
  int r = 0;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int8_t* index_ptr = nullptr;  // temp xpu buffer
  int byte_times = SizeOf(index_type);
  if (index.place() == CPUPlace()) {
    index_ptr = RAII_GUARD.alloc_l3_or_gm<int8_t>(byte_times * index.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(index_ptr);
    const void* cpu_idx_data = nullptr;
    if (index_type == DataType::INT64) {
      cpu_idx_data = reinterpret_cast<const void*>(index.data<int64_t>());
    } else if (index_type == DataType::INT32) {
      cpu_idx_data = reinterpret_cast<const void*>(index.data<int>());
    }
    memory_utils::Copy(dev_ctx.GetPlace(),
                       reinterpret_cast<void*>(index_ptr),
                       CPUPlace(),
                       cpu_idx_data,
                       byte_times * index.numel());
  }
  if (index_type == DataType::INT64) {
    // XDNN may crash on invalid indices, so match CPU validation first.
    ValidateIndexSelectIndex<Context, int64_t>(dev_ctx, index, input_dim[dim]);
    const int64_t* index_data =
        index_ptr ? reinterpret_cast<const int64_t*>(index_ptr)
                  : index.template data<int64_t>();
    r = xpu::index_select<XPUType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(in_data),
        reinterpret_cast<const int64_t*>(index_data),
        reinterpret_cast<XPUType*>(output->data<T>()),
        in_shape,
        index_len,
        dim);
  } else {
    // XDNN may crash on invalid indices, so match CPU validation first.
    ValidateIndexSelectIndex<Context, int>(dev_ctx, index, input_dim[dim]);
    const int* index_data = index_ptr ? reinterpret_cast<const int*>(index_ptr)
                                      : index.template data<int>();
    r = xpu::index_select<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(in_data),
        reinterpret_cast<const int*>(index_data),
        reinterpret_cast<XPUType*>(output->data<T>()),
        in_shape,
        index_len,
        dim);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_select");
}

}  // namespace phi

PD_REGISTER_KERNEL(index_select,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexSelectKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   int64_t) {}
