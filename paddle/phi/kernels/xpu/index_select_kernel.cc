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

#include <type_traits>
#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {

template <typename Context, typename IndexT>
std::vector<IndexT> GetValidatedIndexSelectIndex(const Context& dev_ctx,
                                                 const DenseTensor& index,
                                                 int64_t axis_dim) {
  std::vector<IndexT> index_cpu_data(index.numel());
  if (index.place().GetType() == AllocationType::CPU) {
    const IndexT* index_data = index.data<IndexT>();
    index_cpu_data.assign(index_data, index_data + index.numel());
  } else {
    memory_utils::Copy(CPUPlace(),
                       index_cpu_data.data(),
                       dev_ctx.GetPlace(),
                       index.data<IndexT>(),
                       sizeof(IndexT) * index.numel());
  }

  for (int64_t i = 0; i < index.numel(); ++i) {
    PADDLE_ENFORCE_GE(
        index_cpu_data[i],
        -axis_dim,
        common::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= %ld and < %ld, but got %ld. Please check input "
            "value.",
            -axis_dim,
            axis_dim,
            static_cast<int64_t>(index_cpu_data[i])));
    PADDLE_ENFORCE_LT(
        index_cpu_data[i],
        axis_dim,
        common::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= %ld and < %ld, but got %ld. Please check input "
            "value.",
            -axis_dim,
            axis_dim,
            static_cast<int64_t>(index_cpu_data[i])));
    if (index_cpu_data[i] < 0) {
      index_cpu_data[i] += axis_dim;
    }
  }
  return index_cpu_data;
}

template <typename T, typename Context>
void IndexSelectKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& index,
                       int dim,
                       DenseTensor* output) {
  auto input_dim = x.dims();
  dim = dim >= 0 ? dim : dim + input_dim.size();
  if (input_dim[dim] == 0 && index.numel() > 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dimension of Input(X) on the select axis in OP(index_select) "
        "must be greater than 0 when Input(Index) is not empty."));
  }

  if (output && output->numel() == 0) {
    dev_ctx.template Alloc<T>(output);
    return;
  }
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
  int byte_times = SizeOf(index_type);
  if (index_type == DataType::INT64) {
    auto index_cpu_data = GetValidatedIndexSelectIndex<Context, int64_t>(
        dev_ctx, index, input_dim[dim]);
    int64_t* index_ptr =
        RAII_GUARD.alloc_l3_or_gm<int64_t>(index_cpu_data.size());
    PADDLE_ENFORCE_XDNN_NOT_NULL(index_ptr);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       index_ptr,
                       CPUPlace(),
                       index_cpu_data.data(),
                       byte_times * index.numel());
    r = xpu::index_select<XPUType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(in_data),
        index_ptr,
        reinterpret_cast<XPUType*>(output->data<T>()),
        in_shape,
        index_len,
        dim);
  } else {
    auto index_cpu_data = GetValidatedIndexSelectIndex<Context, int>(
        dev_ctx, index, input_dim[dim]);
    int* index_ptr = RAII_GUARD.alloc_l3_or_gm<int>(index_cpu_data.size());
    PADDLE_ENFORCE_XDNN_NOT_NULL(index_ptr);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       index_ptr,
                       CPUPlace(),
                       index_cpu_data.data(),
                       byte_times * index.numel());
    r = xpu::index_select<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(in_data),
        index_ptr,
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
