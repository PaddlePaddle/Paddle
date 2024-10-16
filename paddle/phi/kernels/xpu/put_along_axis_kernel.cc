// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/put_along_axis_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

int64_t get_reduction_mode(const std::string& reduce) {
  if (reduce == "assign") {
    return 0;
  } else if (reduce == "add") {
    return 1;
  } else if (reduce == "multiply" || reduce == "mul") {
    return 2;
  } else if (reduce == "mean") {
    return 3;
  } else if (reduce == "amax") {
    return 4;
  } else if (reduce == "amin") {
    return 5;
  } else {
    PADDLE_THROW(errors::InvalidArgument(
        "can not support reduce: '%s' for put_along_axis kernel, only "
        "support reduce op: 'add', 'assign', 'mul', 'mean', 'amin', 'amax' and "
        "'multiply', the "
        "default reduce "
        "op is 'assign' ",
        reduce));
  }
}

template <typename T, typename Context>
void PutAlongAxisKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& index,
                        const DenseTensor& value,
                        int axis,
                        const std::string& reduce,
                        bool include_self,
                        DenseTensor* out) {
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0 || index.numel() == 0) return;

  const auto& index_dtype = index.dtype();
  bool index_dtype_match =
      index_dtype == DataType::INT32 || index_dtype == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_dtype_match,
                    true,
                    errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(index_dtype),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  auto input_dtype = x.dtype();
  std::vector<int64_t> x_shape = common::vectorize<int64_t>(x.dims());
  std::vector<int64_t> index_shape = common::vectorize<int64_t>(index.dims());
  std::vector<int64_t> value_shape = common::vectorize<int64_t>(value.dims());
  using XPUType = typename XPUTypeTrait<T>::Type;
  int64_t reduce_mode = get_reduction_mode(reduce);

  bool invalid_input =
      (input_dtype == DataType::INT32 || input_dtype == DataType::INT64) &&
      (!include_self || reduce_mode > 2);
  PADDLE_ENFORCE_EQ(invalid_input,
                    false,
                    errors::InvalidArgument(
                        "Only support include_self = true and reduce mode: "
                        "'add', 'assign' and 'multiply' for int32/int64"));

  PADDLE_ENFORCE_EQ(index.dims().size(),
                    value.dims().size(),
                    errors::InvalidArgument(
                        "The input(Index) and the input(Value) must have same "
                        "rank, but received Index rank is %d, Value rank is %d",
                        index.dims().size(),
                        value.dims().size()));

  if (index_dtype == DataType::INT32) {
    int ret = xpu::paddle_put_along_axis(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<const XPUType*>(value.data<T>()),
        index.data<int32_t>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        x_shape,
        value_shape,
        index_shape,
        axis,
        reduce_mode,
        include_self);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "paddle_put_along_axis");

  } else {
    int ret = xpu::paddle_put_along_axis(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<const XPUType*>(value.data<T>()),
        index.data<int64_t>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        x_shape,
        value_shape,
        index_shape,
        axis,
        reduce_mode,
        include_self);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "paddle_put_along_axis");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(put_along_axis,
                   XPU,
                   ALL_LAYOUT,
                   phi::PutAlongAxisKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float) {}
