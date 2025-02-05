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

#include "paddle/phi/kernels/linspace_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"

namespace phi {

template <typename T, typename Context>
T GetValueOfExpectedType(const Context& ctx, const DenseTensor& x) {
  switch (x.dtype()) {
    case DataType::FLOAT32:
      return static_cast<T>(GetValue<float, Context>(ctx, x));
    case DataType::FLOAT64:
      return static_cast<T>(GetValue<double, Context>(ctx, x));
    case DataType::INT32:
      return static_cast<T>(GetValue<int32_t, Context>(ctx, x));
    case DataType::INT64:
      return static_cast<T>(GetValue<int64_t, Context>(ctx, x));
    case DataType::FLOAT16:
      return static_cast<T>(GetValue<phi::dtype::float16, Context>(ctx, x));
    case DataType::BFLOAT16:
      return static_cast<T>(GetValue<phi::dtype::bfloat16, Context>(ctx, x));
    case DataType::BOOL:
      return static_cast<T>(GetValue<bool, Context>(ctx, x));
    case DataType::INT16:
      return static_cast<T>(GetValue<int16_t, Context>(ctx, x));
    case DataType::UINT8:
      return static_cast<T>(GetValue<uint8_t, Context>(ctx, x));
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          x.dtype()));
  }
}

template <typename T, typename Context>
void LinspaceKernel(const Context& ctx,
                    const DenseTensor& start,
                    const DenseTensor& stop,
                    const DenseTensor& number,
                    DataType dtype,
                    DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  T start_value = GetValueOfExpectedType<T, Context>(ctx, start);
  T stop_value = GetValueOfExpectedType<T, Context>(ctx, stop);
  int64_t num = GetValueOfExpectedType<int64_t, Context>(ctx, number);
  PADDLE_ENFORCE_GE(num,
                    0,
                    common::errors::InvalidArgument(
                        "The num of linspace op should be larger "
                        "than or equal to 0, but received num is %d",
                        num));

  out->Resize(common::make_ddim({num}));
  T* out_data = ctx.template Alloc<T>(out);
  if (num == 0) {
    return;
  }
  int r = xpu::linspace(ctx.x_context(),
                        reinterpret_cast<XPUType*>(out_data),
                        static_cast<XPUType>(start_value),
                        static_cast<XPUType>(stop_value),
                        num);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "linspace");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    linspace, XPU, ALL_LAYOUT, phi::LinspaceKernel, float, int32_t, int64_t) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}
