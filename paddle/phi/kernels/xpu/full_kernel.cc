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

#include "paddle/phi/kernels/full_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/common/memory_utils.h"

namespace phi {

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DataType dtype,
                DenseTensor* out) {
  using XPUInTDType = typename XPUTypeTrait<T>::Type;
  out->Resize(phi::make_ddim(shape.GetData()));
  int numel = out->numel();
  dev_ctx.template Alloc<T>(out);
  auto out_data = reinterpret_cast<XPUInTDType*>(out->data<T>());
  if (numel > 0) {
    int r = xpu::constant(dev_ctx.x_context(),
                          out_data,
                          out->numel(),
                          static_cast<XPUInTDType>(val.to<T>()));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  }
}

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const Scalar& val,
                    DataType dtype,
                    DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto value = val.to<double>();
  using XPUInTDType = typename XPUTypeTrait<T>::Type;
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<std::is_same<T, phi::dtype::float16>::value,
                                float,
                                T>::type>::type;

  auto common_type_value = static_cast<CommonType>(value);

  PADDLE_ENFORCE_EQ(
      (common_type_value >=
       static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
          (common_type_value <=
           static_cast<CommonType>(std::numeric_limits<T>::max())),
      true,
      phi::errors::InvalidArgument(
          "The filled value is out of range for target type, "
          "current kernel type is %s, the range should between %f "
          "and %f, but now value is %f.",
          typeid(T).name(),
          static_cast<CommonType>(std::numeric_limits<T>::lowest()),
          static_cast<CommonType>(std::numeric_limits<T>::max()),
          static_cast<float>(value)));

  PADDLE_ENFORCE_EQ(std::isnan(value),
                    false,
                    phi::errors::InvalidArgument("The filled value is NaN."));
  PADDLE_ENFORCE_EQ(std::isinf(value),
                    false,
                    phi::errors::InvalidArgument("The filled value is Inf."));

  auto out_data = reinterpret_cast<XPUInTDType*>(out->data<T>());
  int r = xpu::constant(dev_ctx.x_context(),
                        out_data,
                        out->numel(),
                        static_cast<XPUInTDType>(value));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
}

}  // namespace phi

PD_REGISTER_KERNEL(full,
                   XPU,
                   ALL_LAYOUT,
                   phi::FullKernel,
                   float,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(full_like,
                   XPU,
                   ALL_LAYOUT,
                   phi::FullLikeKernel,
                   float,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
