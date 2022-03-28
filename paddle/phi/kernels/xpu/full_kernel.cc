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

#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/memcpy.h"

namespace phi {

template <typename InType, typename OutType>
void TensorSetConstantXPU(phi::DenseTensor* tensor,
                          InType value,
                          phi::Place place) {
  auto* begin = tensor->mutable_data<OutType>(place);
  int64_t numel = tensor->numel();
  std::unique_ptr<OutType[]> data_cpu(new OutType[numel]);
  std::fill(
      data_cpu.get(), data_cpu.get() + numel, static_cast<OutType>(value));
  paddle::memory::Copy(place,
                       begin,
                       phi::CPUPlace(),
                       static_cast<void*>(data_cpu.get()),
                       numel * sizeof(OutType));
}

template <typename T, typename Context, typename VType>
void FullValueXPU(const Context& dev_ctx, DenseTensor* tensor, VType val) {
  tensor->mutable_data<T>(dev_ctx.GetPlace());

  PD_VISIT_ALL_TYPES(tensor->dtype(), "FullValueXPU", ([&] {
                       TensorSetConstantXPU<VType, data_t>(
                           tensor, val, dev_ctx.GetPlace());
                     }));
}

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const ScalarArray& shape,
                const Scalar& val,
                DataType dtype,
                DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  FullValueXPU<T>(dev_ctx, out, val.to<T>());
}

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const Scalar& val,
                    DataType dtype,
                    DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto value = val.to<float>();
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
  int ret = xpu::constant(dev_ctx.x_context(),
                          out_data,
                          out->numel(),
                          static_cast<XPUInTDType>(value));
  PADDLE_ENFORCE_EQ(
      ret,
      XPU_SUCCESS,
      phi::errors::External("XPU CONSTANT API return wrong value[%d %s].",
                            ret,
                            XPUAPIErrorMsg[ret]));
}

}  // namespace phi

PD_REGISTER_KERNEL(full,
                   XPU,
                   ALL_LAYOUT,
                   phi::FullKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(full_like,
                   XPU,
                   ALL_LAYOUT,
                   phi::FullLikeKernel,
                   float,
                   int,
                   int64_t,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
