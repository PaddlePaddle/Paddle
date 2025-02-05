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
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/impl/full_with_tensor_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DataType dtype,
                DenseTensor* out) {
  using XPUInTDType = typename XPUTypeTrait<T>::Type;
  out->Resize(common::make_ddim(shape.GetData()));
  dev_ctx.template Alloc<T>(out);
  if (out->numel() > 0) {
    auto out_data = reinterpret_cast<XPUInTDType*>(out->data<T>());
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
  if (out->numel() > 0) {
    auto value = val.to<double>();
    using XPUInTDType = typename XPUTypeTrait<T>::Type;
    using CommonType = typename std::common_type<
        float,
        typename std::conditional<std::is_same<T, phi::dtype::float16>::value,
                                  float,
                                  T>::type>::type;

    auto common_type_value = static_cast<CommonType>(value);
    bool is_out_range = true;
    if (std::isinf(value) || std::isnan(value)) {
      is_out_range = false;
    }
    if ((common_type_value >=
         static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
        (common_type_value <=
         static_cast<CommonType>(std::numeric_limits<T>::max()))) {
      is_out_range = false;
    }

    PADDLE_ENFORCE_EQ(
        is_out_range,
        false,
        common::errors::InvalidArgument(
            "The filled value is out of range for target type, "
            "current kernel type is %s, the range should between %f "
            "and %f, but now value is %f.",
            typeid(T).name(),
            static_cast<CommonType>(std::numeric_limits<T>::lowest()),
            static_cast<CommonType>(std::numeric_limits<T>::max()),
            static_cast<float>(value)));

    auto out_data = reinterpret_cast<XPUInTDType*>(out->data<T>());
    int r = xpu::constant(dev_ctx.x_context(),
                          out_data,
                          out->numel(),
                          static_cast<XPUInTDType>(value));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  }
}

template <typename T, typename Context>
void FullBatchSizeLikeKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const std::vector<int>& shape,
                             const Scalar& val,
                             DataType dtype,
                             int x_batch_size_dim,
                             int out_batch_size_dim,
                             DenseTensor* out) {
  if (x.lod().size() && x_batch_size_dim == 0) {
    // set the correct batch size for the DenseTensor.
    auto odims = out->dims();
    odims[out_batch_size_dim] = static_cast<int>(x.lod().back().size()) - 1;
    FullKernel<T, Context>(dev_ctx, common::vectorize(odims), val, dtype, out);
  }
  FullLikeKernel<T, Context>(dev_ctx, x, val, dtype, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(full,
                   XPU,
                   ALL_LAYOUT,
                   phi::FullKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(full_like,
                   XPU,
                   ALL_LAYOUT,
                   phi::FullLikeKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(full_batch_size_like,
                   XPU,
                   ALL_LAYOUT,
                   phi::FullBatchSizeLikeKernel,
                   float,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(full_with_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::FullWithTensorKernel,
                   float,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);
}
