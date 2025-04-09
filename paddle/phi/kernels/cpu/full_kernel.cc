/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/full_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/impl/full_with_tensor_kernel_impl.h"

namespace phi {

template <typename T, typename Context, typename VType>
void FullValue(const Context& dev_ctx, DenseTensor* tensor, VType val) {
  dev_ctx.template Alloc<T>(tensor);
  if (tensor->numel() == 0) {
    return;
  }
  auto t = phi::EigenVector<T>::Flatten(*tensor);
  t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(val));
}

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DataType dtype UNUSED,
                DenseTensor* out) {
  out->Resize(common::make_ddim(shape.GetData()));
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  FullValue<T>(dev_ctx, out, val.to<T>());
}

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const DenseTensor& x UNUSED,
                    const Scalar& val,
                    DataType dtype UNUSED,
                    DenseTensor* out) {
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    out->Resize(x.dims());
    return;
  }
  if (!std::is_same<T, phi::dtype::complex<float>>::value &&
      !std::is_same<T, phi::dtype::complex<double>>::value) {
    auto value = val.to<double>();
    using CommonType = typename std::common_type<
        float,
        typename std::conditional<std::is_same<T, phi::dtype::float16>::value,
                                  float,
                                  T>::type>::type;

    auto common_type_value = static_cast<CommonType>(value);

    // Check whether the filled value is valid
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
    FullValue<T>(dev_ctx, out, value);
  } else {
    FullValue<T>(dev_ctx, out, val.to<T>());
  }
}

template <typename T, typename Context>
void FullIntArrayKernel(const Context& dev_ctx,
                        const std::vector<int64_t>& shape,
                        DataType dtype UNUSED,
                        DenseTensor* out) {
  out->Resize(common::make_ddim({static_cast<int64_t>(shape.size())}));
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    int64_t val = shape[i];
    out_data[i] = static_cast<T>(val);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(full,
                   CPU,
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
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(full_like,
                   CPU,
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
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(
    full_int_array, CPU, ALL_LAYOUT, phi::FullIntArrayKernel, int, int64_t) {}

PD_REGISTER_KERNEL(full_with_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::FullWithTensorKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);
}
