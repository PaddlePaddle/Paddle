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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
namespace phi {

template <typename InT, typename OutT = InT>
struct FullFunctor {
  OutT value;

  template <typename VType>
  explicit inline FullFunctor(VType val) {
    value = static_cast<OutT>(val);
  }

  __device__ __forceinline__ OutT operator()() const {
    return static_cast<OutT>(value);
  }
};

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DataType dtype,
                DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  int numel = out->numel();
  dev_ctx.template Alloc<T>(out);
  if (numel > 0) {
    // in transformer model the numel of outpout will be zero.
    std::vector<const DenseTensor*> inputs = {};
    std::vector<DenseTensor*> outputs = {out};
    // This function has no input, so the inputs.size() == 0. Use kUnary, but
    // the data will not be loaded in the kernel because the number of
    // parameters in the operator is 0
    phi::funcs::ElementwiseKernel<T>(
        dev_ctx, inputs, &outputs, FullFunctor<T>(val.to<T>()));
  }
}

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const Scalar& val,
                    DataType dtype,
                    DenseTensor* out) {
  auto value = val.to<double>();
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<
          std::is_same<T, phi::dtype::float16>::value ||
              std::is_same<T, phi::dtype::bfloat16>::value,
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
      phi::errors::InvalidArgument(
          "The filled value is out of range for target type, "
          "current kernel type is %s, the range should between %f "
          "and %f, but now value is %f.",
          typeid(T).name(),
          static_cast<CommonType>(std::numeric_limits<T>::lowest()),
          static_cast<CommonType>(std::numeric_limits<T>::max()),
          static_cast<float>(value)));
  std::vector<const DenseTensor*> inputs = {};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  // This function has no input, so the inputs.size() == 0. Use kUnary, but the
  // data will not be loaded in the kernel because the number of parameters in
  // the operator is 0
  int numel = out->numel();
  if (numel > 0) {
    phi::funcs::ElementwiseKernel<T>(
        dev_ctx, inputs, &outputs, FullFunctor<T>(value));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(full,
                   GPU,
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
                   GPU,
                   ALL_LAYOUT,
                   phi::FullLikeKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
