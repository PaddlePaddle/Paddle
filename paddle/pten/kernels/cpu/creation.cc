// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/cpu/creation.h"

#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/hybird/eigen/fill.h"

namespace pten {

template <typename T>
void FillAnyLike(const CPUContext& dev_ctx,
                 const Scalar& val,
                 DenseTensor* out) {
  auto value = val.to<float>();
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<
          std::is_same<T, paddle::platform::float16>::value,
          float,
          T>::type>::type;

  auto common_type_value = static_cast<CommonType>(value);

  PADDLE_ENFORCE_EQ(
      (common_type_value >=
       static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
          (common_type_value <=
           static_cast<CommonType>(std::numeric_limits<T>::max())),
      true,
      paddle::platform::errors::InvalidArgument(
          "The filled value is out of range for target type, "
          "current kernel type is %s, the range should between %f "
          "and %f, but now value is %f.",
          typeid(T).name(),
          static_cast<CommonType>(std::numeric_limits<T>::lowest()),
          static_cast<CommonType>(std::numeric_limits<T>::max()),
          static_cast<float>(value)));
  eigen::fill<CPUContext, T>(dev_ctx, out, value);
}

template <typename T>
void FillConstant(const CPUContext& dev_ctx,
                  const ScalarArray& shape,
                  const Scalar& val,
                  DenseTensor* out) {
  out->Resize(paddle::framework::make_ddim(shape.GetData()));
  eigen::fill<CPUContext, T>(dev_ctx, out, val.to<T>());
}

}  // namespace pten

PT_REGISTER_KERNEL(full_like,
                   CPU,
                   ANY,
                   pten::FillAnyLike,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   paddle::platform::float16) {}

PT_REGISTER_KERNEL(full,
                   CPU,
                   ANY,
                   pten::FillConstant,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   paddle::platform::float16,
                   paddle::platform::bfloat16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}
