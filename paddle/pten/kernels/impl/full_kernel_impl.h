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

#pragma once

#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/hybird/eigen/common.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/eigen/eigen_function.h"

namespace pten {

template <typename DeviceContext, typename T, typename VType>
void fill_(const DeviceContext& context, DenseTensor* tensor, VType val) {
  tensor->mutable_data<T>();
  auto t = pten::EigenVector<T>::Flatten(*tensor);
  t.device(*context.eigen_device()) = t.constant(static_cast<T>(val));
}

template <typename T, typename ContextT>
void Full(const ContextT& dev_ctx,
          const ScalarArray& shape,
          const Scalar& val,
          DenseTensor* out) {
  out->Resize(paddle::framework::make_ddim(shape.GetData()));
  fill_<ContextT, T>(dev_ctx, out, val.to<T>());
}

template <typename T, typename ContextT>
void FullLike(const ContextT& dev_ctx, const Scalar& val, DenseTensor* out) {
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
  fill_<ContextT, T>(dev_ctx, out, value);
}

}  // namespace pten
