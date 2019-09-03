/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <cmath>
#include <limits>
#include <type_traits>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillAnyLikeKernel : public framework::OpKernel<T> {
 public:
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<std::is_same<T, platform::float16>::value,
                                float, T>::type>::type;

  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    // TODO(fangzeyang): Once context.Attribute supports double dtype, this
    // kernel should be updated to support double dtype, too.
    float value = context.Attr<float>("value");

    auto common_type_value = static_cast<CommonType>(value);

    PADDLE_ENFORCE(
        (common_type_value >=
         static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
            (common_type_value <=
             static_cast<CommonType>(std::numeric_limits<T>::max())),
        "filled value is out of range for targeted type in fill_any_like "
        "kernel");

    PADDLE_ENFORCE(!std::isnan(value), "filled value is NaN");

    math::SetConstant<DeviceContext, T> setter;
    setter(context.template device_context<DeviceContext>(), out,
           static_cast<T>(value));
  }
};

}  // namespace operators
}  // namespace paddle
