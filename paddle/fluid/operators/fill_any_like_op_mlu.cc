/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class FillAnyLikeMLUKernel : public framework::OpKernel<T> {
 public:
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<std::is_same<T, platform::float16>::value,
                                float,
                                T>::type>::type;

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    float value = ctx.Attr<float>("value");

    auto common_type_value = static_cast<CommonType>(value);

    PADDLE_ENFORCE_EQ(
        (common_type_value >=
         static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
            (common_type_value <=
             static_cast<CommonType>(std::numeric_limits<T>::max())),
        true,
        platform::errors::InvalidArgument(
            "The filled value is out of range for target type, "
            "current kernel type is %s, the range should between %f "
            "and %f, but now value is %f.",
            typeid(T).name(),
            static_cast<CommonType>(std::numeric_limits<T>::lowest()),
            static_cast<CommonType>(std::numeric_limits<T>::max()),
            value));

    PADDLE_ENFORCE_EQ(
        std::isnan(value),
        false,
        platform::errors::InvalidArgument("The filled value is NaN."));

    auto value_t = static_cast<T>(value);
    MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

    MLUCnnl::Fill(
        ctx, CNNL_POINTER_MODE_HOST, &value_t, out_desc.get(), GetBasePtr(out));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(fill_any_like,
                       ops::FillAnyLikeMLUKernel<int>,
                       ops::FillAnyLikeMLUKernel<int64_t>,
                       ops::FillAnyLikeMLUKernel<float>,
                       ops::FillAnyLikeMLUKernel<plat::float16>);
