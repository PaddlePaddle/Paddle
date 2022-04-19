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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"

#include "paddle/phi/kernels/full_kernel.h"

namespace paddle {
namespace operators {

template <typename T>
class FillAnyLikeXPUKernel : public framework::OpKernel<T> {
 public:
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<std::is_same<T, platform::float16>::value,
                                float, T>::type>::type;
  using XPUInTDType = typename XPUTypeTrait<T>::Type;

  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    float value = context.Attr<float>("value");

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
            static_cast<CommonType>(std::numeric_limits<T>::max()), value));

    PADDLE_ENFORCE_EQ(
        std::isnan(value), false,
        platform::errors::InvalidArgument("The filled value is NaN."));

    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();

    // call phi kernel
    phi::FullLikeKernel<T>(
        static_cast<const typename paddle::framework::ConvertToPhiContext<
            paddle::platform::XPUDeviceContext>::TYPE&>(dev_ctx),
        *x, value, phi::DataType::UNDEFINED, out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(fill_any_like, ops::FillAnyLikeXPUKernel<int>,
                       ops::FillAnyLikeXPUKernel<int64_t>,
                       ops::FillAnyLikeXPUKernel<float>,
                       ops::FillAnyLikeXPUKernel<paddle::platform::float16>);

#endif
