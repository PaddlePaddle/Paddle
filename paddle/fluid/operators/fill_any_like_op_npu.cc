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

#include "paddle/fluid/operators/fill_any_like_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillAnyLikeNPUKernel : public framework::OpKernel<T> {
 public:
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<std::is_same<T, platform::float16>::value,
                                float, T>::type>::type;

  void Compute(const framework::ExecutionContext& context) const override {
    auto data_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));
    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    float value = context.Attr<float>("value");

    auto common_type_value = static_cast<CommonType>(value);

    PADDLE_ENFORCE_EQ(
        (common_type_value >=
         static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
            (common_type_value <=
             static_cast<CommonType>(std::numeric_limits<T>::max())),
        true, platform::errors::InvalidArgument(
                  "filled value is out of range for"
                  " targeted type in fill_any_like, your kernel type is %s"
                  ", please check value you set.",
                  typeid(T).name()));

    PADDLE_ENFORCE_EQ(
        std::isnan(value), false,
        platform::errors::InvalidArgument("filled value should not be NaN,"
                                          " but received NaN"));

    auto shape = out->dims();

    Tensor tensor_tmp(data_type);
    tensor_tmp.mutable_data<T>({1}, context.GetPlace());
    FillNpuTensorWithConstant<T>(&tensor_tmp, static_cast<T>(value));

    out->mutable_data<T>(shape, context.GetPlace());

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("FillD", {tensor_tmp}, {*out},
                                     {{"dims", framework::vectorize(shape)}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    fill_any_like,
    ops::FillAnyLikeNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::FillAnyLikeNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::FillAnyLikeNPUKernel<paddle::platform::NPUDeviceContext,
                              paddle::platform::float16>);
