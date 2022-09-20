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
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

template <typename T>
class FillConstantMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto str_value = ctx.Attr<std::string>("str_value");
    auto float_value = ctx.Attr<float>("value");

    auto *out_var = ctx.Output<framework::Tensor>("Out");

    T value;
    if (str_value.empty()) {
      value = static_cast<T>(float_value);
    } else {
      // handle NaN/Inf first, which cannot be read from stream.
      if (str_value == "inf") {
        value = static_cast<T>(std::numeric_limits<double>::infinity());
      } else if (str_value == "-inf") {
        value = static_cast<T>(-std::numeric_limits<double>::infinity());
      } else if (str_value == "nan") {
        value = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
      } else {
        std::stringstream convert_stream(str_value);
        if (std::is_same<int64_t, T>::value) {
          int64_t tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        } else {
          double tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        }
      }
    }
    const T *value_data = &value;
    cnnlPointerMode_t pointer_mode = CNNL_POINTER_MODE_HOST;
    if (ctx.HasInput("ValueTensor")) {
      auto *value_tensor = ctx.Input<framework::Tensor>("ValueTensor");
      PADDLE_ENFORCE_EQ(
          value_tensor->numel(),
          1,
          platform::errors::InvalidArgument(
              "When use Tensor as value to set Tensor value in fill_cosntant, "
              "value input(ValueTensor) size must be 1, but get %d",
              value_tensor->numel()));
      value_data = value_tensor->data<T>();
      auto tmp_place = value_tensor->place();
      if (platform::is_mlu_place(tmp_place)) {
        pointer_mode = CNNL_POINTER_MODE_DEVICE;
      }
    }

    auto shape = GetShape(ctx);
    out_var->mutable_data<T>(shape, ctx.GetPlace());
    MLUCnnlTensorDesc output_desc(*out_var);
    MLUCnnl::Fill(
        ctx, pointer_mode, value_data, output_desc.get(), GetBasePtr(out_var));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(
    fill_constant,
    paddle::operators::FillConstantMLUKernel<float>,
    paddle::operators::FillConstantMLUKernel<bool>,
    paddle::operators::FillConstantMLUKernel<int>,
    paddle::operators::FillConstantMLUKernel<uint8_t>,
    paddle::operators::FillConstantMLUKernel<int16_t>,
    paddle::operators::FillConstantMLUKernel<int64_t>,
    paddle::operators::FillConstantMLUKernel<paddle::platform::float16>);
