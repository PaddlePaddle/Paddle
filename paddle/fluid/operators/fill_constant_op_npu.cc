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

#include "paddle/fluid/operators/fill_constant_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

template <typename T>
class FillConstantNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto str_value = ctx.Attr<std::string>("str_value");
    auto float_value = ctx.Attr<float>("value");

    auto* out_var = ctx.Output<framework::Tensor>("Out");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

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
    auto shape = GetShape(ctx);

    Tensor tensor_value(data_type);
    tensor_value.mutable_data<T>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<T>(&tensor_value, value);

    out_var->mutable_data<T>(shape, ctx.GetPlace());

    NpuOpRunner runner;
    runner.SetType("Fill")
        .AddInput(framework::vectorize(shape))
        .AddInput(tensor_value)
        .AddOutput(*out_var)
        .Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_NPU_KERNEL(
    fill_constant, paddle::operators::FillConstantNPUKernel<float>,
    paddle::operators::FillConstantNPUKernel<bool>,
    paddle::operators::FillConstantNPUKernel<int>,
#ifdef PADDLE_WITH_ASCEND_INT64
    paddle::operators::FillConstantNPUKernel<int64_t>,
#endif
    paddle::operators::FillConstantNPUKernel<paddle::platform::float16>);
