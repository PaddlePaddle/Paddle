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
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
class FillConstantNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto str_value = ctx.Attr<std::string>("str_value");
    auto float_value = ctx.Attr<float>("value");

    auto *out_var = ctx.Output<framework::Tensor>("Out");
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

    out_var->mutable_data<T>(shape, ctx.GetPlace());
    if (data_type != framework::proto::VarType::BOOL) {
      Tensor tensor_value(framework::TransToPtenDataType(data_type));
      tensor_value.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&tensor_value, value);
      NpuOpRunner runner;
#if (CANN_VERSION_CODE >= 503003)
      runner.SetType("FillD")
          .AddInput(tensor_value)
          .AddOutput(*out_var)
          .AddAttrs(
              {{ "dims",
                 framework::vectorize(shape) }})
          .Run(stream);
#else
      runner.SetType("Fill")
          .AddInput(framework::vectorize(shape))
          .AddInput(tensor_value)
          .AddOutput(*out_var)
          .Run(stream);
#endif
    } else {
      const auto &dev_ctx =
          ctx.template device_context<paddle::platform::NPUDeviceContext>();
      auto op_func = [&shape, &value](
          const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
          const NPUAttributeMap &attrs,
          const platform::NPUDeviceContext &dev_ctx) {
        Tensor tensor_value;
        tensor_value.mutable_data<uint8_t>({1}, dev_ctx.GetPlace());
        FillNpuTensorWithConstant<uint8_t>(&tensor_value,
                                           static_cast<uint8_t>(value));

        NpuOpRunner runner;
        runner.SetType("Fill")
            .AddInput(framework::vectorize(shape))
            .AddInput(tensor_value)
            .AddOutput(outputs[0])
            .Run(dev_ctx.stream());
      };
      NpuOpRunner::TypeAdapter({}, {*out_var}, {}, dev_ctx, op_func, {},
                               {framework::proto::VarType::UINT8});
    }
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
