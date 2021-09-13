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

#include <memory>
#include <string>

#include "paddle/fluid/operators/fill_constant_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillConstantNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto str_value = ctx.Attr<std::string>("str_value");
    auto float_value = ctx.Attr<float>("value");

    auto* out_var = ctx.Output<framework::Tensor>("Out");
    auto place = ctx.GetPlace();
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
    out_var->mutable_data<T>(shape, place);

    if (out_var->type() == framework::proto::VarType::INT64) {
      Tensor tensor_tmp(framework::proto::VarType::INT32);
      Tensor cast_out_var(framework::proto::VarType::INT32);
      tensor_tmp.mutable_data<int>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<int>(&tensor_tmp, static_cast<int>(value));
      cast_out_var.Resize(out_var->dims());
      cast_out_var.mutable_data<int>(ctx.GetPlace());

      const auto& runner = NpuOpRunner("FillD", {tensor_tmp}, {cast_out_var},
                                     {{"dims", framework::vectorize(shape)}});
      runner.Run(stream);

      auto dst_dtype = ConvertToNpuDtype(out_var->type());
      const auto& runner_cast_scale =
          NpuOpRunner("Cast", {cast_out_var}, {*out_var},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_scale.Run(stream);
    } else {
      Tensor tensor_tmp(data_type);
      tensor_tmp.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&tensor_tmp, value);

      const auto& runner = NpuOpRunner("FillD", {tensor_tmp}, {*out_var},
                                     {{"dims", framework::vectorize(shape)}});
      runner.Run(stream);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    fill_constant,
    ops::FillConstantNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::FillConstantNPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::FillConstantNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::FillConstantNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::FillConstantNPUKernel<paddle::platform::NPUDeviceContext,
                               paddle::platform::float16>);

