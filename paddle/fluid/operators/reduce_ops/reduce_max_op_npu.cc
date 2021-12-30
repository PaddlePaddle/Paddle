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

#include "paddle/fluid/operators/reduce_ops/reduce_min_max_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class ReduceMaxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    auto dims = ctx.Attr<std::vector<int>>("dim");
    bool keep_dim = ctx.Attr<bool>("keep_dim");
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    int out_dtype = ctx.Attr<int>("out_dtype");

    auto place = ctx.GetPlace();

    framework::Tensor cast_out(x->type());
    cast_out.Resize(out->dims());
    cast_out.mutable_data<T>(place);

    auto cast_out_dtype = x->type();
    if (out_dtype != -1) {
      cast_out_dtype = static_cast<framework::proto::VarType::Type>(out_dtype);
    }

    if (x->type() != cast_out_dtype) {
      if (cast_out_dtype == framework::proto::VarType::FP32) {
        out->mutable_data<float>(place);
      } else if (cast_out_dtype == framework::proto::VarType::FP16) {
        out->mutable_data<paddle::platform::float16>(place);
      } else if (cast_out_dtype == framework::proto::VarType::INT16) {
        out->mutable_data<int16_t>(place);
      } else if (cast_out_dtype == framework::proto::VarType::INT32) {
        out->mutable_data<int32_t>(place);
      } else if (cast_out_dtype == framework::proto::VarType::INT64) {
        out->mutable_data<int64_t>(place);
      } else if (cast_out_dtype == framework::proto::VarType::FP64) {
        out->mutable_data<double>(place);
      } else if (cast_out_dtype == framework::proto::VarType::BOOL) {
        out->mutable_data<bool>(place);
      }
    } else {
      out->ShareDataWith(cast_out);
    }

    framework::NPUAttributeMap attr_input = {{"axes", dims},
                                             {"keep_dims", keep_dim}};

    if (reduce_all) {
      std::vector<int> dim_vec;
      for (int i = 0; i < x->dims().size(); i++) {
        dim_vec.push_back(i);
      }

      attr_input = {{"axes", dim_vec}, {"keep_dims", keep_dim}};
    }

    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    if (x->type() == framework::proto::VarType::INT64) {
      auto op_func = [](const std::vector<Tensor>& inputs,
                        const std::vector<Tensor>& outputs,
                        const NPUAttributeMap& attrs,
                        const platform::NPUDeviceContext& dev_ctx) {
        const auto& runner =
            NpuOpRunner("ReduceMaxD", {inputs[0]}, {outputs[0]}, attrs);
        runner.Run(dev_ctx.stream());
      };

      NpuOpRunner::TypeAdapter({*x}, {cast_out}, attr_input, dev_ctx, op_func,
                               {framework::proto::VarType::INT32},
                               {framework::proto::VarType::INT32});
    } else {
      const auto& runner =
          NpuOpRunner("ReduceMaxD", {*x}, {cast_out}, attr_input);
      runner.Run(dev_ctx.stream());
    }

    if (x->type() != cast_out_dtype) {
      auto dst_dtype = ConvertToNpuDtype(cast_out_dtype);
      const auto& runner_cast =
          NpuOpRunner("Cast", {cast_out}, {*out},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast.Run(dev_ctx.stream());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    reduce_max, ops::ReduceMaxNPUKernel<plat::NPUDeviceContext, float>,
    ops::ReduceMaxNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::ReduceMaxNPUKernel<plat::NPUDeviceContext, int64_t>,
    ops::ReduceMaxNPUKernel<plat::NPUDeviceContext, int>);
