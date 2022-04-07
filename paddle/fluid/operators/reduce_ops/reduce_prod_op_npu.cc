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

#include "paddle/fluid/operators/reduce_ops/reduce_prod_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class ReduceProdNPUKernel : public framework::OpKernel<T> {
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

    auto cast_out_dtype = framework::TransToProtoVarType(x->dtype());
    if (out_dtype != -1) {
      cast_out_dtype = static_cast<framework::proto::VarType::Type>(out_dtype);
    }

    if (framework::TransToProtoVarType(x->dtype()) != cast_out_dtype) {
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

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner =
        NpuOpRunner("ReduceProdD", {*x}, {cast_out}, attr_input);
    runner.Run(stream);

    if (framework::TransToProtoVarType(x->dtype()) != cast_out_dtype) {
      auto dst_dtype = ConvertToNpuDtype(cast_out_dtype);
      const auto& runner_cast =
          NpuOpRunner("Cast", {cast_out}, {*out},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    reduce_prod, ops::ReduceProdNPUKernel<plat::NPUDeviceContext, float>,
    ops::ReduceProdNPUKernel<plat::NPUDeviceContext, plat::float16>);
