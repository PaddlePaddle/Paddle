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

#include "paddle/fluid/operators/one_hot_v2_op.h"

#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T>
class OneHotV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto* in = ctx.Input<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");
    int depth = ctx.Attr<int>("depth");

    if (ctx.HasInput("depth_tensor")) {
      auto* depth_tensor = ctx.Input<Tensor>("depth_tensor");
      std::vector<int32_t> depth_data;
      framework::TensorToVector(*depth_tensor, dev_ctx, &depth_data);
      depth = depth_data[0];
      auto out_dims = out->dims();
      out_dims[out_dims.size() - 1] = depth;
      out->Resize(out_dims);
    }
    out->mutable_data<float>(ctx.GetPlace());

    float on_value = 1.0f, off_value = 0.0f;
    if (framework::TransToProtoVarType(in->dtype()) ==
        framework::proto::VarType::INT32) {
      NpuOpRunner runner;
      runner.SetType("OneHot")
          .AddInput(*in)
          .AddInput(std::vector<int32_t>({static_cast<int32_t>(depth)}))
          .AddInput(std::vector<float>({on_value}))
          .AddInput(std::vector<float>({off_value}))
          .AddAttr("axis", -1)
          .AddOutput(*out);
      runner.Run(dev_ctx.stream());
    } else {
      Tensor transformed_in;
      transformed_in.mutable_data<int32_t>(in->dims(), dev_ctx.GetPlace());
      const auto& cast_runner = NpuOpRunner("Cast", {*in}, {transformed_in},
                                            {{"dst_type", ACL_INT32}});
      cast_runner.Run(dev_ctx.stream());
      NpuOpRunner runner;
      runner.SetType("OneHot")
          .AddInput(transformed_in)
          .AddInput(std::vector<int32_t>({static_cast<int32_t>(depth)}))
          .AddInput(std::vector<float>({on_value}))
          .AddInput(std::vector<float>({off_value}))
          .AddAttr("axis", -1)
          .AddOutput(*out);
      runner.Run(dev_ctx.stream());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(one_hot_v2, ops::OneHotV2NPUKernel<int32_t>,
                       ops::OneHotV2NPUKernel<int64_t>);
