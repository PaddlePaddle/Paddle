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

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ScatterNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* index = ctx.Input<Tensor>("Ids");
    auto* updates = ctx.Input<Tensor>("Updates");
    bool overwrite = ctx.Attr<bool>("overwrite");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    framework::Tensor tmp_tensor(index->type());
    const auto index_dims = index->dims();
    if (index_dims.size() == 1) {
      tmp_tensor.ShareDataWith(*index);
      std::vector<int64_t> new_dim = {index_dims[0], 1};
      tmp_tensor.Resize(phi::make_ddim(new_dim));
      index = &tmp_tensor;
    }

    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto op_func_update = [](const std::vector<Tensor>& inputs,
                             const std::vector<Tensor>& outputs,
                             const NPUAttributeMap& attrs,
                             const platform::NPUDeviceContext& dev_ctx) {
      const auto& runner =
          NpuOpRunner("TensorScatterUpdate", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
    auto op_func_add = [](const std::vector<Tensor>& inputs,
                          const std::vector<Tensor>& outputs,
                          const NPUAttributeMap& attrs,
                          const platform::NPUDeviceContext& dev_ctx) {
      const auto& runner =
          NpuOpRunner("TensorScatterAdd", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };

    if (overwrite) {
      if (framework::TransToProtoVarType(x->dtype()) ==
          framework::proto::VarType::INT64) {
        NpuOpRunner::TypeAdapter(
            {*x, *index, *updates}, {*out}, {}, dev_ctx, op_func_update,
            {framework::proto::VarType::INT32, framework::proto::VarType::INT32,
             framework::proto::VarType::INT32},
            {framework::proto::VarType::INT32});
      } else {
        const auto& runner_update = NpuOpRunner(
            "TensorScatterUpdate", {*x, *index, *updates}, {*out}, {});
        runner_update.Run(dev_ctx.stream());
      }
    } else {
      if (framework::TransToProtoVarType(x->dtype()) ==
          framework::proto::VarType::INT64) {
        NpuOpRunner::TypeAdapter(
            {*x, *index, *updates}, {*out}, {}, dev_ctx, op_func_add,
            {framework::proto::VarType::INT32, framework::proto::VarType::INT32,
             framework::proto::VarType::INT32},
            {framework::proto::VarType::INT32});
      } else {
        const auto& runner_add =
            NpuOpRunner("TensorScatterAdd", {*x, *index, *updates}, {*out}, {});
        runner_add.Run(dev_ctx.stream());
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    scatter, ops::ScatterNPUKernel<paddle::platform::NPUDeviceContext, float>,
#ifdef PADDLE_WITH_ASCEND_INT64
    ops::ScatterNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
#endif
    ops::ScatterNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::ScatterNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);
#endif
