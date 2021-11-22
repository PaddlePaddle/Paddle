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

#include "paddle/fluid/operators/kron_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/scatter_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static void CastToInt64(const framework::ExecutionContext& ctx,
                        const aclrtStream& stream, const Tensor& in,
                        Tensor* out) {
  out->mutable_data<int64_t>(ctx.GetPlace());
  NpuOpRunner runner;
  runner.SetType("Cast")
      .AddInput(in)
      .AddOutput(*out)
      .AddAttr("dst_type", ACL_INT64)
      .Run(stream);
}

static void CastToFP32(const framework::ExecutionContext& ctx,
                       const aclrtStream& stream, const Tensor& in,
                       Tensor* out) {
  out->mutable_data<float>(ctx.GetPlace());
  NpuOpRunner runner;
  runner.SetType("Cast")
      .AddInput(in)
      .AddOutput(*out)
      .AddAttr("dst_type", ACL_FLOAT)
      .Run(stream);
}

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
    // out->mutable_data<T>(place);

    framework::Tensor tmp_tensor(index->type());
    const auto index_dims = index->dims();
    if (index_dims.size() == 1) {
      tmp_tensor.ShareDataWith(*index);
      std::vector<int64_t> new_dim = {index_dims[0], 1};
      tmp_tensor.Resize(framework::make_ddim(new_dim));
      index = &tmp_tensor;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (x->type() == framework::proto::VarType::INT64) {
      LOG(WARNING) << "x->type(): " << x->type();

      Tensor x_fp32(framework::proto::VarType::FP32);
      x_fp32.Resize(x->dims());
      CastToFP32(ctx, stream, *x, &x_fp32);

      Tensor updates_fp32(framework::proto::VarType::FP32);
      updates_fp32.Resize(updates->dims());
      CastToFP32(ctx, stream, *updates, &updates_fp32);

      Tensor out_fp32(framework::proto::VarType::FP32);
      out_fp32.Resize(out->dims());
      out_fp32.mutable_data<float>(ctx.GetPlace());

      if (overwrite) {
        const auto& runner_update =
            NpuOpRunner("TensorScatterUpdate", {x_fp32, *index, updates_fp32},
                        {out_fp32}, {});
        runner_update.Run(stream);
      } else {
        const auto& runner_add = NpuOpRunner(
            "TensorScatterAdd", {x_fp32, *index, updates_fp32}, {out_fp32}, {});
        runner_add.Run(stream);
      }

      CastToInt64(ctx, stream, out_fp32, out);
    } else {
      out->mutable_data<T>(place);

      if (overwrite) {
        const auto& runner_update = NpuOpRunner(
            "TensorScatterUpdate", {*x, *index, *updates}, {*out}, {});
        runner_update.Run(stream);
      } else {
        const auto& runner_add =
            NpuOpRunner("TensorScatterAdd", {*x, *index, *updates}, {*out}, {});
        runner_add.Run(stream);
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
    ops::ScatterNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);
#endif
