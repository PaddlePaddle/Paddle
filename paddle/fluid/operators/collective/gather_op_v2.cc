/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/gather_op_v2.h"

namespace paddle {
namespace operators {

class GatherOpV2 : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "GatherV2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "GatherV2");
    int root_id = ctx->Attrs().Get<int>("root");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    int nranks = ctx->Attrs().Get<int>("nranks");
    PADDLE_ENFORCE_GE(nranks, 2,
                      platform::errors::InvalidArgument(
                          "The number of ranks (%d) must be greater than 1 "
                          "to use the collective op gather_op_v2.",
                          nranks));
    PADDLE_ENFORCE_GE(
        root_id, 0,
        platform::errors::InvalidArgument(
            "The root_id (%d) for gather_op_v2 must be non-negative.",
            root_id));
    PADDLE_ENFORCE_LT(
        root_id, nranks,
        platform::errors::InvalidArgument(
            "The root_id (%d) for gather_op_v2 must be less than nranks (%d).",
            root_id, nranks));
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for gather_op_v2 must be non-negative.",
            root_id));
    framework::DDim dim = ctx->GetInputDim("X");
    dim[0] = dim[0] * nranks;
    if (dim[0] < 0) dim[0] = -1;
    ctx->SetOutputDim("Out", dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class GatherOpV2Maker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) tensor to be gathered.");
    AddOutput("Out", "(Tensor) the result of gather.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("root", "(int default 0) root id for broadcasting.")
        .SetDefault(0);
    AddAttr<int>("nranks", "(int default 1) number of ranks.").SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(R"DOC(
Gather Operator
Gather tensors from all participators.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(gather_v2, ops::GatherOpV2, ops::GatherOpV2Maker);

REGISTER_OP_CPU_KERNEL(gather_v2, ops::CGatherOpV2CPUKernel<float>,
                       ops::GatherOpV2CPUKernel<double>,
                       ops::GatherOpV2CPUKernel<int>,
                       ops::GatherOpV2CPUKernel<int64_t>,
                       ops::GatherOpV2CPUKernel<plat::float16>);
