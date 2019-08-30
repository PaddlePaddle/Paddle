/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_slicegather_op.h"
#include <future>  // NOLINT
#include <ostream>

namespace paddle {
namespace operators {

class CSliceGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) of CSliceGather op should not be null");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of CSliceGather op should not be null.");
    int nranks = ctx->Attrs().Get<int>("nranks");
    auto in_dim = ctx->GetInputDim("X");
    in_dim[0] = in_dim[0] / nranks;
    in_dim[1] = in_dim[1] * nranks;
    ctx->SetOutputDim("Out", in_dim);
  }
};

class CSliceGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) tensor to be slicegather");
    AddOutput("Out", "(Tensor) the slicegather result");
    AddAttr<int>("ring_id", "(int) communication ring id.").SetDefault(0);
    AddAttr<int>("nranks",
                 "Total trainer count of the distributed training job")
        .SetDefault(1);
    AddComment(R"DOC(
***CSliceGather Operator***

Call NCCL collective  AllGather internally. Note that this op must be used when one
thread is managing one GPU device.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(c_slicegather, ops::CSliceGatherOp, ops::CSliceGatherOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<false>);

REGISTER_OP_CPU_KERNEL(
    c_slicegather, ops::CSliceGatherOpKernel<plat::CPUDeviceContext, float>,
    ops::CSliceGatherOpKernel<plat::CPUDeviceContext, double>,
    ops::CSliceGatherOpKernel<plat::CPUDeviceContext, int>,
    ops::CSliceGatherOpKernel<plat::CPUDeviceContext, int64_t>,
    ops::CSliceGatherOpKernel<plat::CPUDeviceContext, plat::float16>);
