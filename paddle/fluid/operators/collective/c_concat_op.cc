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

#include "paddle/fluid/operators/collective/c_concat_op.h"

namespace paddle {
namespace operators {

class CConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "c_concat");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "c_concat");
    int nranks = ctx->Attrs().Get<int>("nranks");
    int rank = ctx->Attrs().Get<int>("rank");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(nranks, 2, platform::errors::InvalidArgument(
                                     "The number of ranks (%d) for c_concat "
                                     "must be greater than 1.",
                                     nranks));
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for c_concat must be non-negative.", ring_id));
    PADDLE_ENFORCE_GE(
        rank, 0, platform::errors::InvalidArgument(
                     "The rank (%d) for c_concat must be non-negative.", rank));
    PADDLE_ENFORCE_LT(rank, nranks,
                      platform::errors::InvalidArgument(
                          "The value of rank (%d) for c_concat must "
                          "be less than that of nranks.",
                          rank, nranks));

    framework::DDim dim = ctx->GetInputDim("X");
    dim[dim.size() - 1] = dim[dim.size() - 1] * nranks;
    if (dim[dim.size() - 1] < 0) dim[dim.size() - 1] = -1;
    ctx->SetOutputDim("Out", dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

template <typename T>
class CConcatOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("c_split");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

class CConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) tensor to be concated.");
    AddOutput("Out", "(Tensor) the result of concat.");
    AddAttr<int>("rank", "(int default 0) rank id.").SetDefault(0);
    AddAttr<int>("nranks", "(int default 1) number of ranks.").SetDefault(1);
    AddAttr<int>("ring_id", "(int default 0) ring id.").SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default true) eject CUDA operations to calculation stream.")
        .SetDefault(true);
    AddAttr<bool>("use_model_parallel",
                  "(bool default true) use this op with model parallel.")
        .SetDefault(true);
    AddComment(R"DOC(
CConcat Operator
AllGather the tensors on different trainers and concat them along the last dimension.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(c_concat, ops::CConcatOp,
                  ops::CConcatOpGradMaker<paddle::framework::OpDesc>,
                  ops::CConcatOpGradMaker<paddle::imperative::OpBase>,
                  ops::CConcatOpMaker);

REGISTER_OP_CPU_KERNEL(c_concat, ops::CConcatOpCPUKernel<float>,
                       ops::CConcatOpCPUKernel<double>,
                       ops::CConcatOpCPUKernel<int>,
                       ops::CConcatOpCPUKernel<int64_t>,
                       ops::CConcatOpCPUKernel<plat::float16>);
