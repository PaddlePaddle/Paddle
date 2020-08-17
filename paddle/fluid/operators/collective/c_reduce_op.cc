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

#include "paddle/fluid/operators/collective/c_reduce_op.h"

#include <memory>

namespace paddle {
namespace operators {

class CReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "CReduce");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "CReduce");
    int nranks = ctx->Attrs().Get<int>("nranks");
    int root_id = ctx->Attrs().Get<int>("root");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(nranks, 2,
                      platform::errors::InvalidArgument(
                          "The number of ranks (%d) must be greater than 1 "
                          "to use collective op (c_reduce op).",
                          nranks));
    PADDLE_ENFORCE_GE(
        root_id, 0,
        platform::errors::InvalidArgument(
            "The root_id (%d) for c_reduce_op must be positive.", root_id));
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for c_reduce_op must be positive.", root_id));
    framework::DDim dim = ctx->GetInputDim("X");
    if (dim[0] < 0) dim[0] = -1;
    ctx->SetOutputDim("Out", dim);
  }
};

class CReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) tensor to reduce");
    AddOutput("Out", "(Tensor) the result tensor");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
    AddAttr<int>("root", "(int default 0) the root rank id.").SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddAttr<int>("nranks",
                 "Total trainer count of the distributed training job");
    AddComment(R"DOC(
CReduce Operator
the source rank receives the aggregation of data from all ranks in the order of the ranks

reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#reduce
)DOC");
  }
};

template <typename T>
class CReduceOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("c_broadcast");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(c_reduce, ops::CReduceOp,
                  ops::CReduceOpGradMaker<paddle::framework::OpDesc>,
                  ops::CReduceOpGradMaker<paddle::imperative::OpBase>,
                  ops::CReduceOpMaker);

REGISTER_OP_CPU_KERNEL(c_reduce, ops::CReduceOpCPUKernel<float>,
                       ops::CReduceOpCPUKernel<double>,
                       ops::CReduceOpCPUKernel<int>,
                       ops::CReduceOpCPUKernel<int64_t>,
                       ops::CReduceOpCPUKernel<plat::float16>);
