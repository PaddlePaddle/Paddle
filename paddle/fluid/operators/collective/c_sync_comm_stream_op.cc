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
#include "paddle/fluid/operators/collective/c_sync_comm_stream_op.h"

namespace paddle {
namespace operators {

class CSyncCommStreamOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }
};

class CSyncCommStreamOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Dependency of the variable need to sync")
        .AsDuplicable();
    AddOutput("Out", "(Tensor) Dependency of the variable need to sync")
        .AsDuplicable();
    AddAttr<int>("ring_id", "(int default 0) ring id.").SetDefault(0);
    AddComment(R"DOC(
CSyncCommStream Operator
Call communication stream synchronization.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(c_sync_comm_stream,
                             ops::CSyncCommStreamOp,
                             ops::CSyncCommStreamOpMaker);

REGISTER_OP_CUDA_KERNEL(c_sync_comm_stream, ops::CSyncCommStreamKernel<float>);

REGISTER_OP_NPU_KERNEL(c_sync_comm_stream, ops::CSyncCommStreamKernel<float>);

REGISTER_OP_MLU_KERNEL(c_sync_comm_stream, ops::CSyncCommStreamKernel<float>);
