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

#include "paddle/fluid/operators/collective/c_comm_init_all_op.h"

namespace paddle::operators {

class CCommInitAllOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  }
};

class CCommInitAllOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
CCommInitAll operator

Initialize all collective communication context
)DOC");
    AddAttr<std::vector<int>>(
        "devices",
        "(std::vector<int>) which devices does the nccl comm initialized on")
        .SetDefault({});
    AddAttr<int>("ring_id", "(int default 0) user specified ring id")
        .SetDefault(0);
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_comm_init_all,
                  ops::CCommInitAllOp,
                  ops::CCommInitAllOpMaker);
