// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/collective/rpc_token_result_op.h"

#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace operators {

class RpcTokenResultOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

class RpcTokenResultOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Request id.");
    AddOutput("Out", "(Tensor) Response from service.");
    AddOutput("succeed", "Request status, true means succeed.");
    AddComment(R"DOC(
Rpc Token Result Operator

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(rpc_token_result,
                             ops::RpcTokenResultOp,
                             ops::RpcTokenResultOpMaker);

REGISTER_OP_CPU_KERNEL(rpc_token_result, ops::RpcTokenResultOpKernel<int>);

REGISTER_OP_CUDA_KERNEL(rpc_token_result, ops::RpcTokenResultOpKernel<int>);
