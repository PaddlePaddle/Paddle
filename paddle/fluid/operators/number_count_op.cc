// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/number_count_op.h"

namespace paddle {
namespace operators {

class NumberCountOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("numbers"), "Input", "numbers", "NumberCount");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "number_count",
                   "NumberCount");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // the dtype of the numbers should be same as int64
    auto number_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "numbers");

    PADDLE_ENFORCE_EQ(number_dtype, framework::proto::VarType::INT64,
                      platform::errors::InvalidArgument(
                          "The dtype of the number_dtype should be int64"));
    return framework::OpKernelType(number_dtype, ctx.GetPlace());
  }
};

class NumberCountOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("numbers", "(Tensor) The input gate index tensor.");
    AddOutput("Out", "(Tensor) The output number count tensor.");
    AddAttr<int>("upper_range", "（int), The number of different numbers.");

    AddComment(R"DOC(number_count Operator.count numbers.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CPU_KERNEL(number_count, ops::NumberCountOpCPUKernel<int>,
                       ops::NumberCountOpCPUKernel<int64_t>);

REGISTER_OP_WITHOUT_GRADIENT(number_count, ops::NumberCountOp,
                             ops::NumberCountOpMaker);
