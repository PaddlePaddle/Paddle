/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <unordered_map>

#include "paddle/fluid/operators/data/random_flip_op.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {
namespace data {

using framework::OpKernelType;
using framework::Tensor;

class RandomFlipOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of RandomFlipOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of RandomFlipOp should not be null."));

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", framework::make_ddim({x_dims[0], 1}));
    ctx->ShareLoD("X", "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type,
                                   platform::CPUPlace());
  }
};

class RandomFlipOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of flip op.");
    AddOutput("Out", "(Tensor), The output tensor in shape of [N, 1], N is "
                     "the batch size of X, bool data indicates whether to "
                     "perform flip in this sample.");
    AddAttr<float>("probability", "The probability to flip each sample.")
        .SetDefault(0.5);
    AddAttr<int>("seed", "The seed for uniform random generator")
        .SetDefault(0);
    AddComment(R"DOC(
          Random Flip Operator.
      )DOC");
  }
};

class RandomFlipOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::data;
namespace plat = paddle::platform;
REGISTER_OPERATOR(random_flip, ops::RandomFlipOp, ops::RandomFlipOpMaker, ops::RandomFlipOpInferVarType);

REGISTER_OP_CPU_KERNEL(
    random_flip, ops::RandomFlipCPUKernel<float>,
    ops::RandomFlipCPUKernel<double>,
    ops::RandomFlipCPUKernel<uint8_t>);
