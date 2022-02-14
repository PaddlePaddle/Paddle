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

#include "paddle/fluid/operators/data/mirror_normalize_op.h"

namespace paddle {
namespace operators {
namespace data {

class MirrorNormalizeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of MirrorNormalizeOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Mirror"), true,
        platform::errors::NotFound("Input(Mirror) of MirrorNormalizeOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of MirrorNormalizeOp should not be null."));

    auto x_dims = ctx->GetInputDim("X");
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(x_dims.size(), 4,
              platform::errors::NotFound(
                  "Input(X) of MirrorNormalizeOp should be a 4-D Tensor"));

      auto c = x_dims[1];
      auto mean = ctx->Attrs().Get<std::vector<float>>("mean");
      auto std = ctx->Attrs().Get<std::vector<float>>("std");
      PADDLE_ENFORCE_EQ(mean.size(), c,
              platform::errors::NotFound(
                  "The channel number of Input(X) should equal to length of mean"));
      PADDLE_ENFORCE_EQ(mean.size(), c,
              platform::errors::NotFound(
                  "The channel number of Input(X) should equal to length of mean"));
    }

    std::vector<int64_t> output_dims(x_dims.size());
    for (int i = 0; i < x_dims.size(); ++i) {
      output_dims[i] = x_dims[i];
    }
    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    ctx->ShareLoD("X", "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class MirrorNormalizeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of mirror_normalize op.");
    AddInput("Mirror", "(Tensor), The mirror vector for random flip, the "
                       "shape is {N, 1}, N is the batch size of input X");
    AddOutput("Out", "(Tensor), The output tensor in the same shape as "
                     "input X.");
    AddAttr<std::vector<float>>("mean", "The mean value to normalize data");
    AddAttr<std::vector<float>>("std", "The stdvalue to normalize data");
    AddComment(R"DOC(
          Mirror Normalize Operator.
      )DOC");
  }
};

class MirrorNormalizeOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
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
REGISTER_OPERATOR(mirror_normalize, ops::MirrorNormalizeOp, ops::MirrorNormalizeOpMaker, ops::MirrorNormalizeOpInferVarType);

REGISTER_OP_CPU_KERNEL(
    mirror_normalize, ops::MirrorNormalizeCPUKernel<float>,
    ops::MirrorNormalizeCPUKernel<double>);
