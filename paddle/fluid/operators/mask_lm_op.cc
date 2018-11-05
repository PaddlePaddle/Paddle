/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mask_lm_op.h"

namespace paddle {
namespace operators {

class MaskLMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(x_dims.size() == 2 && x_dims[1] == 1,
                   "Input(X) of mask_lm should be lod_tensor shape [N, 1].");

    ctx->SetOutputDim("Out", x_dims);
    ctx->SetOutputDim("Mask", x_dims);
    ctx->SetOutputDim("MaskPos", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("X"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class MaskLMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tokens in index.");
    AddOutput("Out", "The masked output tokens in index.");
    AddOutput("Mask", "The generated mask tokens in index.").AsIntermediate();
    AddOutput("MaskPos", "The abosulte positions of generated mask.")
        .AsIntermediate();
    AddAttr<int>("mask_id", "MASK ID")
        .SetDefault(0)
        .AddCustomChecker([](const int& maskid) {
          PADDLE_ENFORCE(maskid >= 0, "'MASK ID' must not be negative.");
        });
    AddAttr<int>("voc_size", "The size of vocabulary.")
        .SetDefault(0)
        .AddCustomChecker([](const int& voc_s) {
          PADDLE_ENFORCE(voc_s >= 0, "'voc_size' must not be negative.");
        });
    AddAttr<float>("masked_prob",
                   "Probability of setting each token to MASK ID.")
        .SetDefault(.1f)
        .AddCustomChecker([](const float& masked_p) {
          PADDLE_ENFORCE(masked_p >= 0.0f && masked_p <= 1.0f,
                         "'masked_prob' must be between 0.0 and 1.0.");
        });
    AddAttr<bool>("fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug that always the same output units "
                  "will be dropped.")
        .SetDefault(false);
    AddAttr<int>("seed", "seed for random number generator.").SetDefault(0);

    AddComment(R"DOC(
Mask LM Operator.

to do.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mask_lm, ops::MaskLMOp, ops::MaskLMOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    mask_lm, ops::CPUMaskLMKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUMaskLMKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::float16>,
    ops::CPUMaskLMKernel<paddle::platform::CPUDeviceContext, double>,
    ops::CPUMaskLMKernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::CPUMaskLMKernel<paddle::platform::CPUDeviceContext, int>,
    ops::CPUMaskLMKernel<paddle::platform::CPUDeviceContext, int64_t>);
