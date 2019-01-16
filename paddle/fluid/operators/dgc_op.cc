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

#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/dgc_op.h"

namespace paddle {
namespace operators {

class DGCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("U"), "Input(U) of DGCop should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("V"), "Input(V) of DGCop should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of DGCop should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("GradLocal"),
                   "Input(GradLocal) of DGCop should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("U"),
                   "Output(U) of DGCop should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("V"),
                   "Output(V) of DGCop should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("GradLocal"),
                   "Output(GradLocal) of DGCop should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("EncodeGrad"),
                   "Output(EncodeGrad) of DGCop should not be null.");

    float ratio = ctx->Attrs().Get<float>("ratio");
    PADDLE_ENFORCE(ratio > 0.0001 && ratio < 1.0,
                   "ratio of dgc must in range [0.0001, 1.0]");

    /*
    auto dim = ctx->GetInputDim("U");
    ctx->SetOutputDim("U", dim);

    dim = ctx->GetInputDim("V");
    ctx->SetOutputDim("V", dim);
    */

    auto dim = ctx->GetInputDim("Grad");
    ctx->SetOutputDim("EncodeGrad", dim);

    /*
    auto param_dim = ctx->GetInputDim("GradLocal");
    ctx->SetOutputDim("GradLocal", param_dim);
    */
  }
};

/*
class DGCOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    return std::unordered_map<std::string, std::string>{
        {"Grad", "EncodeGrad"}};
  }
};
*/

class DGCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("U", "(Tensor) Middle tensor of DGC");
    AddInput("V", "(Tensor) Middle tensor of DGC");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("GradLocal", "(Tensor) Local gradient for accumulation.");
    AddOutput("EncodeGrad",
              "(Tensor) "
              "Output encoded gradient");
    AddAttr<float>("m",
                   "(float) "
                   "The momentum of learning rate.");
    AddAttr<float>("ratio",
                   "(float, default 1000) "
                   "Reserve topk from tensor.")
        .SetDefault(0.001);
    AddComment(R"DOC(
    Please see appendix D of https://arxiv.org/abs/1712.01887.pdf
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(dgc, ops::DGCOp, ops::DGCOpMaker);
