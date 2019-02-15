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
    // PADDLE_ENFORCE(ctx->HasInput("ratio"),
    //              "Input(ratio) of DGCop should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("current_step"),
                   "Input(current_step) of DGCop should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("rampup_step"),
                   "Input(rampup_step) of DGCop should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("U_out"),
                   "Output(U_out) of DGCop should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("V_out"),
                   "Output(V_out) of DGCop should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("EncodeGrad"),
                   "Output(EncodeGrad) of DGCop should not be null.");

    /*
    std::vector<framework::InferShapeVarPtr> ratio_var_ptrs =
    ctx->GetInputVarPtrs("ratio");
    PADDLE_ENFORCE(ratio_var_ptrs.size() == 1);
    auto ratio_var = boost::get<framework::Variable *>(ratio_var_ptrs[0])
    auto ratio_tensor = ratio_var->Get<framework::Tensor>();
    float ratio = 1 - *ratio_tensor->data<T>();
    PADDLE_ENFORCE(ratio > 0.0009 && ratio < 1.0,
                   "ratio of dgc must in range [0.0009, 1.0]");
    VLOG(10) << "reserved ratio:" << ratio;
    */

    /*
    auto dim = ctx->GetInputDim("Grad");
    int k = static_cast<int>(framework::product(dim) * ratio);

    FIXME(gongwb): need not set output dims?
    ctx->SetOutputDim("EncodeGrad", framework::DDim{2 * k});
    VLOG(10) << "set EncodeGrad dims:" << framework::DDim{2 * k};
    ctx->ShareLoD("Grad", "EncodeGrad");
    */

    // int buf_size = get_buffer_size(k);
    PADDLE_ENFORCE(ctx->HasOutput("Encoded_buf"),
                   "Output(Encoded_buf) of DGCop should not be null.");
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
    AddInput("current_step", "(Tensor) Current step.");
    AddInput("rampup_step", "(Tensor) Ramping up step.");

    AddOutput("U_out",
              "(Tensor) "
              "Output encoded gradient");
    AddOutput("V_out",
              "(Tensor) "
              "Output encoded gradient");
    AddOutput("EncodeGrad",
              "(Tensor) "
              "Output encoded gradient");

    AddAttr<float>("m",
                   "(float, 0.9) "
                   "The momentum of learning rate.")
        .SetDefault(0.9);

    AddComment(R"DOC(
    Please see appendix D of https://arxiv.org/abs/1712.01887.pdf
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
// REGISTER_OPERATOR(dgc, ops::DGCOp,
// paddle::framework::EmptyGradOpMaker, ops::DGCOpMaker)
REGISTER_OP_WITHOUT_GRADIENT(dgc, ops::DGCOp, ops::DGCOpMaker);
