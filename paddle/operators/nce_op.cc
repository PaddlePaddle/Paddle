/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/nce_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class NCEOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"));
    PADDLE_ENFORCE(ctx->HasInput("Label"));
    PADDLE_ENFORCE(ctx->HasInput("W"));
    PADDLE_ENFORCE(ctx->HasOutput("Out"));
    PADDLE_ENFORCE(ctx->HasOutput("SampleLogits"));
    PADDLE_ENFORCE(ctx->HasOutput("SampleLabels"));

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE_EQ(x_dims[0], label_dims[0]);
    if (ctx->HasInput("B")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("W")[0], ctx->GetInputDim("B")[0]);
    }
    int num_sampled_classes = ctx->Attrs().Get<int>("num_sampled_classes");
    int num_classes = ctx->Attrs().Get<int>("num_classes");
    PADDLE_ENFORCE_EQ(num_classes, ctx->GetInputDim("W")[0]);
    PADDLE_ENFORCE_LT(num_sampled_classes, num_classes);

    // set dims of output(Out)
    std::vector<int64_t> out_dims(1);
    out_dims.push_back(x_dims[0]);
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));

    // set dims of output(SampleOut)
    std::vector<int64_t> sample_out_dims(2);
    sample_out_dims.push_back(x_dims[0]);
    sample_out_dims.push_back(num_sampled_classes + 1);
    ctx->SetOutputDim("SampleLogits", framework::make_ddim(sample_out_dims));
    ctx->SetOutputDim("SampleLabels", framework::make_ddim(sample_out_dims));
  }
};

class NCEOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  NCEOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddInput("Label", "");
    AddInput("W", "");
    AddInput("B", "");
    AddInput("SampleWeight", "");
    AddOutput("Out", "");
    AddOutput("SampleLogits", "");
    AddOutput("SampleLabels", "");
    AddAttr<int>("num_classes", "");
    AddAttr<int>("num_sampled_classes", "").SetDefault(10);
    AddComment(R"DOC(
Expand input(X) according to LOD of input(Y).

)DOC");
  }
};

class NCEOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"));
    PADDLE_ENFORCE(ctx->HasInput("W"));
    PADDLE_ENFORCE(ctx->HasInput("Out"));
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "The input(Out@GRAD) should not be null");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }

    auto w_dims = ctx->GetInputDim("W");
    auto w_grad_name = framework::GradVarName("W");
    if (ctx->HasOutput(w_grad_name)) {
      ctx->SetOutputDim(w_grad_name, w_dims);
    }

    auto bias_grad_name = framework::GradVarName("B");
    if (ctx->HasOutput(bias_grad_name)) {
      auto bias_dims = ctx->GetInputDim("B");
      ctx->SetOutputDim(bias_grad_name, bias_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(nce, ops::NCEOp, ops::NCEOpMaker, nce_grad, ops::NCEOpGrad);
REGISTER_OP_CPU_KERNEL(nce, ops::NCEKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(nce_grad,
                       ops::NCEGradKernel<paddle::platform::CPUPlace, float>);
