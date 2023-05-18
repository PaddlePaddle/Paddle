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

#include "paddle/fluid/operators/batch_norm_op.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/data_layout.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

template <typename T>
void BatchNormGradMaker<T>::Apply(GradOpPtr<T> op) const {
  op->SetType(this->ForwardOpType() + "_grad");
  op->SetInput("X", this->Input("X"));
  op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

  op->SetInput("Scale", this->Input("Scale"));
  op->SetInput("Bias", this->Input("Bias"));
  op->SetInput("SavedMean", this->Output("SavedMean"));
  op->SetInput("SavedVariance", this->Output("SavedVariance"));
  if (this->HasOutput("ReserveSpace")) {
    op->SetInput("ReserveSpace", this->Output("ReserveSpace"));
  }

  // used when setting use_global_stats True during training
  if (PADDLE_GET_CONST(bool, this->GetAttr("use_global_stats")) ||
      PADDLE_GET_CONST(bool, this->GetAttr("is_test"))) {
    op->SetInput("Mean", this->Output("MeanOut"));
    op->SetInput("Variance", this->Output("VarianceOut"));
  }

  op->SetAttrMap(this->Attrs());

  op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));
  op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
}

template <typename T>
void BatchNormDoubleGradMaker<T>::Apply(GradOpPtr<T> op) const {
  op->SetType("batch_norm_grad_grad");
  op->SetInput("X", this->Input("X"));
  op->SetInput("Scale", this->Input("Scale"));
  op->SetInput("SavedMean", this->Input("SavedMean"));
  op->SetInput("SavedVariance", this->Input("SavedVariance"));
  if (PADDLE_GET_CONST(bool, this->GetAttr("use_global_stats"))) {
    op->SetInput("Mean", this->Input("Mean"));
    op->SetInput("Variance", this->Input("Variance"));
  }
  op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
  op->SetInput("DDScale", this->OutputGrad(framework::GradVarName("Scale")));
  op->SetInput("DDBias", this->OutputGrad(framework::GradVarName("Bias")));
  op->SetInput("DY", this->Input(framework::GradVarName("Y")));

  op->SetAttrMap(this->Attrs());
  op->SetOutput("DX", this->InputGrad("X"));
  op->SetOutput("DScale", this->InputGrad("Scale"));
  op->SetOutput("DDY", this->InputGrad(framework::GradVarName("Y")));
}

void BatchNormDoubleGradOp::InferShape(
    framework::InferShapeContext *ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BatchNormDoubleGrad");
  OP_INOUT_CHECK(
      ctx->HasInput("Scale"), "Input", "Scale", "BatchNormDoubleGrad");
  OP_INOUT_CHECK(
      ctx->HasInput("SavedMean"), "Input", "SavedMean", "BatchNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedVariance"),
                 "Input",
                 "SavedVariance",
                 "BatchNormDoubleGrad");

  const bool use_global_stats = ctx->Attrs().Get<bool>("use_global_stats");
  if (use_global_stats) {
    OP_INOUT_CHECK(ctx->HasInput("Variance"),
                   "Input",
                   "VarianceOut",
                   "BatchNormDoubleGrad");
  }

  OP_INOUT_CHECK(ctx->HasInput("DY"), "Input", "DY", "BatchNormDoubleGrad");

  // check output
  OP_INOUT_CHECK(ctx->HasOutput("DX"), "Output", "DX", "BatchNormDoubleGrad");

  const auto x_dims = ctx->GetInputDim("X");
  const DataLayout data_layout =
      phi::StringToDataLayout(ctx->Attrs().Get<std::string>("data_layout"));
  const int C =
      ((ctx->IsRunMKLDNNKernel() == true) || (data_layout == DataLayout::kNCHW)
           ? x_dims[1]
           : x_dims[x_dims.size() - 1]);

  if (ctx->HasOutput("DX")) {
    ctx->SetOutputDim("DX", x_dims);
  }
  if (ctx->HasOutput("DScale")) {
    ctx->SetOutputDim("DScale", {C});
  }
  if (ctx->HasOutput("DDY")) {
    ctx->ShareDim("X", "DDY");
  }
}

DECLARE_INPLACE_OP_INFERER(BatchNormDoubleGradOpInplaceInferer, {"DY", "DDY"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(batch_norm_grad_grad,
                  ops::BatchNormDoubleGradOp,
                  ops::BatchNormDoubleGradOpInplaceInferer);
