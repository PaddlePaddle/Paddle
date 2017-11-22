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

#include "paddle/operators/roi_pool_op.h"

namespace paddle {
namespace operators {

class RoiPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of RoiPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Rois"),
                   "Input(Rois) of RoiPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of RoiPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Argmax"),
                   "Output(Argmax) of RoiPoolOp should not be null.");
    auto input_dims = ctx->GetInputDim("X");

    // Initialize the output's dims to maximum,
    // and re-set to real dims by the value of Rois at kernel
    ctx->SetOutputDim("Out", input_dims);
    }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("X")->type()),
        ctx.device_context());
  }
};

class RoiPoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "The gradient of Out should not be null.");
    PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName("X")),
                   "The gradient of X should not be null.");
    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("X")->type()),
        ctx.device_context());
  }
};

class RoiPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RoiPoolOpMaker(framework::OpProto* proto,
                       framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor), "
             "the input of RoiPoolOp.");
    AddInput("Rois",
             "(Tensor), "
             "RoIs (Regions of Interest) to pool over. "
             "Should be a 2-D tensor of shape (num_rois, 5)"
             "given as [[batch_id, x1, y1, x2, y2], …].");
    AddOutput("Out",
              "(Tensor), "
             "RoI pooled output 4-D tensor of shape "
             "(num_rois, channels, pooled_h, pooled_w).");
    AddOutput("Argmax",
              "(Tensor), "
              "Argmaxes corresponding to indices in X used "
              "for gradient computation. Only output "
              "if arg “is_test” is false.").AsIntermediate();
    AddAttr<float>("spatial_scale",
                      "(float, default 1.0), "
                      "Multiplicative spatial scale factor "
                      "to translate ROI coords from their input scale "
                      "to the scale used when pooling.")
                      .SetDefault(1.0);
    AddAttr<int>("pooled_height",
                      "(int, default 1), "
                      "The pooled output height.")
                    .SetDefault(1);
    AddAttr<int>("pooled_width",
                      "(int, default 1), "
                      "The pooled output width.")
                    .SetDefault(1);
    AddComment(R"DOC(
RoiPool operator

ROI Pooling for Faster-RCNN. The link below is a further introduction: 
https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(roi_pool, ops::RoiPoolOp, ops::RoiPoolOpMaker,
            roi_pool_grad, ops::RoiPoolGradOp);
REGISTER_OP_CPU_KERNEL(
    roi_pool,
    ops::CPURoiPoolOpKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    roi_pool_grad,
    ops::CPURoiPoolGradOpKernel<paddle::platform::CPUPlace, float>);
