/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection/box_clip_op.h"

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class BoxClipOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"),
                      true,
                      platform::errors::NotFound("Input(Input) of BoxClipOp "
                                                 "is not found."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("ImInfo"),
                      true,
                      platform::errors::NotFound("Input(ImInfo) of BoxClipOp "
                                                 "is not found."));

    auto input_box_dims = ctx->GetInputDim("Input");
    auto im_info_dims = ctx->GetInputDim("ImInfo");

    if (ctx->IsRuntime()) {
      auto input_box_size = input_box_dims.size();
      PADDLE_ENFORCE_EQ(
          input_box_dims[input_box_size - 1],
          4,
          platform::errors::InvalidArgument(
              "The last dimension of Input(Input) in BoxClipOp must be 4. "
              "But received last dimension = %d",
              input_box_dims[input_box_size - 1]));
      PADDLE_ENFORCE_EQ(im_info_dims.size(),
                        2,
                        platform::errors::InvalidArgument(
                            "The rank of Input(Input) in BoxClipOp must be 2."
                            " But received rank = %d",
                            im_info_dims.size()));
      PADDLE_ENFORCE_EQ(
          im_info_dims[1],
          3,
          platform::errors::InvalidArgument(
              "The last dimension of Input(ImInfo) of BoxClipOp must be 3. "
              "But received last dimension = %d",
              im_info_dims[1]));
    }
    ctx->ShareDim("Input", /*->*/ "Output");
    ctx->ShareLoD("Input", /*->*/ "Output");
  }
};

class BoxClipOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(LoDTensor) "
             "Input is a LoDTensor with shape [..., 4] holds 4 points"
             "in last dimension in format [xmin, ymin, xmax, ymax]");
    AddInput("ImInfo",
             "(Tensor) Information for image reshape is in shape (N, 3), "
             "in format (height, width, im_scale)");
    AddOutput("Output",
              "(LoDTensor) "
              "Output is a LoDTensor with the same shape as Input"
              "and it is the result after clip");
    AddComment(R"DOC(
This operator clips input boxes to original input images.

For each input box, The formula is given as follows:

       $$xmin = \max(\min(xmin, im_w - 1), 0)$$
       $$ymin = \max(\min(ymin, im_h - 1), 0)$$
       $$xmax = \max(\min(xmax, im_w - 1), 0)$$
       $$ymax = \max(\min(ymax, im_h - 1), 0)$$

where im_w and im_h are computed from ImInfo, the formula is given as follows:

       $$im_w = \round(width / im_scale)$$
       $$im_h = \round(height / im_scale)$$
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    box_clip,
    ops::BoxClipOp,
    ops::BoxClipOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(box_clip,
                       ops::BoxClipKernel<phi::CPUContext, float>,
                       ops::BoxClipKernel<phi::CPUContext, double>);
