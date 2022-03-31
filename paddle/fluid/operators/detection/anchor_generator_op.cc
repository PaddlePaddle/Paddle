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

#include "paddle/fluid/operators/detection/anchor_generator_op.h"

namespace paddle {
namespace operators {

class AnchorGeneratorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::InvalidArgument(
            "Input(Input) of AnchorGeneratorOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Anchors"), true,
        platform::errors::InvalidArgument(
            "Output(Anchors) of AnchorGeneratorOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Variances"), true,
        platform::errors::InvalidArgument(
            "Output(Variances) of AnchorGeneratorOp should not be null."));

    auto input_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_EQ(
        input_dims.size(), 4,
        platform::errors::InvalidArgument("The layout of input is NCHW."));

    auto anchor_sizes = ctx->Attrs().Get<std::vector<float>>("anchor_sizes");
    auto aspect_ratios = ctx->Attrs().Get<std::vector<float>>("aspect_ratios");
    auto stride = ctx->Attrs().Get<std::vector<float>>("stride");
    auto variances = ctx->Attrs().Get<std::vector<float>>("variances");

    size_t num_anchors = aspect_ratios.size() * anchor_sizes.size();

    std::vector<int64_t> dim_vec(4);
    dim_vec[0] = input_dims[2];
    dim_vec[1] = input_dims[3];
    dim_vec[2] = num_anchors;
    dim_vec[3] = 4;
    ctx->SetOutputDim("Anchors", phi::make_ddim(dim_vec));
    ctx->SetOutputDim("Variances", phi::make_ddim(dim_vec));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class AnchorGeneratorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor, default Tensor<float>), "
             "the input feature is a tensor with a rank of 4. "
             "The layout is NCHW.");
    AddOutput("Anchors",
              "(Tensor, default Tensor<float>), the output is a "
              "tensor with a rank of 4. The layout is [H, W, num_anchors, 4]. "
              "H is the height of input, W is the width of input, num_anchors "
              "is the box count of each position. "
              "Each anchor is in (xmin, ymin, xmax, ymax) format");
    AddOutput("Variances",
              "(Tensor, default Tensor<float>), the expanded variances for "
              "normalizing bbox regression targets. The layout is [H, W, "
              "num_anchors, 4]. "
              "H is the height of input, W is the width of input, num_anchors "
              "is the box count of each position. "
              "Each variance is in (xcenter, ycenter, w, h) format");

    AddAttr<std::vector<float>>(
        "anchor_sizes",
        "(vector<float>) List of Region Proposal Network(RPN) anchor sizes "
        " given in absolute pixels e.g. (64, 128, 256, 512)."
        " For instance, the anchor size of 64 means the area of this anchor "
        "equals to 64**2.")
        .AddCustomChecker([](const std::vector<float>& anchor_sizes) {
          PADDLE_ENFORCE_GT(anchor_sizes.size(), 0UL,
                            platform::errors::InvalidArgument(
                                "Size of anchor_sizes must be at least 1."));
          for (size_t i = 0; i < anchor_sizes.size(); ++i) {
            PADDLE_ENFORCE_GT(anchor_sizes[i], 0.0,
                              platform::errors::InvalidArgument(
                                  "anchor_sizes[%d] must be positive.", i));
          }
        });
    AddAttr<std::vector<float>>(
        "aspect_ratios",
        "(vector<float>) List of Region Proposal Network(RPN) anchor aspect "
        "ratios, e.g. (0.5, 1, 2)."
        "For instacne, the aspect ratio of 0.5 means the height / width of "
        "this anchor equals 0.5.");

    AddAttr<std::vector<float>>("variances",
                                "(vector<float>) List of variances to be used "
                                "in box regression deltas")
        .AddCustomChecker([](const std::vector<float>& variances) {
          PADDLE_ENFORCE_EQ(variances.size(), 4UL,
                            platform::errors::InvalidArgument(
                                "Must provide 4 variance only."));
          for (size_t i = 0; i < variances.size(); ++i) {
            PADDLE_ENFORCE_GT(variances[i], 0.0,
                              platform::errors::InvalidArgument(
                                  "variance[%d] must be greater than 0.", i));
          }
        });

    AddAttr<std::vector<float>>("stride",
                                "Anchors stride across width and height, "
                                "with a default of (16, 16)")
        .SetDefault(std::vector<float>(2, 16.0))
        .AddCustomChecker([](const std::vector<float>& stride) {
          PADDLE_ENFORCE_EQ(
              stride.size(), 2UL,
              platform::errors::InvalidArgument(
                  "Must provide 2 stride for width and height only."));
          for (size_t i = 0; i < stride.size(); ++i) {
            PADDLE_ENFORCE_GT(stride[i], 0.0,
                              platform::errors::InvalidArgument(
                                  "stride[%d] should be larger than 0.", i));
          }
        });

    AddAttr<float>("offset",
                   "(float) "
                   "Anchor center offset, with a default of 0.5")
        .SetDefault(0.5);
    AddComment(R"DOC(
AnchorGenerator operator
Generates anchors for Faster RCNN, FPN etc. algorithm.
Each position of the input produce N anchors, N =
 size(anchor_sizes) * size(aspect_ratios).

Please get more information from the following papers:
https://arxiv.org/abs/1506.01497.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    anchor_generator, ops::AnchorGeneratorOp, ops::AnchorGeneratorOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(anchor_generator, ops::AnchorGeneratorOpKernel<float>,
                       ops::AnchorGeneratorOpKernel<double>);
