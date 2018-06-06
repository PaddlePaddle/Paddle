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
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of AnchorGeneratorOp should not be null.");

    auto input_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE(input_dims.size() == 4, "The layout of input is NCHW.");

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
    ctx->SetOutputDim("Anchors", framework::make_ddim(dim_vec));
    ctx->SetOutputDim("Variances", framework::make_ddim(dim_vec));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("Input")->type()),
        ctx.device_context());
  }
};

class AnchorGeneratorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor, default Tensor<float>), "
             "the input feature data of AnchorGeneratorOp. "
             "The layout is NCHW.");
    AddOutput("Anchors",
              "(Tensor, default Tensor<float>), the output anchors of "
              "AnchorGeneratorOp. The layout is [H, W, num_anchors, 4]. "
              "H is the height of input, W is the width of input, num_anchors "
              "is the box count of each position. "
              "Each anchor is in (x1, y1, x2, y2) format");
    AddOutput("Variances",
              "(Tensor, default Tensor<float>), the expanded variances of "
              "AnchorGeneratorOp. The layout is [H, W, num_anchors, 4]. "
              "H is the height of input, W is the width of input, num_anchors "
              "is the box count of each position. "
              "Each variance is in (x, y, w, h) format "
              "which is unrelated with Anchor's");

    AddAttr<std::vector<float>>("anchor_sizes",
                                "(vector<float>) List of anchor sizes "
                                "of generated anchors.")
        .AddCustomChecker([](const std::vector<float>& anchor_sizes) {
          PADDLE_ENFORCE_GT(anchor_sizes.size(), 0,
                            "Size of anchor_sizes must be at least 1.");
          for (size_t i = 0; i < anchor_sizes.size(); ++i) {
            PADDLE_ENFORCE_GT(anchor_sizes[i], 0.0,
                              "anchor_sizes[%d] must be positive.", i);
          }
        });
    AddAttr<std::vector<float>>(
        "aspect_ratios",
        "(vector<float>) List of aspect ratios of generated anchors.");

    AddAttr<std::vector<float>>("variances",
                                "(vector<float>) List of variances to be used "
                                "in box regression deltas.")
        .AddCustomChecker([](const std::vector<float>& variances) {
          PADDLE_ENFORCE_EQ(variances.size(), 4,
                            "Must and only provide 4 variance.");
          for (size_t i = 0; i < variances.size(); ++i) {
            PADDLE_ENFORCE_GT(variances[i], 0.0,
                              "variance[%d] must be greater than 0.", i);
          }
        });

    AddAttr<std::vector<float>>("stride",
                                "Anchors stride across width and height.")
        .SetDefault(std::vector<float>(2, 16.0))
        .AddCustomChecker([](const std::vector<float>& stride) {
          PADDLE_ENFORCE_EQ(
              stride.size(), 2,
              "Must and only provide 2 stride for width and height.");
          for (size_t i = 0; i < stride.size(); ++i) {
            PADDLE_ENFORCE_GT(stride[i], 0.0,
                              "stride[%d] should be larger than 0.", i);
          }
        });

    AddAttr<float>("offset",
                   "(float) "
                   "Anchor center offset.")
        .SetDefault(0.5);
    AddComment(R"DOC(
AnchorGenerator operator
Generates anchors for Faster RCNN, FPN etc. algorithm.
Each position of the input produce N anchors, N is determined by
 the count of anchor_sizes, aspect_ratios.

Please get more information from the following papers:
https://arxiv.org/abs/1506.01497.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(anchor_generator, ops::AnchorGeneratorOp,
                  ops::AnchorGeneratorOpMaker,
                  paddle::framework::EmptyGradOpMaker);

REGISTER_OP_CPU_KERNEL(anchor_generator, ops::AnchorGeneratorOpKernel<float>,
                       ops::AnchorGeneratorOpKernel<double>);
