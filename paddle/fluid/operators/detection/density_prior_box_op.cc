/*Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection/density_prior_box_op.h"

namespace paddle {
namespace operators {

class DensityPriorBoxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of DensityPriorBoxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Image"),
                   "Input(Image) of DensityPriorBoxOp should not be null.");

    auto image_dims = ctx->GetInputDim("Image");
    auto input_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE(image_dims.size() == 4, "The layout of image is NCHW.");
    PADDLE_ENFORCE(input_dims.size() == 4, "The layout of input is NCHW.");

    PADDLE_ENFORCE_LT(input_dims[2], image_dims[2],
                      "The height of input must smaller than image.");

    PADDLE_ENFORCE_LT(input_dims[3], image_dims[3],
                      "The width of input must smaller than image.");
    auto variances = ctx->Attrs().Get<std::vector<float>>("variances");

    auto fixed_sizes = ctx->Attrs().Get<std::vector<float>>("fixed_sizes");
    auto fixed_ratios = ctx->Attrs().Get<std::vector<float>>("fixed_ratios");
    auto densities = ctx->Attrs().Get<std::vector<int>>("densities");
    bool flatten = ctx->Attrs().Get<bool>("flatten_to_2d");

    PADDLE_ENFORCE_EQ(fixed_sizes.size(), densities.size(),
                      "The number of fixed_sizes and densities must be equal.");
    size_t num_priors = 0;
    for (size_t i = 0; i < densities.size(); ++i) {
      num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
    }
    if (!flatten) {
      std::vector<int64_t> dim_vec(4);
      dim_vec[0] = input_dims[2];
      dim_vec[1] = input_dims[3];
      dim_vec[2] = num_priors;
      dim_vec[3] = 4;
      ctx->SetOutputDim("Boxes", framework::make_ddim(dim_vec));
      ctx->SetOutputDim("Variances", framework::make_ddim(dim_vec));
    } else {
      int64_t dim0 = input_dims[2] * input_dims[3] * num_priors;
      ctx->SetOutputDim("Boxes", {dim0, 4});
      ctx->SetOutputDim("Variances", {dim0, 4});
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::Tensor>("Input")->type(), ctx.GetPlace());
  }
};

class DensityPriorBoxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(Tensor, default Tensor<float>), "
        "the input feature data of DensityPriorBoxOp, the layout is NCHW.");
    AddInput("Image",
             "(Tensor, default Tensor<float>), "
             "the input image data of DensityPriorBoxOp, the layout is NCHW.");
    AddOutput("Boxes",
              "(Tensor, default Tensor<float>), the output prior boxes of "
              "DensityPriorBoxOp. The layout is [H, W, num_priors, 4]. "
              "H is the height of input, W is the width of input, num_priors "
              "is the box count of each position.");
    AddOutput("Variances",
              "(Tensor, default Tensor<float>), the expanded variances of "
              "DensityPriorBoxOp. The layout is [H, W, num_priors, 4]. "
              "H is the height of input, W is the width of input, num_priors "
              "is the box count of each position.");
    AddAttr<std::vector<float>>("variances",
                                "(vector<float>) List of variances to be "
                                "encoded in density prior boxes.")
        .AddCustomChecker([](const std::vector<float>& variances) {
          PADDLE_ENFORCE_EQ(variances.size(), 4,
                            "Must and only provide 4 variance.");
          for (size_t i = 0; i < variances.size(); ++i) {
            PADDLE_ENFORCE_GT(variances[i], 0.0,
                              "variance[%d] must be greater than 0.", i);
          }
        });
    AddAttr<bool>("clip", "(bool) Whether to clip out-of-boundary boxes.")
        .SetDefault(true);
    AddAttr<bool>("flatten_to_2d",
                  "(bool) Whether to flatten to 2D and "
                  "the second dim is 4.")
        .SetDefault(false);
    AddAttr<float>(
        "step_w",
        "Density prior boxes step across width, 0.0 for auto calculation.")
        .SetDefault(0.0)
        .AddCustomChecker([](const float& step_w) {
          PADDLE_ENFORCE_GE(step_w, 0.0, "step_w should be larger than 0.");
        });
    AddAttr<float>(
        "step_h",
        "Density prior boxes step across height, 0.0 for auto calculation.")
        .SetDefault(0.0)
        .AddCustomChecker([](const float& step_h) {
          PADDLE_ENFORCE_GE(step_h, 0.0, "step_h should be larger than 0.");
        });

    AddAttr<float>("offset",
                   "(float) "
                   "Density prior boxes center offset.")
        .SetDefault(0.5);
    AddAttr<std::vector<float>>("fixed_sizes",
                                "(vector<float>) List of fixed sizes "
                                "of generated density prior boxes.")
        .SetDefault(std::vector<float>{})
        .AddCustomChecker([](const std::vector<float>& fixed_sizes) {
          for (size_t i = 0; i < fixed_sizes.size(); ++i) {
            PADDLE_ENFORCE_GT(fixed_sizes[i], 0.0,
                              "fixed_sizes[%d] should be larger than 0.", i);
          }
        });

    AddAttr<std::vector<float>>("fixed_ratios",
                                "(vector<float>) List of fixed ratios "
                                "of generated density prior boxes.")
        .SetDefault(std::vector<float>{})
        .AddCustomChecker([](const std::vector<float>& fixed_ratios) {
          for (size_t i = 0; i < fixed_ratios.size(); ++i) {
            PADDLE_ENFORCE_GT(fixed_ratios[i], 0.0,
                              "fixed_ratios[%d] should be larger than 0.", i);
          }
        });

    AddAttr<std::vector<int>>("densities",
                              "(vector<float>) List of densities "
                              "of generated density prior boxes.")
        .SetDefault(std::vector<int>{})
        .AddCustomChecker([](const std::vector<int>& densities) {
          for (size_t i = 0; i < densities.size(); ++i) {
            PADDLE_ENFORCE_GT(densities[i], 0,
                              "densities[%d] should be larger than 0.", i);
          }
        });
    AddComment(R"DOC(
        Density Prior box operator
        Each position of the input produce N density prior boxes, N is determined by
        the count of fixed_ratios, densities, the calculation of N is as follows:
        for density in densities:
        N += size(fixed_ratios)*density^2
        )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(density_prior_box, ops::DensityPriorBoxOp,
                  ops::DensityPriorBoxOpMaker,
                  paddle::framework::EmptyGradOpMaker);

REGISTER_OP_CPU_KERNEL(density_prior_box, ops::DensityPriorBoxOpKernel<float>,
                       ops::DensityPriorBoxOpKernel<double>);
