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

    auto min_sizes = ctx->Attrs().Get<std::vector<float>>("min_sizes");
    auto max_sizes = ctx->Attrs().Get<std::vector<float>>("max_sizes");
    auto variances = ctx->Attrs().Get<std::vector<float>>("variances");
    auto aspect_ratios = ctx->Attrs().Get<std::vector<float>>("aspect_ratios");
    bool flip = ctx->Attrs().Get<bool>("flip");
    auto fixed_sizes = ctx->Attrs().Get<std::vector<float>>("fixed_sizes");
    auto fixed_ratios = ctx->Attrs().Get<std::vector<float>>("fixed_ratios");
    auto densities = ctx->Attrs().Get<std::vector<int>>("densities");
    std::vector<float> aspect_ratios_vec;
    ExpandAspectRatios(aspect_ratios, flip, &aspect_ratios_vec);

    size_t num_priors = aspect_ratios_vec.size() * min_sizes.size();
    PADDLE_ENFORCE_EQ(fixed_sizes.size(), densities.size(),
                      "The number of fixed_sizes and densities must be equal.");

    if (fixed_sizes.size() > 0) {
      if (densities.size() > 0) {
        for (size_t i = 0; i < densities.size(); ++i) {
          if (fixed_ratios.size() > 0) {
            num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
          } else {
            num_priors += (aspect_ratios_vec.size()) * (pow(densities[i], 2));
          }
        }
      }
    }
    if (max_sizes.size() > 0) {
      PADDLE_ENFORCE_EQ(max_sizes.size(), min_sizes.size(),
                        "The number of min_size and max_size must be equal.");
      num_priors += max_sizes.size();
      for (size_t i = 0; i < max_sizes.size(); ++i) {
        PADDLE_ENFORCE_GT(max_sizes[i], min_sizes[i],
                          "max_size[%d] must be greater than min_size[%d].", i,
                          i);
      }
    }

    std::vector<int64_t> dim_vec(4);
    dim_vec[0] = input_dims[2];
    dim_vec[1] = input_dims[3];
    dim_vec[2] = num_priors;
    dim_vec[3] = 4;
    ctx->SetOutputDim("Boxes", framework::make_ddim(dim_vec));
    ctx->SetOutputDim("Variances", framework::make_ddim(dim_vec));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("Input")->type()),
        platform::CPUPlace());
  }
};

class DensityPriorBoxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(Tensor, default Tensor<float>), "
        "the input feature data of DensityPriorBoxOp, The layout is NCHW.");
    AddInput("Image",
             "(Tensor, default Tensor<float>), "
             "the input image data of DensityPriorBoxOp, The layout is NCHW.");
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

    AddAttr<std::vector<float>>("min_sizes",
                                "(vector<float>) List of min sizes "
                                "of generated density prior boxes.")
        .AddCustomChecker([](const std::vector<float>& min_sizes) {
          PADDLE_ENFORCE_GT(min_sizes.size(), 0,
                            "Size of min_sizes must be at least 1.");
          for (size_t i = 0; i < min_sizes.size(); ++i) {
            PADDLE_ENFORCE_GT(min_sizes[i], 0.0,
                              "min_sizes[%d] must be positive.", i);
          }
        });
    AddAttr<std::vector<float>>(
        "max_sizes",
        "(vector<float>) List of max sizes of generated density prior boxes.")
        .SetDefault(std::vector<float>{});
    AddAttr<std::vector<float>>("aspect_ratios",
                                "(vector<float>) List of aspect ratios of "
                                "generated density prior boxes.");

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
    AddAttr<bool>("flip", "(bool) Whether to flip aspect ratios.")
        .SetDefault(true);
    AddAttr<bool>("clip", "(bool) Whether to clip out-of-boundary boxes.")
        .SetDefault(true);

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

    AddAttr<bool>(
        "min_max_aspect_ratios_order",
        "(bool) If set True, the output prior box is in order of"
        "[min, max, aspect_ratios], which is consistent with Caffe."
        "Please note, this order affects the weights order of convolution layer"
        "followed by and does not affect the final detection results.")
        .SetDefault(false);
    AddComment(R"DOC(
Density Prior box operator
Generate density prior boxes for SSD(Single Shot MultiBox Detector) algorithm.
Each position of the input produce N prior boxes, N is determined by
 the count of min_sizes, max_sizes and aspect_ratios, The size of the
 box is in range(min_size, max_size) interval, which is generated in
 sequence according to the aspect_ratios.
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
