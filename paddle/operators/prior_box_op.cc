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

#include "paddle/operators/prior_box_op.h"

namespace paddle {
namespace operators {

class PriorBoxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(X) of PriorBoxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Image"),
                   "Input(Offset) of PriorBoxOp should not be null.");

    auto image_dims = ctx->GetInputDim("Image");
    auto input_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE(image_dims.size() == 4,
                   "The format of input tensor is NCHW.");
    PADDLE_ENFORCE(input_dims.size() == 4,
                   "The format of input tensor is NCHW.");

    PADDLE_ENFORCE_LT(input_dims[2], image_dims[2],
                      "The height of input must smaller than image.");

    PADDLE_ENFORCE_LT(input_dims[3], image_dims[3],
                      "The width of input must smaller than image.");

    auto min_sizes = ctx->Attrs().Get<std::vector<int>>("min_sizes");
    auto max_sizes = ctx->Attrs().Get<std::vector<int>>("max_sizes");
    auto variances = ctx->Attrs().Get<std::vector<float>>("variances");
    auto input_aspect_ratio =
        ctx->Attrs().Get<std::vector<float>>("aspect_ratios");
    bool flip = ctx->Attrs().Get<bool>("flip");

    PADDLE_ENFORCE_GT(min_sizes.size(), 0,
                      "Size of min_size must be at least 1.");
    for (size_t i = 0; i < min_sizes.size(); ++i) {
      PADDLE_ENFORCE_GT(min_sizes[i], 0, "min_sizes[%d] must be positive.", i);
    }

    std::vector<float> aspect_ratios;
    expand_aspect_ratios(input_aspect_ratio, flip, aspect_ratios);

    int num_priors = aspect_ratios.size() * min_sizes.size();
    if (max_sizes.size() > 0) {
      PADDLE_ENFORCE_EQ(max_sizes.size(), min_sizes.size(),
                        "The length of min_size and max_size must be equal.");
      for (size_t i = 0; i < min_sizes.size(); ++i) {
        PADDLE_ENFORCE_GT(max_sizes[i], min_sizes[i],
                          "max_size[%d] must be greater than min_size[%d].", i,
                          i);
        num_priors += 1;
      }
    }

    if (variances.size() > 1) {
      PADDLE_ENFORCE_EQ(variances.size(), 4,
                        "Must and only provide 4 variance.");
      for (size_t i = 0; i < variances.size(); ++i) {
        PADDLE_ENFORCE_GT(variances[i], 0.0,
                          "variance[%d] must be greater than 0.", i);
      }
    } else if (variances.size() == 1) {
      PADDLE_ENFORCE_GT(variances[0], 0.0,
                        "variance[0] must be greater than 0.");
    }

    const int img_h = ctx->Attrs().Get<int>("img_h");
    PADDLE_ENFORCE_GT(img_h, 0, "img_h should be larger than 0.");
    const int img_w = ctx->Attrs().Get<int>("img_w");
    PADDLE_ENFORCE_GT(img_w, 0, "img_w should be larger than 0.");

    const float step_h = ctx->Attrs().Get<float>("step_h");
    PADDLE_ENFORCE_GT(step_h, 0.0, "step_h should be larger than 0.");
    const float step_w = ctx->Attrs().Get<float>("step_w");
    PADDLE_ENFORCE_GT(step_w, 0.0, "step_w should be larger than 0.");

    const int layer_height = input_dims[2];
    const int layer_width = input_dims[3];

    std::vector<int64_t> dim_vec(5);
    dim_vec[0] = 2;
    dim_vec[1] = layer_height;
    dim_vec[2] = layer_width;
    dim_vec[3] = num_priors;
    dim_vec[4] = 4;
    auto output_dim = framework::make_ddim(dim_vec);
    ctx->SetOutputDim("Out", output_dim);
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("Image")->type()),
        ctx.device_context());
  }
};

class PriorBoxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  PriorBoxOpMaker(framework::OpProto* proto,
                  framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Input",
             "(Tensor, default Tensor<float>), "
             "the input feature data of PriorBoxOp, The format is NCHW.");
    AddInput("Image",
             "(Tensor, default Tensor<float>), "
             "the input image data of PriorBoxOp, The format is NCHW.");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), the output prior boxes of "
              "PriorBoxOp. The format is [2, layer_height, layer_width, "
              "num_priors, 4]");
    AddAttr<std::vector<int>>("min_sizes", "(vector<int>) ",
                              "List of min sizes of generated prior boxes.");
    AddAttr<std::vector<int>>("max_sizes", "(vector<int>) ",
                              "List of max sizes of generated prior boxes.")
        .SetDefault({});
    AddAttr<std::vector<float>>(
        "aspect_ratios", "(vector<float>) ",
        "List of aspect ratios of generated prior boxes.")
        .SetDefault({});
    AddAttr<std::vector<float>>(
        "variances", "(vector<float>) ",
        "List of variances to be encoded in prior boxes.")
        .SetDefault({0.1});
    AddAttr<bool>("flip", "(bool) ", "Whether to flip aspect ratios.")
        .SetDefault(true);
    AddAttr<bool>("clip", "(bool) ", "Whether to clip out-of-boundary boxes.")
        .SetDefault(true);
    AddAttr<int>("img_w", "").SetDefault(0);
    AddAttr<int>("img_h", "").SetDefault(0);
    AddAttr<float>("step_w",
                   "Prior boxes step across width, 0 for auto calculation.")
        .SetDefault(0.0);
    AddAttr<float>("step_h",
                   "Prior boxes step across height, 0 for auto calculation.")
        .SetDefault(0.0);
    AddAttr<float>("offset",
                   "(float) "
                   "Prior boxes center offset.")
        .SetDefault(0.5);
    AddComment(R"DOC(
Prior box operator
Generate prior boxes for SSD(Single Shot MultiBox Detector) algorithm.
Please get more information from the following papers:
https://arxiv.org/abs/1512.02325.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(prior_box, ops::PriorBoxOp, ops::PriorBoxOpMaker);
REGISTER_OP_CPU_KERNEL(
    prior_box, ops::PriorBoxOpKernel<paddle::platform::CPUPlace, float>,
    ops::PriorBoxOpKernel<paddle::platform::CPUPlace, double>);
