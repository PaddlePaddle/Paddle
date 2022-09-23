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

#include "paddle/fluid/operators/detection/prior_box_op.h"
#include <string>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/infermeta/binary.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace operators {

class PriorBoxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_input_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Input");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_input_type)) {
      auto input_image_type = framework::TransToProtoVarType(
          ctx.Input<framework::Tensor>("Image")->dtype());
      int customized_type_value =
          framework::OpKernelType::kDefaultCustomizedTypeValue;
      if (input_image_type == framework::DataTypeTrait<float>::DataType()) {
        customized_type_value = kPriorBoxFLOAT;
      } else if (input_image_type ==
                 framework::DataTypeTrait<double>::DataType()) {
        customized_type_value = kPriorBoxDOUBLE;
      }
      return framework::OpKernelType(input_input_type,
                                     ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN,
                                     customized_type_value);
    }
#endif
    return framework::OpKernelType(input_input_type, ctx.GetPlace());
  }
};

class PriorBoxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor, default Tensor<float>), "
             "the input feature data of PriorBoxOp, The layout is NCHW.");
    AddInput("Image",
             "(Tensor, default Tensor<float>), "
             "the input image data of PriorBoxOp, The layout is NCHW.");
    AddOutput("Boxes",
              "(Tensor, default Tensor<float>), the output prior boxes of "
              "PriorBoxOp. The layout is [H, W, num_priors, 4]. "
              "H is the height of input, W is the width of input, num_priors "
              "is the box count of each position.");
    AddOutput("Variances",
              "(Tensor, default Tensor<float>), the expanded variances of "
              "PriorBoxOp. The layout is [H, W, num_priors, 4]. "
              "H is the height of input, W is the width of input, num_priors "
              "is the box count of each position.");

    AddAttr<std::vector<float>>("min_sizes",
                                "(vector<float>) List of min sizes "
                                "of generated prior boxes.")
        .AddCustomChecker([](const std::vector<float>& min_sizes) {
          PADDLE_ENFORCE_GT(
              min_sizes.size(),
              0,
              platform::errors::InvalidArgument("Size of min_sizes must be "
                                                "at least 1."));
          for (size_t i = 0; i < min_sizes.size(); ++i) {
            PADDLE_ENFORCE_GT(min_sizes[i],
                              0.0,
                              platform::errors::OutOfRange(
                                  "min_sizes[%d] must be larger "
                                  "than 0. But received: min_sizes[%d] is %f.",
                                  i,
                                  i,
                                  min_sizes[i]));
          }
        });
    AddAttr<std::vector<float>>(
        "max_sizes",
        "(vector<float>) List of max sizes of generated prior boxes.")
        .SetDefault(std::vector<float>{});
    AddAttr<std::vector<float>>(
        "aspect_ratios",
        "(vector<float>) List of aspect ratios of generated prior boxes.");

    AddAttr<std::vector<float>>(
        "variances",
        "(vector<float>) List of variances to be encoded in prior boxes.")
        .AddCustomChecker([](const std::vector<float>& variances) {
          PADDLE_ENFORCE_EQ(variances.size(),
                            4,
                            platform::errors::InvalidArgument(
                                "The length of variance must "
                                "be 4. But received: variances' length is %d.",
                                variances.size()));
          for (size_t i = 0; i < variances.size(); ++i) {
            PADDLE_ENFORCE_GT(variances[i],
                              0.0,
                              platform::errors::OutOfRange(
                                  "variance[%d] must be greater "
                                  "than 0. But received: variance[%d] = %f",
                                  i,
                                  i,
                                  variances[i]));
          }
        });
    AddAttr<bool>("flip", "(bool) Whether to flip aspect ratios.")
        .SetDefault(true);
    AddAttr<bool>("clip", "(bool) Whether to clip out-of-boundary boxes.")
        .SetDefault(true);

    AddAttr<float>("step_w",
                   "Prior boxes step across width, 0.0 for auto calculation.")
        .SetDefault(0.0)
        .AddCustomChecker([](const float& step_w) {
          PADDLE_ENFORCE_GE(step_w,
                            0.0,
                            platform::errors::InvalidArgument(
                                "step_w should be larger "
                                "than 0. But received: step_w = %f.",
                                step_w));
        });
    AddAttr<float>("step_h",
                   "Prior boxes step across height, 0.0 for auto calculation.")
        .SetDefault(0.0)
        .AddCustomChecker([](const float& step_h) {
          PADDLE_ENFORCE_GE(step_h,
                            0.0,
                            platform::errors::InvalidArgument(
                                "step_h should be larger "
                                "than 0. But received: step_h = %f.",
                                step_h));
        });

    AddAttr<float>("offset",
                   "(float) "
                   "Prior boxes center offset.")
        .SetDefault(0.5);
    AddAttr<bool>(
        "min_max_aspect_ratios_order",
        "(bool) If set True, the output prior box is in order of"
        "[min, max, aspect_ratios], which is consistent with Caffe."
        "Please note, this order affects the weights order of convolution layer"
        "followed by and does not affect the final detection results.")
        .SetDefault(false);
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<bool>(
        "use_quantizer",
        "(bool, default false) "
        "This parameter is no longer used. Use 'mkldnn_data_type' instead.")
        .SetDefault(false);
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"});
    AddComment(R"DOC(
Prior box operator
Generate prior boxes for SSD(Single Shot MultiBox Detector) algorithm.
Each position of the input produce N prior boxes, N is determined by
 the count of min_sizes, max_sizes and aspect_ratios, The size of the
 box is in range(min_size, max_size) interval, which is generated in
 sequence according to the aspect_ratios.

Please get more information from the following papers:
https://arxiv.org/abs/1512.02325.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(prior_box,
                            PriorBoxInferShapeFunctor,
                            PD_INFER_META(phi::PriorBoxInferMeta));

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    prior_box,
    ops::PriorBoxOp,
    ops::PriorBoxOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    PriorBoxInferShapeFunctor);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(prior_box,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    FF,
                                    ops::kPriorBoxFLOAT,
                                    ops::PriorBoxOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(prior_box,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    DD,
                                    ops::kPriorBoxDOUBLE,
                                    ops::PriorBoxOpKernel<double, double>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(prior_box,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    U8F,
                                    ops::kPriorBoxFLOAT,
                                    ops::PriorBoxOpKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(prior_box,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    S8F,
                                    ops::kPriorBoxFLOAT,
                                    ops::PriorBoxOpKernel<int8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(prior_box,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    U8D,
                                    ops::kPriorBoxDOUBLE,
                                    ops::PriorBoxOpKernel<uint8_t, double>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(prior_box,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    S8D,
                                    ops::kPriorBoxDOUBLE,
                                    ops::PriorBoxOpKernel<int8_t, double>);
