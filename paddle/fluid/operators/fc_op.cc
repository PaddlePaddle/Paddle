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

#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/infermeta/ternary.h"

#include <vector>

namespace paddle {
namespace operators {

class FCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Input");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class FCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor), The input tensor of fully connected operator.");
    AddInput("W", "(Tensor), The weight fc op with shape (I, O).");
    AddInput("Bias", "(Tensor, optional) Bias vector with shape (1 x O")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor) The output tensor of fully connected operator. ");
    AddAttr<int>("in_num_col_dims",
                 "(int, default 1), The fc op can take tensors with more than "
                 "two dimensions as its inputs.")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<std::string>("activation_type",
                         "Activation type used in fully connected operator.")
        .SetDefault("");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<bool>(
        "padding_weights",
        "(bool, default false) When padding weights in the fc fuse pass, "
        "the 'padding_weights' attribute is set as true.")
        .SetDefault(false);
    AddAttr<bool>(framework::kAllKernelsMustComputeRuntimeShape,
                  "Skip calling InferShape() function in the runtime.")
        .SetDefault(true);
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
    /* int8 parameters */
    AddAttr<float>("Scale_in",
                   "(float, default 1.0f), The quantize scale of input data")
        .SetDefault(1.0f);
    AddAttr<std::vector<float>>("Scale_weights",
                                "(std::vector<float>, default {1.0f}), The "
                                "quantize scale of weights data")
        .SetDefault({1.0f});
    AddAttr<float>("Scale_out",
                   "(float, default 1.0f), The quantize scale of output data")
        .SetDefault(1.0f);
    AddAttr<bool>("force_fp32_output",
                  "(bool, default false) Force INT8 kernel output FP32, only "
                  "used in MKL-DNN INT8")
        .SetDefault(false);
    AddComment(R"DOC(
Fully Connected Operator.

The fully connected operation calculates the output based on the input, weights and bias.
The size of each dimension of the parameters checked in the infer-shape.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(fc,
                            FcInferShapeFunctor,
                            PD_INFER_META(phi::FcInferMeta));

REGISTER_OPERATOR(
    fc,
    ops::FCOp,
    ops::FCOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    FcInferShapeFunctor);
