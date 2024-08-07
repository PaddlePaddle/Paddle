/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/generator/get_expected_kernel_func.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle::operators {

class FusedConvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor), input 0 of conv2d op.");
    AddInput("Filter", "(Tensor), input 1 of conv2d op.");
    AddOutput("Output", "(Tensor), output 0 of conv2d op.");
    AddAttr<std::vector<int>>("strides",
                              "(std::vector<int>), attribute 0 for conv2d op.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(std::vector<int>), attribute 1 for conv2d op.")
        .SetDefault({0, 0});
    AddAttr<std::string>("padding_algorithm",
                         "(std::string), attribute 2 for conv2d op.")
        .SetDefault("EXPLICIT");
    AddAttr<std::vector<int>>("dilations",
                              "(std::vector<int>), attribute 3 for conv2d op.")
        .SetDefault({1, 1});
    AddAttr<int>("groups", "(int), attribute 4 for conv2d op.").SetDefault(1);
    AddAttr<std::string>("data_format",
                         "(std::string), attribute 5 for conv2d op.")
        .SetDefault("NCHW");
    AddComment(R"DOC(
TODO: Documentation of conv2d op.
)DOC");
    Apply();
  }

 protected:
  void Apply() {
    AddInput("Bias",
             "(Tensor) Bias to be added to each output of filter application."
             "The format of output tensor is X (one-dimensional) of size equal"
             "to the number of output channels. Only used with MKL-DNN.")
        .AsDispensable();
    AddInput("ResidualData",
             "(Tensor) Tensor with residual data "
             "to which convolution output will be added."
             "Used with fuse_residual_connection fusion.")
        .AsDispensable();
    AddAttr<std::string>("fuse_activation",
                         "(string, default \"\") Only used in mkldnn kernel")
        .SetDefault("");
    AddAttr<bool>("fuse_residual_connection",
                  "(bool, default false) Only used in mkldnn kernel. Used "
                  "whenever convolution output is as an input to residual "
                  "connection.")
        .SetDefault(false);
    AddAttr<bool>("force_fp32_output",
                  "(bool, default false) Force INT8 kernel output FP32, only "
                  "used in MKL-DNN INT8")
        .SetDefault(false);
    AddComment(R"DOC(
Convolution Operator.

The convolution operation calculates the output based on the input, filter
and strides, paddings, dilations, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input) and Output(Output) are in NCHW or NHWC format. Where N is batch
size, C is the number of channels, H is the height of the feature, and W is
the width of the feature.
Filters(Input) is MCHW format format. Where M is the number of output image channels, C is
the number of input image channels, H is the height of the filter, and W
is the width of the filter.
Parameters(strides, paddings, dilations) are two elements. These two elements represent
height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       Input shape: $(N, C_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{out}, C_{in}, H_f, W_f)$
  Output:
  ?
       Output shape: $(N, C_{out}, H_{out}, W_{out})$
  Where
$$
       H_{out}= \frac{(H_{in} + pad_height_top + pad_height_bottom - (dilations[0] * (H_f - 1) + 1))}{strides[0]}+ 1 \\
       W_{out}= \frac{(W_{in} + pad_width_left + pad_width_right - (dilations[1] * (W_f - 1) + 1))}{strides[1]}+ 1
$$
)DOC");
  }
};

class FusedConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetConvExpectedKernelType(ctx, this);
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(fused_conv2d,
                            FusedConv2DInferShapeFunctor,
                            PD_INFER_META(phi::FusedConvInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(fused_conv3d,
                            FusedConv3DInferShapeFunctor,
                            PD_INFER_META(phi::FusedConvInferMeta));

// fused_conv2d is only used for onednn inference.
REGISTER_OPERATOR(
    fused_conv2d,
    ops::FusedConvOp,
    ops::FusedConvOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    FusedConv2DInferShapeFunctor);

// fused_conv3d is only used for onednn inference.
REGISTER_OPERATOR(
    fused_conv3d,
    ops::FusedConvOp,
    ops::FusedConvOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    FusedConv3DInferShapeFunctor);
