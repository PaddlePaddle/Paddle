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

#include "paddle/fluid/operators/conv_op.h"

namespace paddle {
namespace operators {

class FusedConvOpMaker : public Conv2DOpMaker {
 protected:
  void Apply() override {
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"});
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
    AddAttr<bool>("use_mkldnn", "(bool, default false) Used in mkldnn kernel")
        .SetDefault(true);
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

// fused_conv2d is only used for onednn inference.
REGISTER_OPERATOR(
    fused_conv2d,
    ops::ConvOp,
    ops::FusedConvOpMaker,
    ops::ConvOpInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

// fused_conv3d is only used for onednn inference.
REGISTER_OPERATOR(
    fused_conv3d,
    ops::ConvOp,
    ops::FusedConvOpMaker,
    ops::ConvOpInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
