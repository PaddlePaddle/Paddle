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

#include <string>
#include <vector>
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

// This fused conv follows the equation:
//   y = act ( alpha1 * conv(x) + alpha2 * z + bias ).
//   here, y is Output,
//         x is Input,
//         z is ResidualData,
//         bias is Bias
// When `split_channels` is set, y will be split into multiple outputs,
// each output has split_channels[i] number of channels.
class Conv2DFusionOpMaker : public Conv2DOpMaker {
 protected:
  void Apply() override {
    AddAttr<std::string>(
        "activation",
        "The activation type can be 'identity', 'sigmoid', 'relu', 'relu6' "
        "'relux' , 'tanh', 'band_pass'")
        .SetDefault("relu");
    AddAttr<std::vector<int>>(
        "split_channels",
        "When `split_channels` are set, there will be multiple outputs, the "
        "output size is equal to the number of `split_channels`.")
        .SetDefault({});
    AddOutput("Outputs",
              "This Outputs is used when setting `split_channels`."
              "Usually used to fuse conv with same input and same filter size, "
              "padding, stride, dilation size.")
        .AsDuplicable()
        .AsDispensable();
    AddInput("AlgoCache",
             "The cache of convolution algorithm, a RAW type variable.")
        .AsDispensable();
    AddAttr<int>(
        "search_times",
        "The number of exhaustive search times for convolution algorithm.")
        .SetDefault(-1);
  }
};

class Conv2DFusionOp : public operators::ConvOp {
 public:
  using operators::ConvOp::ConvOp;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "Conv2DFusion");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "Conv2DFusion");

    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_EQ(
        in_dims.size(), 4U,
        platform::errors::InvalidArgument(
            "The input's dimension of Operator(Conv2DFusion) is expected "
            "to be 4. But received: input's dimension = %u, shape = [%s].",
            in_dims.size(), in_dims));

    // In some case, attribute data_format is "AnyLayout".
    std::string data_format = ctx->Attrs().Get<std::string>("data_format");
    PADDLE_ENFORCE_NE(
        data_format, "NHWC",
        platform::errors::PermissionDenied(
            "Operator(Conv2DFusion) only supports data format of "
            "channel first (NCHW) now. But recieved: data_format = '%s'.",
            data_format));

    std::vector<int64_t> output_shape = ComputeOutputShape(ctx);
    ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
    ctx->ShareLoD("Input", "Output");

    std::vector<int> split_channels =
        ctx->Attrs().Get<std::vector<int>>("split_channels");
    if (split_channels.size()) {
      OP_INOUT_CHECK(ctx->HasOutputs("Outputs"), "Output", "Outputs",
                     "Conv2DFusion");
      PADDLE_ENFORCE_EQ(
          ctx->Outputs("Outputs").size(), split_channels.size(),
          platform::errors::InvalidArgument(
              "The number of Output(Outputs) of operator 'Conv2DFusion' is "
              "expected to be equal to the length of Attr(split_channels). But "
              "reiceved: the number of Output(Outputs) = %u; the length of "
              "Attr(split_channels) = %u, the content = [%s].",
              ctx->Outputs("Outputs").size(), split_channels.size(),
              framework::make_ddim(split_channels)));

      int split_channels_sum = 0;
      std::vector<framework::DDim> output_shapes(split_channels.size());
      for (size_t i = 0; i < split_channels.size(); ++i) {
        split_channels_sum += split_channels[i];
        output_shapes[i] =
            framework::make_ddim({output_shape[0], split_channels[i],
                                  output_shape[2], output_shape[3]});
      }
      PADDLE_ENFORCE_EQ(
          split_channels_sum, output_shape[1],
          platform::errors::InvalidArgument(
              "The sum of Attr(split_channels) is expected to be equal to the "
              "total output channels. But recieved: the sum of "
              "Attr(split_channels) = %d, the total output channels = %d.",
              split_channels_sum, output_shape[1]));

      ctx->SetOutputsDim("Outputs", output_shapes);
    }
  }
};

// TODO(qingqing): add gradient operator for conv2d_fusion

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    conv2d_fusion, ops::Conv2DFusionOp, ops::Conv2DFusionOpMaker,
    ops::ConvOpInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
