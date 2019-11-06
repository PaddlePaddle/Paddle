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
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace operators {

// This fused conv follows the equation:
//   y = act ( alpha1 * conv(x) + alpha2 * z + bias ).
//   here, y is Output,
//         x is Input,
//         z is ResidualData,
//         bias is Bias
// When `split_channels` is set, y will be splitted into multiple outputs,
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

class Conv2DFusionOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      "Input(Input) of ConvOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Filter"), true,
                      "Input(Filter) of ConvOp should not be null.");
    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");

    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    std::vector<int> dilations =
        ctx->Attrs().Get<std::vector<int>>("dilations");
    std::string padding_algorithm =
        ctx->Attrs().Get<std::string>("padding_algorithm");

    framework::DDim in_data_dims;
    in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());

    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    std::vector<int64_t> output_shape({in_dims[0]});
    output_shape.push_back(filter_dims[0]);

    for (size_t i = 0; i < in_data_dims.size(); ++i) {
      if ((!ctx->IsRuntime()) &&
          (in_data_dims[i] <= 0 || filter_dims[i + 2] <= 0)) {
        output_shape.push_back(-1);
      } else {
        output_shape.push_back(
            ConvOutputSize(in_data_dims[i], filter_dims[i + 2], dilations[i],
                           paddings[2 * i], paddings[2 * i + 1], strides[i]));
      }
    }

    PADDLE_ENFORCE_EQ(ctx->HasOutput("Output"), true,
                      "Output(Output) of ConvOp should not be null.");
    ctx->SetOutputDim("Output", framework::make_ddim(output_shape));

    std::vector<int> channels =
        ctx->Attrs().Get<std::vector<int>>("split_channels");
    if (channels.size()) {
      PADDLE_ENFORCE_EQ(ctx->HasOutputs("Outputs"), true,
                        "Output(Outputs) of ConvOp should not be null.");
      std::vector<framework::DDim> oshapes;
      oshapes.reserve(channels.size());
      for (size_t i = 0; i < channels.size(); ++i) {
        oshapes.push_back(
            {output_shape[0], channels[i], output_shape[2], output_shape[3]});
      }
      ctx->SetOutputsDim("Outputs", oshapes);
    }
  }
};

// TODO(qingqing): add gradient operator for conv2d_fusion

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    conv2d_fusion, ops::ConvOp, ops::Conv2DFusionOpMaker,
    ops::Conv2DFusionOpInferShape, ops::ConvOpInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
