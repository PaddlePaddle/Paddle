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
    ctx->SetOutputDim("Output", phi::make_ddim(output_shape));
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
              phi::make_ddim(split_channels)));

      int split_channels_sum = 0;
      std::vector<framework::DDim> output_shapes(split_channels.size());
      for (size_t i = 0; i < split_channels.size(); ++i) {
        split_channels_sum += split_channels[i];
        output_shapes[i] = phi::make_ddim({output_shape[0], split_channels[i],
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

  std::vector<int64_t> ComputeOutputShape(
      framework::InferShapeContext* ctx) const {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "Conv");
    OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter", "Conv");

    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");

    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    std::string padding_algorithm =
        ctx->Attrs().Get<std::string>("padding_algorithm");
    int groups = ctx->Attrs().Get<int>("groups");
    std::vector<int> dilations =
        ctx->Attrs().Get<std::vector<int>>("dilations");
    int dilation_size = dilations.size();
    for (int i = 0; i < dilation_size; ++i) {
      PADDLE_ENFORCE_GT(
          dilations[i], 0,
          platform::errors::InvalidArgument(
              "The dilation of Op(Conv) should be larget than 0, but received "
              "dilation is %d.",
              dilations[i]));
    }
    const std::string data_format =
        ctx->Attrs().Get<std::string>("data_format");

    // MKL-DNN Kernels are using NCHW order of dims description
    // so we ignore data_format consideration for MKL-DNN kernel
    const bool channel_last = (ctx->IsRunMKLDNNKernel() == false) &&
                              (data_format == "NHWC" || data_format == "NDHWC");

    PADDLE_ENFORCE_EQ(
        in_dims.size() == 4 || in_dims.size() == 5, true,
        platform::errors::InvalidArgument(
            "The input of Op(Conv) should be a 4-D or 5-D Tensor. But "
            "received: input's dimension is %u, input's shape is [%s].",
            in_dims.size(), in_dims));

    PADDLE_ENFORCE_EQ(
        in_dims.size(), filter_dims.size(),
        platform::errors::InvalidArgument(
            "The input's dimension and filter's dimension of "
            "Op(Conv) should be equal. But received: the input's shape is "
            "[%s], "
            "the input's dimension is %d; the filter's shape is [%s],  "
            "the filter's dimension is %d.",
            in_dims, in_dims.size(), filter_dims, filter_dims.size()));

    int stride_size = strides.size();
    for (int i = 0; i < stride_size; ++i) {
      PADDLE_ENFORCE_GT(
          strides[i], 0,
          platform::errors::InvalidArgument(
              "The stride of Op(Conv) should be larget than 0, but received "
              "stride is %d.",
              strides[i]));
    }

    int in_sub_stride_size = in_dims.size() - stride_size;
    PADDLE_ENFORCE_EQ(
        in_dims.size(), strides.size() + 2U,
        platform::errors::InvalidArgument(
            "The difference of input's dimension and Attr(strides)'s "
            "length must be euqal to 2 for Op(Conv). "
            "But received: input's dimension is %d, input's shape is [%s]; "
            "Attr(stride)'s length is %d, Attr(stride) is [%s]; "
            "difference of input's dimention and Attr(strides)'s length = %u.",
            in_dims.size(), in_dims, strides.size(), phi::make_ddim(strides),
            in_sub_stride_size));

    const auto input_channels =
        channel_last ? in_dims[in_dims.size() - 1] : in_dims[1];

    PADDLE_ENFORCE_EQ(
        input_channels, filter_dims[1] * groups,
        platform::errors::InvalidArgument(
            "The number of input's channels should be equal to filter's "
            "channels "
            "* groups for Op(Conv). But received: the input's channels is %d, "
            "the input's shape is [%s]; the filter's channels is %d, the "
            "filter's shape is [%s]; the groups is %d, the data_format is %s. "
            "The error may come from wrong data_format setting.",
            input_channels, in_dims, filter_dims[1], filter_dims, groups,
            data_format));
    PADDLE_ENFORCE_EQ(
        filter_dims[0] % groups, 0,
        platform::errors::InvalidArgument(
            "The number of output's channels (filter's first dimension) of "
            "Op(Conv) should be divided by groups. But received: "
            "the output channels is %d, the filter's shape is [%s], "
            "the groups is %d.",
            filter_dims[0], filter_dims, groups));

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_GT(
          filter_dims[0], 0,
          platform::errors::InvalidArgument(
              "the size of filter at axis 0 should be greater than 0"));
    }

    framework::DDim in_data_dims;
    if (channel_last) {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    }

    framework::DDim filter_data_dims =
        phi::slice_ddim(filter_dims, 2, filter_dims.size());

    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    std::vector<int64_t> output_shape({in_dims[0]});
    if (!channel_last) {
      output_shape.push_back(filter_dims[0]);
    }
    for (int i = 0; i < in_data_dims.size(); ++i) {
      if ((!ctx->IsRuntime()) &&
          (in_data_dims[i] <= 0 || filter_dims[i + 2] <= 0)) {
        output_shape.push_back(-1);
      } else {
        output_shape.push_back(
            ConvOutputSize(in_data_dims[i], filter_data_dims[i], dilations[i],
                           paddings[2 * i], paddings[2 * i + 1], strides[i]));
      }
    }
    if (channel_last) {
      output_shape.push_back(filter_dims[0]);
    }

    return output_shape;
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
