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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

class FusedScaleBiasReluConvBnstatsOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // inputs
    AddInput("Input", "");
    AddInput("Filter", "");
    AddInput("Scale", "").AsDispensable();
    AddInput("Bias", "").AsDispensable();
    // outputs
    AddOutput("Output", "");
    AddOutput("SumOutput", "");
    AddOutput("SqSumOutput", "");
    // conv params
    // conv params
    AddAttr<std::vector<int>>("strides",
                              "(vector<int> default:{1, 1}), the "
                              "strides(h_stride, w_stride) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(vector<int> default:{0, 0}), the "
                              "paddings(pad_height_top, pad_height_bottom, "
                              "pad_width_left, pad_wifth_right)  of "
                              "convolution operator.")
        .SetDefault({0, 0});
    AddAttr<std::string>(
        "padding_algorithm",
        "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
        "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
        "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
        .SetDefault("EXPLICIT");
    AddAttr<std::vector<int>>("dilations",
                              "(vector<int> default:{1, 1}), the "
                              "dilations(h_dilation, w_dilation) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<int>(
        "groups",
        "(int default:1), the groups number of the convolution operator. ")
        .SetDefault(1);
    AddAttr<std::string>("data_format", "(string, NHWC) Must be NHWC.")
        .SetDefault("NHWC");
    // fusion options
    AddAttr<bool>(
        "fuse_prologue",
        R"DOC((bool, default true). Whether to fuse scale bias relu.)DOC")
        .SetDefault(true);
    AddComment(R"DOC(
FusedScaleBiasReluConvBnstats Operator
It fuses the following operations:
Output = Conv(ReLU(Input * Scale + Bias))
SumOutput = Output.sum([0,1,2])
SqSumOutput = (Output**2).sum([0,1,2])

Optionally, if fuse_prologue is false, there will be no
scale and bias:
Output = Conv(Input)

Requirements:
- data_format should be NHWC.
- Input, Output should have NHWC layout and FP16 dtype.
- Scale, Bias should have shape [K], where K is the first
dimension of Filter. The dtype should be FP16
- SumOutput, SqSumOutput should have shape [K] and dtype FP32.
)DOC");
  }
};

class FusedScaleBiasReluConvBnstatsOp : public operators::ConvOp {
 public:
  using operators::ConvOp::ConvOp;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"),
                   "Input",
                   "Input",
                   "FusedScaleBiasReluConvBnstats");
    bool fuse_prologue = ctx->Attrs().Get<bool>("fuse_prologue");
    if (fuse_prologue) {
      OP_INOUT_CHECK(ctx->HasInput("Scale"),
                     "Input",
                     "Scale",
                     "FusedScaleBiasReluConvBnstats");
      OP_INOUT_CHECK(ctx->HasInput("Bias"),
                     "Input",
                     "Bias",
                     "FusedScaleBiasReluConvBnstats");
    }
    OP_INOUT_CHECK(ctx->HasOutput("Output"),
                   "Output",
                   "Output",
                   "FusedScaleBiasReluConvBnstats");
    OP_INOUT_CHECK(ctx->HasOutput("SumOutput"),
                   "Output",
                   "SumOutput",
                   "FusedScaleBiasReluConvBnstats");
    OP_INOUT_CHECK(ctx->HasOutput("SqSumOutput"),
                   "Output",
                   "SqSumOutput",
                   "FusedScaleBiasReluConvBnstats");

    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_EQ(
        in_dims.size(),
        4U,
        platform::errors::InvalidArgument(
            "The input's dimension of Operator(FusedScaleBiasReluConvBnstats) "
            "is expected "
            "to be 4. But received: input's dimension = %u, shape = [%s].",
            in_dims.size(),
            in_dims));

    // Check if data format is NHWC".
    std::string data_format = ctx->Attrs().Get<std::string>("data_format");
    PADDLE_ENFORCE_EQ(
        data_format,
        "NHWC",
        platform::errors::PermissionDenied(
            "Operator(FusedScaleBiasReluConvBnstats) only supports data format "
            "of "
            "channel last (NHWC) now. But recieved: data_format = '%s'.",
            data_format));
    int groups = ctx->Attrs().Get<int>("groups");
    PADDLE_ENFORCE_EQ(groups,
                      1,
                      platform::errors::InvalidArgument(
                          "Expect group to be 1, got %d.", groups));

    if (fuse_prologue) {
      const int64_t C = in_dims[in_dims.size() - 1];
      auto scale_dim = ctx->GetInputDim("Scale");
      auto bias_dim = ctx->GetInputDim("Bias");

      PADDLE_ENFORCE_EQ(
          scale_dim.size(),
          1UL,
          platform::errors::PreconditionNotMet(
              "ShapeError: the dimension of scale must equal to 1."
              "But received: the shape of scale is [%s], the dimension "
              "of scale is [%d]",
              scale_dim,
              scale_dim.size()));
      PADDLE_ENFORCE_EQ(
          bias_dim.size(),
          1UL,
          platform::errors::PreconditionNotMet(
              "ShapeError: the dimension of bias must equal to 1."
              "But received: the shape of bias is [%s],the dimension "
              "of bias is [%d]",
              bias_dim,
              bias_dim.size()));
      bool check = true;
      if ((!ctx->IsRuntime()) &&
          (phi::product(scale_dim) <= 0 || phi::product(bias_dim) <= 0)) {
        check = false;
      }

      if (check) {
        PADDLE_ENFORCE_EQ(
            scale_dim[0],
            C,
            platform::errors::PreconditionNotMet(
                "ShapeError: the shape of scale must equal to [%d]"
                "But received: the shape of scale is [%d]",
                C,
                scale_dim[0]));
        PADDLE_ENFORCE_EQ(bias_dim[0],
                          C,
                          platform::errors::PreconditionNotMet(
                              "ShapeError: the shape of bias must equal to [%d]"
                              "But received: the shape of bias is [%d]",
                              C,
                              bias_dim[0]));
      }
    }
    std::vector<int64_t> output_shape = ComputeOutputShape(ctx);
    ctx->SetOutputDim("Output", phi::make_ddim(output_shape));

    const int64_t K = output_shape[output_shape.size() - 1];
    ctx->SetOutputDim("SumOutput", {K});
    ctx->SetOutputDim("SqSumOutput", {K});
    ctx->ShareLoD("Input", "Output");
  }

  std::vector<int64_t> ComputeOutputShape(
      framework::InferShapeContext* ctx) const {
    OP_INOUT_CHECK(ctx->HasInput("Input"),
                   "Input",
                   "Input",
                   "FusedScaleBiasReluConvBnstats");
    OP_INOUT_CHECK(ctx->HasInput("Filter"),
                   "Input",
                   "Filter",
                   "FusedScaleBiasReluConvBnstats");

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
          dilations[i],
          0,
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
        in_dims.size() == 4 || in_dims.size() == 5,
        true,
        platform::errors::InvalidArgument(
            "The input of Op(Conv) should be a 4-D or 5-D Tensor. But "
            "received: input's dimension is %u, input's shape is [%s].",
            in_dims.size(),
            in_dims));

    PADDLE_ENFORCE_EQ(
        in_dims.size(),
        filter_dims.size(),
        platform::errors::InvalidArgument(
            "The input's dimension and filter's dimension of "
            "Op(Conv) should be equal. But received: the input's shape is "
            "[%s], "
            "the input's dimension is %d; the filter's shape is [%s],  "
            "the filter's dimension is %d.",
            in_dims,
            in_dims.size(),
            filter_dims,
            filter_dims.size()));

    int stride_size = strides.size();
    for (int i = 0; i < stride_size; ++i) {
      PADDLE_ENFORCE_GT(
          strides[i],
          0,
          platform::errors::InvalidArgument(
              "The stride of Op(Conv) should be larget than 0, but received "
              "stride is %d.",
              strides[i]));
    }

    int in_sub_stride_size = in_dims.size() - stride_size;
    PADDLE_ENFORCE_EQ(
        in_dims.size(),
        strides.size() + 2U,
        platform::errors::InvalidArgument(
            "The difference of input's dimension and Attr(strides)'s "
            "length must be euqal to 2 for Op(Conv). "
            "But received: input's dimension is %d, input's shape is [%s]; "
            "Attr(stride)'s length is %d, Attr(stride) is [%s]; "
            "difference of input's dimention and Attr(strides)'s length = %u.",
            in_dims.size(),
            in_dims,
            strides.size(),
            phi::make_ddim(strides),
            in_sub_stride_size));

    const auto input_channels =
        channel_last ? in_dims[in_dims.size() - 1] : in_dims[1];

    PADDLE_ENFORCE_EQ(
        input_channels,
        filter_dims[1] * groups,
        platform::errors::InvalidArgument(
            "The number of input's channels should be equal to filter's "
            "channels "
            "* groups for Op(Conv). But received: the input's channels is %d, "
            "the input's shape is [%s]; the filter's channels is %d, the "
            "filter's shape is [%s]; the groups is %d, the data_format is %s. "
            "The error may come from wrong data_format setting.",
            input_channels,
            in_dims,
            filter_dims[1],
            filter_dims,
            groups,
            data_format));
    PADDLE_ENFORCE_EQ(
        filter_dims[0] % groups,
        0,
        platform::errors::InvalidArgument(
            "The number of output's channels (filter's first dimension) of "
            "Op(Conv) should be divided by groups. But received: "
            "the output channels is %d, the filter's shape is [%s], "
            "the groups is %d.",
            filter_dims[0],
            filter_dims,
            groups));

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_GT(
          filter_dims[0],
          0,
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
    UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

    std::vector<int64_t> output_shape({in_dims[0]});
    if (!channel_last) {
      output_shape.push_back(filter_dims[0]);
    }
    for (int i = 0; i < in_data_dims.size(); ++i) {
      if ((!ctx->IsRuntime()) &&
          (in_data_dims[i] <= 0 || filter_dims[i + 2] <= 0)) {
        output_shape.push_back(-1);
      } else {
        output_shape.push_back(ConvOutputSize(in_data_dims[i],
                                              filter_data_dims[i],
                                              dilations[i],
                                              paddings[2 * i],
                                              paddings[2 * i + 1],
                                              strides[i]));
      }
    }
    if (channel_last) {
      output_shape.push_back(filter_dims[0]);
    }

    return output_shape;
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Input");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fused_scale_bias_relu_conv_bnstats,
                             ops::FusedScaleBiasReluConvBnstatsOp,
                             ops::FusedScaleBiasReluConvBnstatsOpMaker);
