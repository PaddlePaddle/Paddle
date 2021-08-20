// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class DepthwiseConvNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // input
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* filter = context.Input<Tensor>("Filter");
    // output
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());
    // attr
    const std::vector<int> stride = context.Attr<std::vector<int>>("strides");
    std::vector<int> padding = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilation = context.Attr<std::vector<int>>("dilations");
    const std::string data_format = context.Attr<std::string>("data_format");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    // npu stream
    auto stream =
        context.template device_context<platform::NPUDeviceContext>().stream();

    // check dimension
    const bool channel_last = data_format == "NHWC";
    if (channel_last) {
      // NHWC
      PADDLE_ENFORCE_EQ(
          output->dims()[output->dims().size() - 1],
          input->dims()[input->dims().size() - 1],
          platform::errors::InvalidArgument(
              "ShapeError: The output channels must be equal to the "
              "input channels. But receivced output channel number is %d "
              "and input channel number is %d",
              output->dims()[output->dims().size() - 1],
              input->dims()[input->dims().size() - 1]));
    } else {
      // NCHW
      PADDLE_ENFORCE_EQ(
          output->dims()[1], input->dims()[1],
          platform::errors::InvalidArgument(
              "ShapeError: The output channels must be equal to the "
              "input channels. But receivced output channel number is %d "
              "and input channel number is %d",
              output->dims()[1], input->dims()[1]));
    }

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    }
    filter_data_dims = framework::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&padding, &dilation, padding_algorithm,
                             in_data_dims, stride, ksize);

    // Transform filter (n, 1, h, w) --> (1, n, h, w)
    Tensor transformed_filter(filter->type());
    transformed_filter.mutable_data<T>({filter->dims()[1], filter->dims()[0],
                                        filter->dims()[2], filter->dims()[3]},
                                       context.device_context().GetPlace());
    std::vector<int> perm = {1, 0, 2, 3};
    const auto& runner_trans = NpuOpRunner(
        "TransposeD", {*filter}, {transformed_filter}, {{"perm", perm}});
    runner_trans.Run(stream);

    // construct NPU attr
    std::vector<int> strides(4, 1);
    std::vector<int> dilations(4, 1);

    Tensor input_tensor, output_tensor;
    input_tensor.ShareDataWith(*input);
    output_tensor.ShareDataWith(*output);

    if (channel_last) {
      input_tensor.set_layout(DataLayout::kNHWC);
      output_tensor.set_layout(DataLayout::kNHWC);
      strides[1] = stride[0];
      strides[2] = stride[1];
      dilations[1] = dilation[0];
      dilations[2] = dilation[1];
    } else {
      strides[2] = stride[0];
      strides[3] = stride[1];
      dilations[2] = dilation[0];
      dilations[3] = dilation[1];
    }

    // CANN OP
    const auto& runner =
        NpuOpRunner("DepthwiseConv2D", {input_tensor, transformed_filter},
                    {output_tensor}, {{"strides", strides},
                                      {"dilations", dilations},
                                      {"pads", padding},
                                      {"data_format", data_format}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    depthwise_conv2d,
    ops::DepthwiseConvNPUKernel<paddle::platform::NPUDeviceContext,
                                paddle::platform::float16>);
