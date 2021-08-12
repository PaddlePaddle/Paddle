// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T = int>
inline void UpdatePaddingAndDilationNPU(std::vector<T>* paddings,
                                        std::vector<T>* dilation,
                                        const std::string padding_algorithm,
                                        const framework::DDim data_dims,
                                        const std::vector<T>& strides,
                                        const std::vector<T>& ksize) {
  // set padding size == data_dims.size() * 2
  auto data_shape = framework::vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        data_dims.size() * 2, paddings->size(),
        platform::errors::InvalidArgument(
            "Attribute padding's size should be the same or twice as the "
            "input's dimension. "
            "But recieved: padding's size is %d, padding is [%s]; input's "
            "dimension is %d, input's shape is [%s].",
            paddings->size(), framework::make_ddim(*paddings), data_dims.size(),
            data_dims));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;

      // dilation
      *(dilation->begin() + i) = 1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

template <typename DeviceContext, typename T>
inline void ResizeChannelFirst(const framework::ExecutionContext& context,
                               const Tensor* input, Tensor* transformed_input) {
  transformed_input->Resize(input->dims());
  auto in_dims_vec = framework::vectorize(input->dims());
  in_dims_vec[1] = input->dims()[3];
  in_dims_vec[2] = input->dims()[1];
  in_dims_vec[3] = input->dims()[2];
  transformed_input->Resize(framework::make_ddim(in_dims_vec));
  transformed_input->mutable_data<T>(context.GetPlace());
}

template <typename DeviceContext, typename T>
inline void ResizeChannelLast(const framework::ExecutionContext& context,
                              const Tensor* input, Tensor* transformed_input) {
  transformed_input->Resize(input->dims());
  auto in_dims_vec = framework::vectorize(input->dims());
  in_dims_vec[1] = input->dims()[2];
  in_dims_vec[2] = input->dims()[3];
  in_dims_vec[3] = input->dims()[1];
  transformed_input->Resize(framework::make_ddim(in_dims_vec));
  transformed_input->mutable_data<T>(context.GetPlace());
}

template <typename DeviceContext, typename T>
inline void TransChannelFirst(const framework::ExecutionContext& context,
                              const Tensor* input, Tensor* transformed_input) {
  auto stream =
      context.template device_context<platform::NPUDeviceContext>().stream();
  std::vector<int> perm = {0, 3, 1, 2};
  const auto& runner_trans = NpuOpRunner(
      "TransposeD", {*input}, {*transformed_input}, {{"perm", perm}});
  runner_trans.Run(stream);
}

template <typename DeviceContext, typename T>
inline void TransChannelLast(const framework::ExecutionContext& context,
                             const Tensor* input, Tensor* transformed_input) {
  auto stream =
      context.template device_context<platform::NPUDeviceContext>().stream();
  std::vector<int> perm = {0, 2, 3, 1};
  const auto& runner_trans = NpuOpRunner(
      "TransposeD", {*input}, {*transformed_input}, {{"perm", perm}});
  runner_trans.Run(stream);
}

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
    VLOG(3) << padding_algorithm;
    // npu stream
    auto stream =
        context.template device_context<platform::NPUDeviceContext>().stream();

    // check dimension
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");
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

    Tensor transformed_input(input->type());
    Tensor transformed_output(output->type());
    if (channel_last) {
      // Transform input&output: NHWC --> NCHW
      VLOG(3) << "transform NHWC to NCHW";
      ResizeChannelFirst<DeviceContext, T>(context, input, &transformed_input);
      TransChannelFirst<DeviceContext, T>(context, input, &transformed_input);

      ResizeChannelFirst<DeviceContext, T>(context, output,
                                           &transformed_output);
    } else {
      transformed_input = *input;
      transformed_output = *output;
    }

    // update padding and dilation
    auto in_dims = transformed_input.dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims =
        framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilationNPU(&padding, &dilation, padding_algorithm,
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
    std::vector<int> strides = {1, 1, stride[0], stride[1]};
    std::vector<int> dilations = {1, 1, dilation[0], dilation[1]};

    // CANN OP
    const auto& runner = NpuOpRunner(
        "DepthwiseConv2D", {transformed_input, transformed_filter},
        {transformed_output}, {{"strides", strides},
                               {"dilations", dilations},
                               {"pads", padding},
                               {"data_format", std::string("NCHW")}});
    runner.Run(stream);

    // Transform output: NCHW --> NHWC
    if (channel_last) {
      TransChannelLast<DeviceContext, T>(context, &transformed_output, output);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    depthwise_conv2d,
    ops::DepthwiseConvNPUKernel<paddle::platform::NPUDeviceContext,
                                paddle::platform::float16>);
