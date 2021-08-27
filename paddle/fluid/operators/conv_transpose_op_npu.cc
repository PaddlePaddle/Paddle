/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>

#include "paddle/fluid/operators/conv_transpose_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class Conv2DTransposeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // input
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* filter = context.Input<Tensor>("Filter");
    // output
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());
    // attr
    std::vector<int> output_padding =
        context.Attr<std::vector<int>>("output_padding");
    const std::vector<int> stride = context.Attr<std::vector<int>>("strides");
    std::vector<int> padding = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilation = context.Attr<std::vector<int>>("dilations");
    const std::string data_format = context.Attr<std::string>("data_format");
    int groups = context.Attr<int>("groups");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    // npu stream
    auto stream =
        context.template device_context<platform::NPUDeviceContext>().stream();

    // check dimension
    const bool channel_last = data_format == "NHWC";

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

    for (auto i = output_padding.size(); i < 4; ++i) {
      output_padding.insert(output_padding.begin(), 0);
    }
    auto output_dim_vec = framework::vectorize(output_tensor.dims());
    // CANN OP
    const auto& runner =
        NpuOpRunner("Conv2DTransposeD", {input_tensor, *filter},
                    {output_tensor}, {{"input_size", output_dim_vec},
                                      {"strides", strides},
                                      {"dilations", dilations},
                                      {"output_padding", output_padding},
                                      {"groups", groups},
                                      {"pads", padding},
                                      {"data_format", data_format}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

// conv2d
REGISTER_OP_NPU_KERNEL(
    conv2d_transpose,
    ops::Conv2DTransposeNPUKernel<plat::NPUDeviceContext, plat::float16>);
