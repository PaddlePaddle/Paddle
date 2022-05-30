/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/conv_transpose_op.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using NPUDeviceContext = platform::NPUDeviceContext;

template <typename T>
class Conv2DTransposeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    Tensor* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());
    std::vector<int> output_padding =
        ctx.Attr<std::vector<int>>("output_padding");
    const std::vector<int> stride = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> padding = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilation = ctx.Attr<std::vector<int>>("dilations");
    const std::string data_format = ctx.Attr<std::string>("data_format");
    int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");

    // check dimension
    const bool channel_last = data_format == "NHWC";

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    }
    filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    phi::UpdatePaddingAndDilation(&padding, &dilation, padding_algorithm,
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
    auto output_dim_vec = phi::vectorize(output_tensor.dims());

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
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

template <typename T>
class Conv2DTransposeGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    const Tensor* output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    if ((!input_grad) && (!filter_grad)) return;

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const int groups = ctx.Attr<int>("groups");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");
    const framework::DataLayout data_layout =
        framework::StringToDataLayout(data_format);

    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    // auto out_grad_dims = output_grad->dims();
    // const int batch_size = static_cast<int>(input->dims()[0]);

    const bool channel_last = (data_layout == framework::DataLayout::kNHWC);

    framework::DDim in_data_dims;
    if (channel_last) {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    }
    framework::DDim filter_data_dims =
        phi::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    phi::UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                                  in_data_dims, strides, ksize);

    std::vector<int> strides_vec(4, 1);
    std::vector<int> dilations_vec(4, 1);

    Tensor input_tensor, output_grad_tensor;
    input_tensor.ShareDataWith(*input);
    output_grad_tensor.ShareDataWith(*output_grad);
    if (channel_last) {
      input_tensor.set_layout(DataLayout::kNHWC);
      output_grad_tensor.set_layout(DataLayout::kNHWC);
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      dilations_vec[1] = dilations[0];
      dilations_vec[2] = dilations[1];
    } else {
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
      dilations_vec[2] = dilations[0];
      dilations_vec[3] = dilations[1];
    }

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
      const auto& runner = NpuOpRunner(
          "Conv2DBackpropFilterD", {output_grad_tensor, input_tensor},
          {*filter_grad}, {{"filter_size", phi::vectorize<int>(filter_dims)},
                           {"strides", strides_vec},
                           {"pads", paddings},
                           {"dilations", dilations_vec},
                           {"groups", groups},
                           {"data_format", data_format}});
      runner.Run(stream);
    }
    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
      Tensor input_grad_tensor;
      input_grad_tensor.ShareDataWith(*input_grad);
      if (channel_last) {
        input_grad_tensor.set_layout(DataLayout::kNHWC);
      }
      const auto& runner =
          NpuOpRunner("Conv2D", {output_grad_tensor, *filter},
                      {input_grad_tensor}, {{"strides", strides_vec},
                                            {"pads", paddings},
                                            {"dilations", dilations_vec},
                                            {"groups", groups},
                                            {"data_format", data_format}});
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(conv2d_transpose, ops::Conv2DTransposeNPUKernel<float>,
                       ops::Conv2DTransposeNPUKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(conv2d_transpose_grad,
                       ops::Conv2DTransposeGradNPUKernel<float>,
                       ops::Conv2DTransposeGradNPUKernel<plat::float16>);
