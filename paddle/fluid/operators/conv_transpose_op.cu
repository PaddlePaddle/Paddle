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

#include "paddle/fluid/operators/conv_transpose_op.h"
#include "paddle/phi/kernels/gpu/depthwise_conv.h"

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class DepthwiseConvTransposeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const std::string data_layout_str =
        context.Attr<std::string>("data_format");
    const framework::DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const Tensor* input = context.Input<Tensor>("Input");
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    int groups = context.Attr<int>("groups");
    PADDLE_ENFORCE_EQ(
        groups, filter.dims()[0],
        platform::errors::InvalidArgument(
            "groups should be error to the 1st dimension of filter. But "
            "received groups is %d and filter dimension[0] is %d",
            groups, filter.dims()[0]));

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    for (auto v : dilations) {
      PADDLE_ENFORCE_EQ(v, 1, platform::errors::InvalidArgument(
                                  "dilations should be 1 in depthwise conv. "
                                  "But received dilations is %d",
                                  v));
    }

    auto in_dims = input->dims();
    auto filter_dims = filter.dims();

    framework::DDim in_data_dims;
    if (data_layout != framework::DataLayout::kNHWC) {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    }
    framework::DDim filter_data_dims =
        phi::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    output->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    phi::funcs::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, output, static_cast<T>(0));

    math::DepthwiseConvInputGradFunctor<phi::GPUContext, T>
        depthwiseConvInputGrad;
    depthwiseConvInputGrad(
        static_cast<const typename framework::ConvertToPhiContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *output, filter, *input, strides,
        std::vector<int>{paddings[0], paddings[2], paddings[1], paddings[3]},
        dilations, output, data_layout);
  }
};

template <typename DeviceContext, typename T>
class DepthwiseConvTransposeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const std::string data_layout_str =
        context.Attr<std::string>("data_format");
    const framework::DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));
    Tensor filter = *context.Input<Tensor>("Filter");

    if (!input_grad && !filter_grad) return;

    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    auto in_dims = input->dims();
    auto filter_dims = filter.dims();

    framework::DDim in_data_dims;
    if (data_layout != framework::DataLayout::kNHWC) {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    }
    framework::DDim filter_data_dims =
        phi::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    if (input_grad) {
      math::DepthwiseConvFunctor<phi::GPUContext, T> depthwiseConv;
      depthwiseConv(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *output_grad, filter, strides,
          std::vector<int>{paddings[0], paddings[2], paddings[1], paddings[3]},
          dilations, input_grad, data_layout);
    }

    if (filter_grad) {
      phi::funcs::SetConstant<DeviceContext, T> set_zero;
      filter_grad->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, filter_grad, static_cast<T>(0));

      math::DepthwiseConvFilterGradFunctor<phi::GPUContext, T>
          depthwiseConvFilterGrad;
      depthwiseConvFilterGrad(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *output_grad, *input, strides,
          std::vector<int>{paddings[0], paddings[2], paddings[1], paddings[3]},
          dilations, filter_grad, data_layout);
    }
  }
};

}  // namespace operators
}  // namespace paddle
// conv2d
REGISTER_OP_CUDA_KERNEL(conv2d_transpose,
                        ops::GemmConvTransposeKernel<CUDA, float>,
                        ops::GemmConvTransposeKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(conv2d_transpose_grad,
                        ops::GemmConvTransposeGradKernel<CUDA, float>,
                        ops::GemmConvTransposeGradKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(conv2d_transpose_grad_grad,
                        ops::GemmConvTransposeGradKernel<CUDA, float>,
                        ops::GemmConvTransposeGradKernel<CUDA, double>);

// conv3d
REGISTER_OP_CUDA_KERNEL(conv3d_transpose,
                        ops::GemmConvTransposeKernel<CUDA, float>,
                        ops::GemmConvTransposeKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(conv3d_transpose_grad,
                        ops::GemmConvTransposeGradKernel<CUDA, float>,
                        ops::GemmConvTransposeGradKernel<CUDA, double>);

// depthwise conv2d
REGISTER_OP_CUDA_KERNEL(depthwise_conv2d_transpose,
                        ops::DepthwiseConvTransposeKernel<CUDA, float>,
                        ops::DepthwiseConvTransposeKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(depthwise_conv2d_transpose_grad,
                        ops::DepthwiseConvTransposeGradKernel<CUDA, float>,
                        ops::DepthwiseConvTransposeGradKernel<CUDA, double>);
