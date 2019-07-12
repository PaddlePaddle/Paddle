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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedPoolingDescriptor = platform::ScopedPoolingDescriptor;
using DataLayout = platform::DataLayout;
using PoolingMode = platform::PoolingMode;
template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;

template <typename T>
class PoolCUDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");

    const Tensor *input = ctx.Input<Tensor>("X");
    Tensor *output = ctx.Output<Tensor>("Out");

    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>(ctx.GetPlace());

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    bool exclusive = ctx.Attr<bool>("exclusive");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    if (ctx.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(input->dims()[i + 2]);
      }
    }

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedPoolingDescriptor pool_desc;
    DataLayout layout;

    if (strides.size() == 2U) {
      layout = DataLayout::kNCHW;
    } else {
      layout = DataLayout::kNCDHW;
    }

    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()));
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output->dims()));

    PoolingMode pooling_mode;
    if (pooling_type == "max") {
      pooling_mode = PoolingMode::kMaximum;
    } else {
      pooling_mode = exclusive ? PoolingMode::kAverageExclusive
                               : PoolingMode::kAverageInclusive;
    }

    cudnnPoolingDescriptor_t cudnn_pool_desc =
        pool_desc.descriptor(pooling_mode, ksize, paddings, strides);

    // ------------------- cudnn pool algorithm ---------------------
    auto handle = ctx.cuda_device_context().cudnn_handle();
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    CUDNN_ENFORCE(platform::dynload::cudnnPoolingForward(
        handle, cudnn_pool_desc, &alpha, cudnn_input_desc, input_data, &beta,
        cudnn_output_desc, output_data));
  }
};

template <typename T>
class PoolCUDNNGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");

    const Tensor *input = ctx.Input<Tensor>("X");
    const Tensor *output = ctx.Input<Tensor>("Out");
    const Tensor *output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    bool exclusive = ctx.Attr<bool>("exclusive");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    if (ctx.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(input->dims()[i + 2]);
      }
    }

    const T *input_data = input->data<T>();
    const T *output_data = output->data<T>();
    const T *output_grad_data = output_grad->data<T>();

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedPoolingDescriptor pool_desc;
    DataLayout layout;

    if (strides.size() == 2U) {
      layout = DataLayout::kNCHW;
    } else {
      layout = DataLayout::kNCDHW;
    }

    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()));
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output->dims()));

    PoolingMode pooling_mode;
    if (pooling_type == "max") {
      if (FLAGS_cudnn_deterministic) {
        pooling_mode = PoolingMode::kMaximumDeterministic;
      } else {
        pooling_mode = PoolingMode::kMaximum;
      }
    } else {
      pooling_mode = exclusive ? PoolingMode::kAverageExclusive
                               : PoolingMode::kAverageInclusive;
    }

    cudnnPoolingDescriptor_t cudnn_pool_desc =
        pool_desc.descriptor(pooling_mode, ksize, paddings, strides);

    // ------------------- cudnn pool algorithm ---------------------
    auto handle = ctx.cuda_device_context().cudnn_handle();
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    if (input_grad) {
      T *input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset input_grad.

      CUDNN_ENFORCE(platform::dynload::cudnnPoolingBackward(
          handle, cudnn_pool_desc, &alpha, cudnn_output_desc, output_data,
          cudnn_output_desc, output_grad_data, cudnn_input_desc, input_data,
          &beta, cudnn_input_desc, input_grad_data));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_KERNEL(pool2d, CUDNN, plat::CUDAPlace,
                   ops::PoolCUDNNOpKernel<float>,
                   ops::PoolCUDNNOpKernel<double>,
                   ops::PoolCUDNNOpKernel<plat::float16>);
REGISTER_OP_KERNEL(pool2d_grad, CUDNN, plat::CUDAPlace,
                   ops::PoolCUDNNGradOpKernel<float>,
                   ops::PoolCUDNNGradOpKernel<double>,
                   ops::PoolCUDNNGradOpKernel<plat::float16>);

REGISTER_OP_KERNEL(pool3d, CUDNN, plat::CUDAPlace,
                   ops::PoolCUDNNOpKernel<float>,
                   ops::PoolCUDNNOpKernel<double>,
                   ops::PoolCUDNNOpKernel<plat::float16>);
REGISTER_OP_KERNEL(pool3d_grad, CUDNN, plat::CUDAPlace,
                   ops::PoolCUDNNGradOpKernel<float>,
                   ops::PoolCUDNNGradOpKernel<double>);
