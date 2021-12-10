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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/label_smooth_op.h"
namespace paddle {
namespace operators {

template <typename T>
__global__ void LabelSmoothRunOriginKernel(const int N, const float epsilon,
                                           const int label_dim, const T* src,
                                           T* dst) {
  CUDA_KERNEL_LOOP(idx, N) {
    dst[idx] = static_cast<T>(1 - epsilon) * src[idx] +
               static_cast<T>(epsilon / label_dim);
  }
}

template <typename T>
__global__ void LabelSmoothRunDistKernel(const int N, const float epsilon,
                                         const int dist_numel, const T* src,
                                         const T* dist_data, T* dst) {
  CUDA_KERNEL_LOOP(idx, N) {
    int dist_idx = idx % dist_numel;
    dst[idx] = static_cast<T>(1 - epsilon) * src[idx] +
               static_cast<T>(epsilon) * dist_data[dist_idx];
  }
}

template <typename T>
__global__ void LabelSmoothGradRunKernel(const int N, const float epsilon,
                                         const T* src, T* dst) {
  CUDA_KERNEL_LOOP(idx, N) {
    dst[idx] = static_cast<T>(1 - epsilon) * src[idx];
  }
}

template <typename DeviceContext, typename T>
class LabelSmoothGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_t = ctx.Output<framework::LoDTensor>("Out");
    auto* in_t = ctx.Input<framework::LoDTensor>("X");
    auto* dist_t = ctx.Input<framework::Tensor>("PriorDist");
    auto label_dim = in_t->dims()[in_t->dims().size() - 1];
    auto epsilon = ctx.Attr<float>("epsilon");
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    auto size_prob = in_t->numel();
    const T* in_data = in_t->data<T>();
    T* out_data = out_t->mutable_data<T>(ctx.GetPlace());
    int threads = 512;
    int grid = (size_prob + threads - 1) / threads;
    auto stream = ctx.cuda_device_context().stream();
    if (dist_t) {
      auto dist_numel = dist_t->numel();
      const T* dist_data = dist_t->data<T>();
      LabelSmoothRunDistKernel<T><<<grid, threads, 0, stream>>>(
          size_prob, epsilon, dist_numel, in_data, dist_data, out_data);

    } else {
      LabelSmoothRunOriginKernel<T><<<grid, threads, 0, stream>>>(
          size_prob, epsilon, label_dim, in_data, out_data);
    }
  }
};

template <typename DeviceContext, typename T>
class LabelSmoothGradGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_out_t = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_in_t = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    d_in_t->mutable_data<T>(ctx.GetPlace());

    auto epsilon = ctx.Attr<float>("epsilon");
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    const T* in_data = d_out_t->data<T>();
    auto size_prob = d_out_t->numel();
    T* out_data = d_in_t->mutable_data<T>(ctx.GetPlace());
    int threads = 512;
    int grid = (size_prob + threads - 1) / threads;
    auto stream = ctx.cuda_device_context().stream();
    LabelSmoothGradRunKernel<T><<<grid, threads, 0, stream>>>(
        size_prob, epsilon, in_data, out_data);
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    label_smooth,
    ops::LabelSmoothGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LabelSmoothGPUKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    label_smooth_grad,
    ops::LabelSmoothGradGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LabelSmoothGradGPUKernel<paddle::platform::CUDADeviceContext, double>);
