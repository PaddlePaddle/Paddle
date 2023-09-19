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

#pragma once
#include <cstdio>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
#endif
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/top_k_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/funcs/top_k_function_cuda.h"
// set cub base traits in order to handle float16

namespace paddle {
namespace operators {

#define FIXED_BLOCK_DIM_BASE(dim, ...) \
  case (dim): {                        \
    constexpr auto kBlockDim = (dim);  \
    __VA_ARGS__;                       \
  } break

#define FIXED_MAXLENGTH_BASE(MaxLength, ...) \
  case (MaxLength): {                        \
    constexpr auto maxLength = (MaxLength);  \
    __VA_ARGS__;                             \
  } break

#define FIXED_BLOCK_DIM(...)                 \
  FIXED_BLOCK_DIM_BASE(1024, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(512, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);   \
  FIXED_BLOCK_DIM_BASE(32, ##__VA_ARGS__)

#define FIXED_MAXLENGTH(...)              \
  FIXED_MAXLENGTH_BASE(1, ##__VA_ARGS__); \
  FIXED_MAXLENGTH_BASE(2, ##__VA_ARGS__); \
  FIXED_MAXLENGTH_BASE(3, ##__VA_ARGS__); \
  FIXED_MAXLENGTH_BASE(4, ##__VA_ARGS__); \
  FIXED_MAXLENGTH_BASE(5, ##__VA_ARGS__)

template <typename DeviceContext, typename T>
class TopkOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()),
        true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    auto* indices = ctx.Output<phi::DenseTensor>("Indices");
    int k = static_cast<int>(ctx.Attr<int>("k"));

    auto* k_t = ctx.Input<phi::DenseTensor>("K");
    if (k_t) {
      phi::DenseTensor k_host;
      framework::TensorCopySync(*k_t, platform::CPUPlace(), &k_host);
      k = k_host.data<int>()[0];
      framework::DDim output_dims = output->dims();
      output_dims[output_dims.size() - 1] = k;
      output->Resize(output_dims);
      indices->Resize(output_dims);
    }

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    // FIXME(typhoonzero): data is always converted to type T?

    framework::DDim inputdims = input->dims();
    const int64_t input_height =
        phi::product(phi::slice_ddim(inputdims, 0, inputdims.size() - 1));
    const int64_t input_width = inputdims[inputdims.size() - 1];
    const auto& dev_ctx = ctx.cuda_device_context();
    if ((input_width <= 1024 || k >= 128 || k == input_width)) {
      if (phi::funcs::SortTopk<T>(
              dev_ctx, input, input_width, input_height, k, output, indices)) {
        // Successed, return.
        return;
      } else {
        LOG(INFO) << "TopKOP: Some errors happened when use cub sorting, use "
                     "default topk kernel.";
      }
    }
    int64_t* indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());
    if (k > input_width) k = input_width;

    // NOTE: pass lds and dim same to input width.
    // NOTE: old matrix implementation of stride is different to eigen.
    // TODO(typhoonzero): refine this kernel.
    const int kMaxHeight = 2048;
    int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
    phi::backends::gpu::GpuLaunchConfig config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, input_width);
    switch (config.thread_per_block.x) {
      FIXED_BLOCK_DIM(switch (phi::funcs::getMaxLength(k)) {
        FIXED_MAXLENGTH(
            phi::funcs::KeMatrixTopK<T, maxLength, kBlockDim>
            <<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(output_data,
                                                        k,
                                                        indices_data,
                                                        input_data,
                                                        input_width,
                                                        input_width,
                                                        static_cast<int>(k),
                                                        gridx,
                                                        input_height));
        default:
          PADDLE_THROW(platform::errors::Fatal(
              "the input k has error when use getMaxLength function to get the "
              "maxLength."));
      });
      default:
        PADDLE_THROW(platform::errors::Unavailable(
            "Calculation error occurred in TopK Operator's CUDA Kernel."));
    }
  }
};

template <typename DeviceContext, typename T>
class TopkOpGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()),
        true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));
    auto* x = context.Input<phi::DenseTensor>("X");
    auto* out_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<phi::DenseTensor>("Indices");
    auto* x_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));

    T* x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    const T* out_grad_data = out_grad->data<T>();
    const int64_t* indices_data = indices->data<int64_t>();
    size_t k = indices->dims()[indices->dims().size() - 1];

    framework::DDim xdims = x->dims();
    const size_t row =
        phi::product(phi::slice_ddim(xdims, 0, xdims.size() - 1));
    const size_t col = xdims[xdims.size() - 1];
    const auto& dev_ctx = context.cuda_device_context();
    const int kMaxHeight = 2048;
    int gridx = row < kMaxHeight ? row : kMaxHeight;
    switch (phi::funcs::GetDesiredBlockDim(col)) {
      FIXED_BLOCK_DIM(
          phi::funcs::AssignGrad<T, 5, kBlockDim>
          <<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
              x_grad_data, indices_data, out_grad_data, row, col, k));
      default:
        PADDLE_THROW(
            platform::errors::Unavailable("Error occurs when Assign Grad."));
    }
  }
};
#undef FIXED_BLOCK_DIM_BASE
#undef FIXED_BLOCK_DIM

}  // namespace operators
}  // namespace paddle
REGISTER_OP_CUDA_KERNEL(
    top_k,
    paddle::operators::TopkOpCUDAKernel<phi::GPUContext, float>,
    paddle::operators::TopkOpCUDAKernel<phi::GPUContext, double>,
    paddle::operators::TopkOpCUDAKernel<phi::GPUContext, int>,
    paddle::operators::TopkOpCUDAKernel<phi::GPUContext, int64_t>,
    paddle::operators::TopkOpCUDAKernel<phi::GPUContext,
                                        paddle::platform::float16>);

REGISTER_OP_CUDA_KERNEL(
    top_k_grad,
    paddle::operators::TopkOpGradCUDAKernel<phi::GPUContext, float>,
    paddle::operators::TopkOpGradCUDAKernel<phi::GPUContext, double>,
    paddle::operators::TopkOpGradCUDAKernel<phi::GPUContext, int>,
    paddle::operators::TopkOpGradCUDAKernel<phi::GPUContext, int64_t>,
    paddle::operators::TopkOpGradCUDAKernel<phi::GPUContext,
                                            paddle::platform::float16>);
