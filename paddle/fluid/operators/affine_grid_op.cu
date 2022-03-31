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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/affine_grid_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void LinspaceKernel(T start, T step, int64_t size, T* out) {
  CUDA_KERNEL_LOOP(index, size) { out[index] = start + step * index; }
}

template <typename T>
struct Linspace<paddle::platform::CUDADeviceContext, T> {
  void operator()(T start, T end, int count, bool align_corners,
                  framework::Tensor* numbers,
                  const framework::ExecutionContext& ctx) {
    T* number_data = numbers->mutable_data<T>({count}, ctx.GetPlace());
    T slice = (end - start) / (T)(count - 1);
    if (!align_corners) {
      slice = (end - start) / (T)count;
      start *= (T)(count - 1) / (T)count;
    }
    auto stream = ctx.cuda_device_context().stream();
    int block = 512;
    int grid = (count + block - 1) / block;
    LinspaceKernel<T><<<grid, block, 0, stream>>>(start, slice, count,
                                                  number_data);
  }
};

template <typename T>
__global__ void affine_grid_kernel(const int count, int n, int out_h, int out_w,
                                   T h_start, T w_start, T h_step, T w_step,
                                   const T* theta,  // N, 2, 3
                                   T* output) {
  CUDA_KERNEL_LOOP(index, count) {
    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int n = index / (out_w * out_h);

    T h_coor = h_step * static_cast<T>(h) + static_cast<T>(h_start);
    T w_coor = w_step * static_cast<T>(w) + static_cast<T>(w_start);

    int theta_offset = n * 6;  // 2 * 3;
    // affine from (h_coor, w_coor) to (x, y)
    output[index * 2] = theta[theta_offset] * w_coor +
                        theta[theta_offset + 1] * h_coor +
                        theta[theta_offset + 2];
    output[index * 2 + 1] = theta[theta_offset + 3] * w_coor +
                            theta[theta_offset + 4] * h_coor +
                            theta[theta_offset + 5];
  }
}

template <typename T>
__global__ void affine_grid_grad_kernel(const int count, int n, int out_h,
                                        int out_w, T h_start, T w_start,
                                        T h_step, T w_step,
                                        const T* out_grad,  // N, H, W, 2
                                        T* theta_grad) {    // N, 2, 3
  CUDA_KERNEL_LOOP(index, count) {
    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int n = index / (out_w * out_h);
    T h_coor = h_step * static_cast<T>(h) + static_cast<T>(h_start);
    T w_coor = w_step * static_cast<T>(w) + static_cast<T>(w_start);

    int theta_offset = n * 6;  // 2 * 3;
    T out_grad_x = out_grad[index * 2];
    platform::CudaAtomicAdd(theta_grad + theta_offset, out_grad_x * w_coor);
    platform::CudaAtomicAdd(theta_grad + theta_offset + 1, out_grad_x * h_coor);
    platform::CudaAtomicAdd(theta_grad + theta_offset + 2, out_grad_x);

    T out_grad_y = out_grad[index * 2 + 1];
    platform::CudaAtomicAdd(theta_grad + theta_offset + 3, out_grad_y * w_coor);
    platform::CudaAtomicAdd(theta_grad + theta_offset + 4, out_grad_y * h_coor);
    platform::CudaAtomicAdd(theta_grad + theta_offset + 5, out_grad_y);
  }
}

template <typename T>
class AffineGridOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* theta = ctx.Input<Tensor>("Theta");
    int n = theta->dims()[0];
    auto size_attr = ctx.Attr<std::vector<int>>("output_shape");
    auto align_corners = ctx.Attr<bool>("align_corners");
    int h = 0;
    int w = 0;
    if (size_attr.size() == 0) {
      auto* output_shape = ctx.Input<Tensor>("OutputShape");
      Tensor h_sizes;
      framework::TensorCopy(*output_shape, platform::CPUPlace(), &h_sizes);
      const int* h_size_data = h_sizes.data<int>();
      h = h_size_data[2];
      w = h_size_data[3];
    } else {
      h = size_attr[2];
      w = size_attr[3];
    }
    auto* output = ctx.Output<Tensor>("Output");
    T* out_data = output->mutable_data<T>({n, h, w, 2}, ctx.GetPlace());

    T h_step;
    T w_step;
    T h_start = -1;
    T w_start = -1;
    if (align_corners) {
      h_step = static_cast<T>(2) / static_cast<T>(h - 1);
      w_step = static_cast<T>(2) / static_cast<T>(w - 1);
    } else {
      h_step = static_cast<T>(2) / static_cast<T>(h);
      w_step = static_cast<T>(2) / static_cast<T>(w);

      h_start *= static_cast<T>(h - 1) / static_cast<T>(h);
      w_start *= static_cast<T>(w - 1) / static_cast<T>(w);
    }

    const int count = n * h * w;
    int block = 512;
    int grid = (count + block - 1) / block;
    auto cu_stream = ctx.cuda_device_context().stream();
    affine_grid_kernel<<<grid, block, 0, cu_stream>>>(
        count, n, h, w, h_start, w_start, h_step, w_step,
        theta->data<T>(),  // N, 2, 3
        out_data);
  }
};

template <typename T>
class AffineGridGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto theta_grad = ctx.Output<Tensor>(framework::GradVarName("Theta"));
    int n = output_grad->dims()[0];
    auto size_attr = ctx.Attr<std::vector<int>>("output_shape");
    auto align_corners = ctx.Attr<bool>("align_corners");
    int h = 0;
    int w = 0;
    if (size_attr.size() == 0) {
      auto* output_shape = ctx.Input<Tensor>("OutputShape");
      Tensor h_sizes;
      framework::TensorCopy(*output_shape, platform::CPUPlace(), &h_sizes);
      const int* h_size_data = h_sizes.data<int>();
      h = h_size_data[2];
      w = h_size_data[3];
    } else {
      h = size_attr[2];
      w = size_attr[3];
    }
    T* theta_grad_data = theta_grad->mutable_data<T>({n, 2, 3}, ctx.GetPlace());
    phi::funcs::SetConstant<paddle::platform::CUDADeviceContext, T>()(
        ctx.cuda_device_context(), theta_grad, static_cast<T>(0));

    T h_step;
    T w_step;
    T h_start = -1;
    T w_start = -1;
    if (align_corners) {
      h_step = static_cast<T>(2) / static_cast<T>(h - 1);
      w_step = static_cast<T>(2) / static_cast<T>(w - 1);
    } else {
      h_step = static_cast<T>(2) / static_cast<T>(h);
      w_step = static_cast<T>(2) / static_cast<T>(w);

      h_start *= static_cast<T>(h - 1) / static_cast<T>(h);
      w_start *= static_cast<T>(w - 1) / static_cast<T>(w);
    }
    const int count = n * h * w;
    VLOG(3) << "count: " << count << "; h_step: " << h_step
            << "; w_step: " << w_step << "; h_start: " << h_start
            << "; w_start: " << w_start;
    int block = 512;
    int grid = (count + block - 1) / block;
    auto cu_stream = ctx.cuda_device_context().stream();
    affine_grid_grad_kernel<<<grid, block, 0, cu_stream>>>(
        count, n, h, w, h_start, w_start, h_step, w_step,
        output_grad->data<T>(), theta_grad_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(affine_grid, ops::AffineGridOpCUDAKernel<float>,
                        ops::AffineGridOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(affine_grid_grad,
                        ops::AffineGridGradOpCUDAKernel<float>,
                        ops::AffineGridGradOpCUDAKernel<double>);
