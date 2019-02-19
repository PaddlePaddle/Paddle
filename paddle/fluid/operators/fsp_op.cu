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
#include "paddle/fluid/operators/fsp_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

inline int DivUp(int x, int y) { return (x + y - 1) / y; }

template <typename T>
__global__ void FSPKernel(const size_t batch_size, const size_t x_channel,
                          const size_t y_channel, const size_t height,
                          const size_t width, const T* x_data, const T* y_data,
                          T* output_data) {
  const int batch_id = blockIdx.x;
  const int h = blockIdx.y * blockDim.x + threadIdx.x;
  const int w = blockIdx.z * blockDim.y + threadIdx.y;

  if (h < x_channel && w < y_channel) {
    const T* x_data_ptr =
        x_data + batch_id * (x_channel * height * width) + h * (height * width);
    const T* y_data_ptr =
        y_data + batch_id * (y_channel * height * width) + w * (height * width);
    T sum = 0;
    size_t count = height * width;
    for (size_t i = 0; i < count; ++i) {
      sum += (x_data_ptr[i] * y_data_ptr[i]);
    }
    output_data[batch_id * x_channel * y_channel + h * y_channel + w] =
        sum / count;
  }
}

template <typename T>
__global__ void FSPGradKernel(const size_t batch_size, const size_t x_channel,
                              const size_t y_channel, const size_t in_height,
                              const size_t in_width, const T* x_data,
                              const T* y_data, const T* d_output_data,
                              T* d_x_data, T* d_y_data) {
  const int batch_id = blockIdx.x;
  const int in_h_id = blockIdx.y * blockDim.x + threadIdx.x;
  const int in_w_id = blockIdx.z * blockDim.y + threadIdx.y;

  if (in_h_id < in_height && in_w_id < in_width) {
    // calculate d_x
    if (d_x_data != nullptr) {
      for (int x_channel_id = threadIdx.z; x_channel_id < x_channel;
           x_channel_id += blockDim.z) {
        int d_x_offset = batch_id * x_channel * in_height * in_width +
                         x_channel_id * in_height * in_width +
                         in_h_id * in_width + in_w_id;
        for (int y_channel_id = 0; y_channel_id < y_channel; ++y_channel_id) {
          T d_out = d_output_data[batch_id * x_channel * y_channel +
                                  x_channel_id * y_channel + y_channel_id];
          d_x_data[d_x_offset] +=
              y_data[batch_id * y_channel * in_height * in_width +
                     y_channel_id * in_height * in_width + in_h_id * in_width +
                     in_w_id] *
              d_out / (in_height * in_width);
        }
      }
    }

    // calculate d_y
    if (d_y_data != nullptr) {
      for (int y_channel_id = threadIdx.z; y_channel_id < y_channel;
           y_channel_id += blockDim.z) {
        int d_y_offset = batch_id * y_channel * in_height * in_width +
                         y_channel_id * in_height * in_width +
                         in_h_id * in_width + in_w_id;
        for (int x_channel_id = 0; x_channel_id < x_channel; ++x_channel_id) {
          T d_out = d_output_data[batch_id * x_channel * y_channel +
                                  x_channel_id * y_channel + y_channel_id];
          d_y_data[d_y_offset] +=
              x_data[batch_id * x_channel * in_height * in_width +
                     x_channel_id * in_height * in_width + in_h_id * in_width +
                     in_w_id] *
              d_out / (in_height * in_width);
        }
      }
    }
  }
}

inline static dim3 GetDesiredBlockDim(int dim) {
  if (dim > 512) {
    return dim3(1, 1, 1024);
  } else if (dim >= 256) {
    return dim3(2, 2, 256);
  } else if (dim >= 64) {
    return dim3(4, 4, 64);
  } else {
    return dim3(8, 8, 16);
  }
}

template <typename T>
class CUDAFSPOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "It must use CUDAPlace.");
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* output = context.Output<Tensor>("Out");
    const T* x_data = x->data<T>();
    const T* y_data = y->data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    const size_t batch_size = x_dims[0];
    const size_t x_channel = x_dims[1];
    const size_t y_channel = y_dims[1];
    const size_t height = x_dims[2];
    const size_t width = x_dims[3];

    auto& dev_context = context.cuda_device_context();
    dim3 block_dim(32, 32);
    dim3 grid_dim(batch_size, DivUp(height, block_dim.x),
                  DivUp(width, block_dim.y));

    FSPKernel<T><<<grid_dim, block_dim, 0, dev_context.stream()>>>(
        batch_size, x_channel, y_channel, height, width, x_data, y_data,
        output_data);
  }
};

template <typename T>
class CUDAFSPGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "It must use CUDAPlace.");

    Tensor* d_x = context.Output<Tensor>(framework::GradVarName("X"));
    Tensor* d_y = context.Output<Tensor>(framework::GradVarName("Y"));
    if ((d_x == nullptr) && (d_y == nullptr)) {
      return;
    }
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_out_data = d_out->data<T>();
    auto d_out_dims = d_out->dims();
    const T* x_data = nullptr;
    const T* y_data = nullptr;
    T* d_x_data = nullptr;
    T* d_y_data = nullptr;
    size_t h = 0;
    size_t w = 0;

    math::SetConstant<platform::CUDADeviceContext, T> set_zero;
    if (d_x) {
      d_x_data = d_x->mutable_data<T>(context.GetPlace());
      set_zero(context.template device_context<platform::CUDADeviceContext>(),
               d_x, static_cast<T>(0));
      auto* y = context.Input<Tensor>("Y");
      y_data = y->data<T>();
      auto d_x_dims = d_x->dims();
      h = d_x_dims[2];
      w = d_x_dims[3];
    }
    if (d_y) {
      d_y_data = d_y->mutable_data<T>(context.GetPlace());
      set_zero(context.template device_context<platform::CUDADeviceContext>(),
               d_y, static_cast<T>(0));
      auto* x = context.Input<Tensor>("X");
      x_data = x->data<T>();
      auto d_y_dims = d_y->dims();
      h = d_y_dims[2];
      w = d_y_dims[3];
    }

    const size_t batch_size = d_out_dims[0];
    const size_t x_channel = d_out_dims[1];
    const size_t y_channel = d_out_dims[2];
    const size_t in_height = h;
    const size_t in_width = w;

    auto& dev_context = context.cuda_device_context();
    //    size_t max_channel = (x_channel>y_channel)?x_channel:y_channel;
    //    auto block_dim = GetDesiredBlockDim(max_channel);
    dim3 block_dim(1, 1, 512);
    LOG(ERROR) << "DivUp(in_height, block_dim.x): "
               << DivUp(in_height, block_dim.x);
    LOG(ERROR) << "DivUp(in_width, block_dim.y): "
               << DivUp(in_width, block_dim.y);
    dim3 grid_dim(batch_size, DivUp(in_height, block_dim.x),
                  DivUp(in_width, block_dim.y));

    FSPGradKernel<T><<<grid_dim, block_dim, 0, dev_context.stream()>>>(
        batch_size, x_channel, y_channel, in_height, in_width, x_data, y_data,
        d_out_data, d_x_data, d_y_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fsp, ops::CUDAFSPOpKernel<float>);
REGISTER_OP_CUDA_KERNEL(fsp_grad, ops::CUDAFSPGradOpKernel<float>);
