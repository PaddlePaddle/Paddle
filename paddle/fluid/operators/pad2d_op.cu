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

#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

using framework::Tensor;

template <typename T>
__global__ void Pad2DConstNCHW(const int nthreads, const T* in_data,
                               const int num, const int channels,
                               const int in_height, const int in_width,
                               const int out_height, const int out_width,
                               const int pad_top, const int pad_left, T value,
                               T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int nc = index / out_width;
    const int out_w = index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    int in_h = out_h - pad_top;
    int in_w = out_w - pad_left;
    out_data[index] =
        (in_h < 0 || in_w < 0 || in_h >= in_height || in_w >= in_width)
            ? value
            : in_data[(nc * in_height + in_h) * in_width + in_w];
  }
}

template <typename T>
__global__ void Pad2DConstNHWC(const int nthreads, const T* in_data,
                               const int num, const int channels,
                               const int in_height, const int in_width,
                               const int out_height, const int out_width,
                               const int pad_top, const int pad_left, T value,
                               T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    const int in_h = out_h - pad_top;
    const int in_w = out_w - pad_left;
    out_data[index] =
        (in_h < 0 || in_w < 0 || in_h >= in_height || in_w >= in_width)
            ? value
            : in_data[((n * in_height + in_h) * in_width + in_w) * channels +
                      c];
  }
}

template <typename T>
__global__ void Pad2DReflectNCHW(const int nthreads, const T* in_data,
                                 const int num, const int channels,
                                 const int in_height, const int in_width,
                                 const int out_height, const int out_width,
                                 const int pad_top, const int pad_left,
                                 T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int nc = index / out_width;
    const int out_w = index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    int in_h = out_h - pad_top;
    int in_w = out_w - pad_left;
    in_h = max(in_h, -in_h);                     // reflect by 0
    in_h = min(in_h, 2 * in_height - in_h - 2);  // reflect by in_height
    in_w = max(in_w, -in_w);                     // reflect by 0
    in_w = min(in_w, 2 * in_width - in_w - 2);   // reflect by in_width
    out_data[index] = in_data[(nc * in_height + in_h) * in_width + in_w];
  }
}

template <typename T>
__global__ void Pad2DReflectNHWC(const int nthreads, const T* in_data,
                                 const int num, const int channels,
                                 const int in_height, const int in_width,
                                 const int out_height, const int out_width,
                                 const int pad_top, const int pad_left,
                                 T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    int in_h = out_h - pad_top;
    int in_w = out_w - pad_left;
    in_h = max(in_h, -in_h);
    in_h = min(in_h, 2 * in_height - in_h - 2);
    in_w = max(in_w, -in_w);
    in_w = min(in_w, 2 * in_width - in_w - 2);
    out_data[index] =
        in_data[((n * in_height + in_h) * in_width + in_w) * channels + c];
  }
}

template <typename T>
__global__ void Pad2DEdgeNCHW(const int nthreads, const T* in_data,
                              const int num, const int channels,
                              const int in_height, const int in_width,
                              const int out_height, const int out_width,
                              const int pad_top, const int pad_left,
                              T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int nc = index / out_width;
    const int out_w = index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    int in_h = min(in_height - 1, max(out_h - pad_top, 0));
    int in_w = min(in_width - 1, max(out_w - pad_left, 0));
    out_data[index] = in_data[(nc * in_height + in_h) * in_width + in_w];
  }
}

template <typename T>
__global__ void Pad2DEdgeNHWC(const int nthreads, const T* in_data,
                              const int num, const int channels,
                              const int in_height, const int in_width,
                              const int out_height, const int out_width,
                              const int pad_top, const int pad_left,
                              T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    int in_h = min(in_height - 1, max(out_h - pad_top, 0));
    int in_w = min(in_width - 1, max(out_w - pad_left, 0));
    out_data[index] =
        in_data[((n * in_height + in_h) * in_width + in_w) * channels + c];
  }
}

template <typename T>
__global__ void Pad2DGradConstNCHW(const int in_size, T* d_in_data,
                                   const int num, const int channels,
                                   const int in_height, const int in_width,
                                   const int out_height, const int out_width,
                                   const int pad_top, const int pad_left,
                                   const T* d_out_data) {
  CUDA_KERNEL_LOOP(in_index, in_size) {
    int nc = in_index / in_width;
    const int out_w = in_index % in_width + pad_left;
    const int out_h = nc % in_height + pad_top;
    nc /= in_height;
    d_in_data[in_index] =
        d_out_data[(nc * out_height + out_h) * out_width + out_w];
  }
}

template <typename T>
__global__ void Pad2DGradConstNHWC(const int in_size, T* d_in_data,
                                   const int num, const int channels,
                                   const int in_height, const int in_width,
                                   const int out_height, const int out_width,
                                   const int pad_top, const int pad_left,
                                   const T* d_out_data) {
  CUDA_KERNEL_LOOP(in_index, in_size) {
    int n = in_index / channels;
    const int c = in_index % channels;
    const int out_w = n % in_width + pad_left;
    n /= in_width;
    const int out_h = n % in_height + pad_top;
    n /= in_height;
    d_in_data[in_index] =
        d_out_data[((n * out_height + out_h) * out_width + out_w) * channels +
                   c];
  }
}

template <typename T>
__global__ void Pad2DGradReflectNCHW(const int out_size, T* d_in_data,
                                     const int num, const int channels,
                                     const int in_height, const int in_width,
                                     const int out_height, const int out_width,
                                     const int pad_top, const int pad_left,
                                     const T* d_out_data) {
  CUDA_KERNEL_LOOP(out_index, out_size) {
    int nc = out_index / out_width;
    const int out_w = out_index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    int in_h = out_h - pad_top;
    int in_w = out_w - pad_left;
    in_h = max(in_h, -in_h);
    in_w = max(in_w, -in_w);
    in_h = min(in_h, 2 * in_height - in_h - 2);
    in_w = min(in_w, 2 * in_width - in_w - 2);
    platform::CudaAtomicAdd(
        &d_in_data[(nc * in_height + in_h) * in_width + in_w],
        d_out_data[out_index]);
  }
}

template <typename T>
__global__ void Pad2DGradReflectNHWC(const int out_size, T* d_in_data,
                                     const int num, const int channels,
                                     const int in_height, const int in_width,
                                     const int out_height, const int out_width,
                                     const int pad_top, const int pad_left,
                                     const T* d_out_data) {
  CUDA_KERNEL_LOOP(out_index, out_size) {
    const int c = out_index % channels;
    int n = out_index / channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    int in_h = out_h - pad_top;
    int in_w = out_w - pad_left;
    in_h = max(in_h, -in_h);
    in_w = max(in_w, -in_w);
    in_h = min(in_h, in_height * 2 - in_h - 2);
    in_w = min(in_w, in_width * 2 - in_w - 2);
    platform::CudaAtomicAdd(
        &d_in_data[((n * in_height + in_h) * in_width + in_w) * channels + c],
        d_out_data[out_index]);
  }
}

template <typename T>
__global__ void Pad2DGradEdgeNCHW(const int out_size, T* d_in_data,
                                  const int num, const int channels,
                                  const int in_height, const int in_width,
                                  const int out_height, const int out_width,
                                  const int pad_top, const int pad_left,
                                  const T* d_out_data) {
  CUDA_KERNEL_LOOP(out_index, out_size) {
    int nc = out_index / out_width;
    const int out_w = out_index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    const int in_h = min(in_height - 1, max(out_h - pad_top, 0));
    const int in_w = min(in_width - 1, max(out_w - pad_left, 0));
    platform::CudaAtomicAdd(
        &d_in_data[(nc * in_height + in_h) * in_width + in_w],
        d_out_data[out_index]);
  }
}

template <typename T>
__global__ void Pad2DGradEdgeNHWC(const int out_size, T* d_in_data,
                                  const int num, const int channels,
                                  const int in_height, const int in_width,
                                  const int out_height, const int out_width,
                                  const int pad_top, const int pad_left,
                                  const T* d_out_data) {
  CUDA_KERNEL_LOOP(out_index, out_size) {
    const int c = out_index % channels;
    int n = out_index / channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    const int in_h = min(in_height - 1, max(out_h - pad_top, 0));
    const int in_w = min(in_width - 1, max(out_w - pad_left, 0));
    platform::CudaAtomicAdd(
        &d_in_data[((n * in_height + in_h) * in_width + in_w) * channels + c],
        d_out_data[out_index]);
  }
}

static inline void GetPaddings(int* paddings,
                               const framework::ExecutionContext& context) {
  auto* paddings_t = context.Input<Tensor>("Paddings");
  if (paddings_t) {
    Tensor pads;
    framework::TensorCopySync(*paddings_t, platform::CPUPlace(), &pads);
    auto pads_data = pads.data<int>();
    paddings[0] = pads_data[0];
    paddings[1] = pads_data[1];
    paddings[2] = pads_data[2];
    paddings[3] = pads_data[3];
  } else {
    auto pads = context.Attr<std::vector<int>>("paddings");
    std::copy(pads.begin(), pads.end(), paddings);
  }
}

template <typename T>
class Pad2dCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    int pads[4];
    GetPaddings(pads, context);
    auto mode = context.Attr<std::string>("mode");
    auto data_format = context.Attr<std::string>("data_format");
    T value = static_cast<T>(context.Attr<float>("pad_value"));

    auto* x = context.Input<Tensor>("X");
    auto in_dims = x->dims();
    const T* in_data = x->data<T>();
    auto* out = context.Output<Tensor>("Out");
    auto out_dims = out->dims();
    if (data_format == "NCHW") {
      out_dims[0] = in_dims[0];
      out_dims[1] = in_dims[1];
      out_dims[2] = in_dims[2] + pads[0] + pads[1];
      out_dims[3] = in_dims[3] + pads[2] + pads[3];
    } else {
      out_dims[0] = in_dims[0];
      out_dims[1] = in_dims[1] + pads[0] + pads[1];
      out_dims[2] = in_dims[2] + pads[2] + pads[3];
      out_dims[3] = in_dims[3];
    }
    T* out_data = out->mutable_data<T>(out_dims, context.GetPlace());
    const int pad_top = pads[0];
    const int pad_left = pads[2];
    const int num = in_dims[0];

    auto stream = context.cuda_device_context().stream();
    int block = PADDLE_CUDA_NUM_THREADS;
    const int out_size = out->numel();
    int grid = (out_size + block - 1) / block;

    if (data_format == "NCHW") {
      const int channels = in_dims[1];
      const int in_height = in_dims[2];
      const int in_width = in_dims[3];
      const int out_height = out_dims[2];
      const int out_width = out_dims[3];
      if (mode == "reflect") {
        Pad2DReflectNCHW<T><<<grid, block, 0, stream>>>(
            out_size, in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, out_data);
      } else if (mode == "edge") {
        Pad2DEdgeNCHW<T><<<grid, block, 0, stream>>>(
            out_size, in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, out_data);
      } else {
        Pad2DConstNCHW<T><<<grid, block, 0, stream>>>(
            out_size, in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, value, out_data);
      }
    } else {
      const int channels = in_dims[3];
      const int in_height = in_dims[1];
      const int in_width = in_dims[2];
      const int out_height = out_dims[1];
      const int out_width = out_dims[2];
      if (mode == "reflect") {
        Pad2DReflectNHWC<T><<<grid, block, 0, stream>>>(
            out_size, in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, out_data);
      } else if (mode == "edge") {
        Pad2DEdgeNHWC<T><<<grid, block, 0, stream>>>(
            out_size, in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, out_data);
      } else {
        Pad2DConstNHWC<T><<<grid, block, 0, stream>>>(
            out_size, in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, value, out_data);
      }
    }
  }
};

template <typename T>
class Pad2dGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    int pads[4];
    GetPaddings(pads, context);
    auto mode = context.Attr<std::string>("mode");
    auto data_format = context.Attr<std::string>("data_format");
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_in = context.Output<Tensor>(framework::GradVarName("X"));
    auto d_in_dims = d_in->dims();
    auto d_out_dims = d_out->dims();
    const T* d_out_data = d_out->data<T>();
    T* d_in_data = d_in->mutable_data<T>(context.GetPlace());

    math::SetConstant<platform::CUDADeviceContext, T> set_zero;
    set_zero(context.template device_context<platform::CUDADeviceContext>(),
             d_in, static_cast<T>(0));

    const int pad_top = pads[0];
    const int pad_left = pads[2];
    const int num = d_in_dims[0];

    auto stream = context.cuda_device_context().stream();
    int block = PADDLE_CUDA_NUM_THREADS;
    const int out_size = d_out->numel();
    const int in_size = d_in->numel();
    int grid = (out_size + block - 1) / block;

    if (data_format == "NCHW") {
      const int channels = d_in_dims[1];
      const int in_height = d_in_dims[2];
      const int in_width = d_in_dims[3];
      const int out_height = d_out_dims[2];
      const int out_width = d_out_dims[3];
      if (mode == "reflect") {
        Pad2DGradReflectNCHW<T><<<grid, block, 0, stream>>>(
            out_size, d_in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, d_out_data);
      } else if (mode == "edge") {
        Pad2DGradEdgeNCHW<T><<<grid, block, 0, stream>>>(
            out_size, d_in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, d_out_data);
      } else {
        grid = (in_size + block - 1) / block;
        Pad2DGradConstNCHW<T><<<grid, block, 0, stream>>>(
            in_size, d_in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, d_out_data);
      }
    } else {
      const int channels = d_in_dims[3];
      const int in_height = d_in_dims[1];
      const int in_width = d_in_dims[2];
      const int out_height = d_out_dims[1];
      const int out_width = d_out_dims[2];
      if (mode == "reflect") {
        Pad2DGradReflectNHWC<T><<<grid, block, 0, stream>>>(
            out_size, d_in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, d_out_data);
      } else if (mode == "edge") {
        Pad2DGradEdgeNHWC<T><<<grid, block, 0, stream>>>(
            out_size, d_in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, d_out_data);
      } else {
        grid = (in_size + block - 1) / block;
        Pad2DGradConstNHWC<T><<<grid, block, 0, stream>>>(
            in_size, d_in_data, num, channels, in_height, in_width, out_height,
            out_width, pad_top, pad_left, d_out_data);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(pad2d, ops::Pad2dCUDAKernel<plat::float16>,
                        ops::Pad2dCUDAKernel<float>,
                        ops::Pad2dCUDAKernel<double>, ops::Pad2dCUDAKernel<int>,
                        ops::Pad2dCUDAKernel<int64_t>);
REGISTER_OP_CUDA_KERNEL(pad2d_grad, ops::Pad2dGradCUDAKernel<plat::float16>,
                        ops::Pad2dGradCUDAKernel<float>,
                        ops::Pad2dGradCUDAKernel<double>);
