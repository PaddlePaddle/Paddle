// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/pad3d_grad_kernel.h"

#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void Pad3DGradConstNCDHW(const int in_size,
                                    T* d_in_data,
                                    const int num,
                                    const int channels,
                                    const int in_depth,
                                    const int in_height,
                                    const int in_width,
                                    const int out_depth,
                                    const int out_height,
                                    const int out_width,
                                    const int pad_front,
                                    const int pad_top,
                                    const int pad_left,
                                    const T* d_out_data) {
  CUDA_KERNEL_LOOP(in_index, in_size) {
    const int in_w = in_index % in_width;

    int nc = in_index / in_width;
    const int in_h = nc % in_height;

    nc /= in_height;
    const int in_d = nc % in_depth;

    nc /= in_depth;

    const int out_d = in_d + pad_front;
    const int out_h = in_h + pad_top;
    const int out_w = in_w + pad_left;
    d_in_data[in_index] =
        d_out_data[nc * out_depth * out_height * out_width +
                   out_d * out_height * out_width + out_h * out_width + out_w];
  }
}

template <typename T>
__global__ void Pad3DGradConstNDHWC(const int in_size,
                                    T* d_in_data,
                                    const int num,
                                    const int channels,
                                    const int in_depth,
                                    const int in_height,
                                    const int in_width,
                                    const int out_depth,
                                    const int out_height,
                                    const int out_width,
                                    const int pad_front,
                                    const int pad_top,
                                    const int pad_left,
                                    const T* d_out_data) {
  CUDA_KERNEL_LOOP(in_index, in_size) {
    const int c = in_index % channels;
    int n = in_index / channels;

    const int in_w = n % in_width;
    n /= in_width;

    const int in_h = n % in_height;
    n /= in_height;

    const int in_d = n % in_depth;
    n /= in_depth;

    const int out_d = in_d + pad_front;
    const int out_h = in_h + pad_top;
    const int out_w = in_w + pad_left;

    d_in_data[in_index] =
        d_out_data[n * out_depth * out_height * out_width * channels +
                   out_d * out_height * out_width * channels +
                   out_h * out_width * channels + out_w * channels + c];
  }
}

template <typename T>
__global__ void Pad3DGradReflectNCDHW(const int out_size,
                                      T* d_in_data,
                                      const int num,
                                      const int channels,
                                      const int in_depth,
                                      const int in_height,
                                      const int in_width,
                                      const int out_depth,
                                      const int out_height,
                                      const int out_width,
                                      const int pad_front,
                                      const int pad_top,
                                      const int pad_left,
                                      const T* d_out_data) {
  CUDA_KERNEL_LOOP(out_index, out_size) {
    int nc = out_index / out_width;
    const int out_w = out_index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    const int out_d = nc % out_depth;
    nc /= out_depth;

    int in_d = out_d - pad_front;
    int in_h = out_h - pad_top;
    int in_w = out_w - pad_left;

    in_d = max(in_d, -in_d);
    in_h = max(in_h, -in_h);
    in_w = max(in_w, -in_w);

    in_d = min(in_d, 2 * in_depth - in_d - 2);
    in_h = min(in_h, 2 * in_height - in_h - 2);
    in_w = min(in_w, 2 * in_width - in_w - 2);

    phi::CudaAtomicAdd(
        &d_in_data[nc * in_depth * in_height * in_width +
                   in_d * in_height * in_width + in_h * in_width + in_w],
        d_out_data[out_index]);
  }
}

template <typename T>
__global__ void Pad3DGradReflectNDHWC(const int out_size,
                                      T* d_in_data,
                                      const int num,
                                      const int channels,
                                      const int in_depth,
                                      const int in_height,
                                      const int in_width,
                                      const int out_depth,
                                      const int out_height,
                                      const int out_width,
                                      const int pad_front,
                                      const int pad_top,
                                      const int pad_left,
                                      const T* d_out_data) {
  CUDA_KERNEL_LOOP(out_index, out_size) {
    const int c = out_index % channels;
    int n = out_index / channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    const int out_d = n % out_depth;
    n /= out_depth;

    int in_d = out_d - pad_front;
    int in_h = out_h - pad_top;
    int in_w = out_w - pad_left;

    in_d = max(in_d, -in_d);
    in_h = max(in_h, -in_h);
    in_w = max(in_w, -in_w);

    in_d = min(in_d, in_depth * 2 - in_d - 2);
    in_h = min(in_h, in_height * 2 - in_h - 2);
    in_w = min(in_w, in_width * 2 - in_w - 2);
    phi::CudaAtomicAdd(
        &d_in_data[n * in_depth * in_height * in_width * channels +
                   in_d * in_height * in_width * channels +
                   in_h * in_width * channels + in_w * channels + c],
        d_out_data[out_index]);
  }
}

template <typename T>
__global__ void Pad3DGradReplicateNCDHW(const int out_size,
                                        T* d_in_data,
                                        const int num,
                                        const int channels,
                                        const int in_depth,
                                        const int in_height,
                                        const int in_width,
                                        const int out_depth,
                                        const int out_height,
                                        const int out_width,
                                        const int pad_front,
                                        const int pad_top,
                                        const int pad_left,
                                        const T* d_out_data) {
  CUDA_KERNEL_LOOP(out_index, out_size) {
    int nc = out_index / out_width;
    const int out_w = out_index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    const int out_d = nc % out_depth;
    nc /= out_depth;

    const int in_d = min(in_depth - 1, max(out_d - pad_front, 0));
    const int in_h = min(in_height - 1, max(out_h - pad_top, 0));
    const int in_w = min(in_width - 1, max(out_w - pad_left, 0));

    phi::CudaAtomicAdd(
        &d_in_data[nc * in_depth * in_height * in_width +
                   in_d * in_height * in_width + in_h * in_width + in_w],
        d_out_data[out_index]);
  }
}

template <typename T>
__global__ void Pad3DGradReplicateNDHWC(const int out_size,
                                        T* d_in_data,
                                        const int num,
                                        const int channels,
                                        const int in_depth,
                                        const int in_height,
                                        const int in_width,
                                        const int out_depth,
                                        const int out_height,
                                        const int out_width,
                                        const int pad_front,
                                        const int pad_top,
                                        const int pad_left,
                                        const T* d_out_data) {
  CUDA_KERNEL_LOOP(out_index, out_size) {
    const int c = out_index % channels;
    int n = out_index / channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    const int out_d = n % out_depth;
    n /= out_depth;

    const int in_d = min(in_depth - 1, max(out_d - pad_front, 0));
    const int in_h = min(in_height - 1, max(out_h - pad_top, 0));
    const int in_w = min(in_width - 1, max(out_w - pad_left, 0));

    phi::CudaAtomicAdd(
        &d_in_data[n * in_depth * in_height * in_width * channels +
                   in_d * in_height * in_width * channels +
                   in_h * in_width * channels + in_w * channels + c],
        d_out_data[out_index]);
  }
}

template <typename T>
__global__ void Pad3DGradCircularNCDHW(const int out_size,
                                       T* d_in_data,
                                       const int num,
                                       const int channels,
                                       const int in_depth,
                                       const int in_height,
                                       const int in_width,
                                       const int out_depth,
                                       const int out_height,
                                       const int out_width,
                                       const int pad_front,
                                       const int pad_top,
                                       const int pad_left,
                                       const T* d_out_data) {
  CUDA_KERNEL_LOOP(out_index, out_size) {
    int nc = out_index / out_width;
    const int out_w = out_index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    const int out_d = nc % out_depth;
    nc /= out_depth;

    int in_d = ((out_d - pad_front) % in_depth + in_depth) % in_depth;
    int in_h = ((out_h - pad_top) % in_height + in_height) % in_height;
    int in_w = ((out_w - pad_left) % in_width + in_width) % in_width;

    phi::CudaAtomicAdd(
        &d_in_data[nc * in_depth * in_height * in_width +
                   in_d * in_height * in_width + in_h * in_width + in_w],
        d_out_data[out_index]);
  }
}

template <typename T>
__global__ void Pad3DGradCircularNDHWC(const int out_size,
                                       T* d_in_data,
                                       const int num,
                                       const int channels,
                                       const int in_depth,
                                       const int in_height,
                                       const int in_width,
                                       const int out_depth,
                                       const int out_height,
                                       const int out_width,
                                       const int pad_front,
                                       const int pad_top,
                                       const int pad_left,
                                       const T* d_out_data) {
  CUDA_KERNEL_LOOP(out_index, out_size) {
    const int c = out_index % channels;
    int n = out_index / channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    const int out_d = n % out_depth;
    n /= out_depth;

    int in_d = ((out_d - pad_front) % in_depth + in_depth) % in_depth;
    int in_h = ((out_h - pad_top) % in_height + in_height) % in_height;
    int in_w = ((out_w - pad_left) % in_width + in_width) % in_width;

    phi::CudaAtomicAdd(
        &d_in_data[n * in_depth * in_height * in_width * channels +
                   in_d * in_height * in_width * channels +
                   in_h * in_width * channels + in_w * channels + c],
        d_out_data[out_index]);
  }
}

template <typename T, typename Context>
void Pad3dGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out_grad,
                     const IntArray& paddings,
                     const std::string& mode,
                     float pad_value,
                     const std::string& data_format,
                     DenseTensor* x_grad) {
  std::vector<int64_t> pads = paddings.GetData();
  auto* d_out = &out_grad;
  auto* d_in = x_grad;
  auto d_in_dims = d_in->dims();
  auto d_out_dims = d_out->dims();
  const T* d_out_data = d_out->data<T>();
  T* d_in_data = dev_ctx.template Alloc<T>(d_in);

  phi::funcs::SetConstant<Context, T>()(dev_ctx, d_in, static_cast<T>(0));

  const int pad_left = pads[0];
  const int pad_top = pads[2];
  const int pad_front = pads[4];

  const int num = d_in_dims[0];

  auto stream = dev_ctx.stream();
  int block = PADDLE_CUDA_NUM_THREADS;
  const int out_size = d_out->numel();
  const int in_size = d_in->numel();
  int grid = (out_size + block - 1) / block;

  if (data_format == "NCDHW") {
    const int channels = d_in_dims[1];
    const int in_depth = d_in_dims[2];
    const int in_height = d_in_dims[3];
    const int in_width = d_in_dims[4];
    const int out_depth = d_out_dims[2];
    const int out_height = d_out_dims[3];
    const int out_width = d_out_dims[4];

    if (mode == "reflect") {
      Pad3DGradReflectNCDHW<T><<<grid, block, 0, stream>>>(out_size,
                                                           d_in_data,
                                                           num,
                                                           channels,
                                                           in_depth,
                                                           in_height,
                                                           in_width,
                                                           out_depth,
                                                           out_height,
                                                           out_width,
                                                           pad_front,
                                                           pad_top,
                                                           pad_left,
                                                           d_out_data);
    } else if (mode == "replicate") {
      Pad3DGradReplicateNCDHW<T><<<grid, block, 0, stream>>>(out_size,
                                                             d_in_data,
                                                             num,
                                                             channels,
                                                             in_depth,
                                                             in_height,
                                                             in_width,
                                                             out_depth,
                                                             out_height,
                                                             out_width,
                                                             pad_front,
                                                             pad_top,
                                                             pad_left,
                                                             d_out_data);
    } else if (mode == "circular") {
      Pad3DGradCircularNCDHW<T><<<grid, block, 0, stream>>>(out_size,
                                                            d_in_data,
                                                            num,
                                                            channels,
                                                            in_depth,
                                                            in_height,
                                                            in_width,
                                                            out_depth,
                                                            out_height,
                                                            out_width,
                                                            pad_front,
                                                            pad_top,
                                                            pad_left,
                                                            d_out_data);
    } else {
      grid = (in_size + block - 1) / block;
      Pad3DGradConstNCDHW<T><<<grid, block, 0, stream>>>(in_size,
                                                         d_in_data,
                                                         num,
                                                         channels,
                                                         in_depth,
                                                         in_height,
                                                         in_width,
                                                         out_depth,
                                                         out_height,
                                                         out_width,
                                                         pad_front,
                                                         pad_top,
                                                         pad_left,
                                                         d_out_data);
    }
  } else {
    const int channels = d_in_dims[4];
    const int in_depth = d_in_dims[1];
    const int in_height = d_in_dims[2];
    const int in_width = d_in_dims[3];
    const int out_depth = d_out_dims[1];
    const int out_height = d_out_dims[2];
    const int out_width = d_out_dims[3];
    if (mode == "reflect") {
      Pad3DGradReflectNDHWC<T><<<grid, block, 0, stream>>>(out_size,
                                                           d_in_data,
                                                           num,
                                                           channels,
                                                           in_depth,
                                                           in_height,
                                                           in_width,
                                                           out_depth,
                                                           out_height,
                                                           out_width,
                                                           pad_front,
                                                           pad_top,
                                                           pad_left,
                                                           d_out_data);
    } else if (mode == "replicate") {
      Pad3DGradReplicateNDHWC<T><<<grid, block, 0, stream>>>(out_size,
                                                             d_in_data,
                                                             num,
                                                             channels,
                                                             in_depth,
                                                             in_height,
                                                             in_width,
                                                             out_depth,
                                                             out_height,
                                                             out_width,
                                                             pad_front,
                                                             pad_top,
                                                             pad_left,
                                                             d_out_data);
    } else if (mode == "circular") {
      Pad3DGradCircularNDHWC<T><<<grid, block, 0, stream>>>(out_size,
                                                            d_in_data,
                                                            num,
                                                            channels,
                                                            in_depth,
                                                            in_height,
                                                            in_width,
                                                            out_depth,
                                                            out_height,
                                                            out_width,
                                                            pad_front,
                                                            pad_top,
                                                            pad_left,
                                                            d_out_data);
    } else {
      grid = (in_size + block - 1) / block;
      Pad3DGradConstNDHWC<T><<<grid, block, 0, stream>>>(in_size,
                                                         d_in_data,
                                                         num,
                                                         channels,
                                                         in_depth,
                                                         in_height,
                                                         in_width,
                                                         out_depth,
                                                         out_height,
                                                         out_width,
                                                         pad_front,
                                                         pad_top,
                                                         pad_left,
                                                         d_out_data);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(pad3d_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::Pad3dGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
