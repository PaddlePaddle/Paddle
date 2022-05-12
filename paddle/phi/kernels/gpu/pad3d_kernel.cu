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

#include "paddle/phi/kernels/pad3d_kernel.h"

#include <algorithm>

#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void Pad3DConstNCDHW(const int nthreads,
                                const T* in_data,
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
                                T value,
                                T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int nc = index / out_width;

    const int out_w = index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    const int out_d = nc % out_depth;
    nc /= out_depth;

    int in_d = out_d - pad_front;
    int in_h = out_h - pad_top;
    int in_w = out_w - pad_left;
    out_data[index] =
        (in_d < 0 || in_h < 0 || in_w < 0 || in_d >= in_depth ||
         in_h >= in_height || in_w >= in_width)
            ? value
            : in_data[nc * in_depth * in_height * in_width +
                      in_d * in_height * in_width + in_h * in_width + in_w];
  }
}

template <typename T>
__global__ void Pad3DConstNDHWC(const int nthreads,
                                const T* in_data,
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
                                T value,
                                T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    const int out_d = n % out_depth;
    n /= out_depth;
    const int in_d = out_d - pad_front;
    const int in_h = out_h - pad_top;
    const int in_w = out_w - pad_left;

    out_data[index] =
        (in_d < 0 || in_h < 0 || in_w < 0 || in_d >= in_depth ||
         in_h >= in_height || in_w >= in_width)
            ? value
            : in_data[n * in_depth * in_height * in_width * channels +
                      in_d * in_height * in_width * channels +
                      in_h * in_width * channels + in_w * channels + c];
  }
}

template <typename T>
__global__ void Pad3DReflectNCDHW(const int nthreads,
                                  const T* in_data,
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
                                  T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int nc = index / out_width;

    const int out_w = index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    const int out_d = nc % out_depth;
    nc /= out_depth;

    int in_d = out_d - pad_front;
    int in_h = out_h - pad_top;
    int in_w = out_w - pad_left;

    in_d = max(in_d, -in_d);                     // reflect by 0
    in_d = min(in_d, 2 * in_depth - in_d - 2);   // reflect by in_depth
    in_h = max(in_h, -in_h);                     // reflect by 0
    in_h = min(in_h, 2 * in_height - in_h - 2);  // reflect by in_height
    in_w = max(in_w, -in_w);                     // reflect by 0
    in_w = min(in_w, 2 * in_width - in_w - 2);   // reflect by in_width
    out_data[index] =
        in_data[(nc * in_depth * in_height + in_d * in_height + in_h) *
                    in_width +
                in_w];
  }
}

template <typename T>
__global__ void Pad3DReflectNDHWC(const int nthreads,
                                  const T* in_data,
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
                                  T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
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
    in_d = min(in_d, 2 * in_depth - in_d - 2);
    in_h = max(in_h, -in_h);
    in_h = min(in_h, 2 * in_height - in_h - 2);
    in_w = max(in_w, -in_w);
    in_w = min(in_w, 2 * in_width - in_w - 2);

    out_data[index] = in_data[n * in_depth * in_height * in_width * channels +
                              in_d * in_height * in_width * channels +
                              in_h * in_width * channels + in_w * channels + c];
  }
}

template <typename T>
__global__ void Pad3DReplicateNCDHW(const int nthreads,
                                    const T* in_data,
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
                                    T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int nc = index / out_width;

    const int out_w = index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    const int out_d = nc % out_depth;
    nc /= out_depth;

    int in_d = min(in_depth - 1, max(out_d - pad_front, 0));
    int in_h = min(in_height - 1, max(out_h - pad_top, 0));
    int in_w = min(in_width - 1, max(out_w - pad_left, 0));

    out_data[index] =
        in_data[(nc * in_depth * in_height + in_d * in_height + in_h) *
                    in_width +
                in_w];
  }
}

template <typename T>
__global__ void Pad3DReplicateNDHWC(const int nthreads,
                                    const T* in_data,
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
                                    T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    const int out_d = n % out_depth;
    n /= out_depth;

    int in_d = min(in_depth - 1, max(out_d - pad_front, 0));
    int in_h = min(in_height - 1, max(out_h - pad_top, 0));
    int in_w = min(in_width - 1, max(out_w - pad_left, 0));

    out_data[index] = in_data[n * in_depth * in_height * in_width * channels +
                              in_d * in_height * in_width * channels +
                              in_h * in_width * channels + in_w * channels + c];
  }
}

template <typename T>
__global__ void Pad3DCircularNCDHW(const int nthreads,
                                   const T* in_data,
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
                                   T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int nc = index / out_width;

    const int out_w = index % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    const int out_d = nc % out_depth;
    nc /= out_depth;

    int in_d = ((out_d - pad_front) % in_depth + in_depth) % in_depth;
    int in_h = ((out_h - pad_top) % in_height + in_height) % in_height;
    int in_w = ((out_w - pad_left) % in_width + in_width) % in_width;

    out_data[index] =
        in_data[(nc * in_depth * in_height + in_d * in_height + in_h) *
                    in_width +
                in_w];
  }
}

template <typename T>
__global__ void Pad3DCircularNDHWC(const int nthreads,
                                   const T* in_data,
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
                                   T* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    const int c = index % channels;
    const int out_w = n % out_width;
    n /= out_width;
    const int out_h = n % out_height;
    n /= out_height;
    const int out_d = n % out_depth;
    n /= out_depth;

    int in_d = ((out_d - pad_front) % in_depth + in_depth) % in_depth;
    int in_h = ((out_h - pad_top) % in_height + in_height) % in_height;
    int in_w = ((out_w - pad_left) % in_width + in_width) % in_width;

    out_data[index] = in_data[n * in_depth * in_height * in_width * channels +
                              in_d * in_height * in_width * channels +
                              in_h * in_width * channels + in_w * channels + c];
  }
}

template <typename T, typename Context>
void Pad3dKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const IntArray& paddings,
                 const std::string& mode,
                 float pad_value,
                 const std::string& data_format,
                 DenseTensor* out) {
  std::vector<int64_t> pads = paddings.GetData();

  auto in_dims = x.dims();
  const T* in_data = x.data<T>();
  auto out_dims = out->dims();
  T value = static_cast<T>(pad_value);

  if (data_format == "NCDHW") {
    out_dims[0] = in_dims[0];
    out_dims[1] = in_dims[1];
    out_dims[2] = in_dims[2] + pads[4] + pads[5];
    out_dims[3] = in_dims[3] + pads[2] + pads[3];
    out_dims[4] = in_dims[4] + pads[0] + pads[1];
  } else {
    out_dims[0] = in_dims[0];
    out_dims[1] = in_dims[1] + pads[4] + pads[5];
    out_dims[2] = in_dims[2] + pads[2] + pads[3];
    out_dims[3] = in_dims[3] + pads[0] + pads[1];
    out_dims[4] = in_dims[4];
  }
  out->Resize(out_dims);
  T* out_data = dev_ctx.template Alloc<T>(out);

  int channels = in_dims[1];
  int in_depth = in_dims[2];
  int in_height = in_dims[3];
  int in_width = in_dims[4];
  int out_depth = out_dims[2];
  int out_height = out_dims[3];
  int out_width = out_dims[4];
  if (data_format == "NDHWC") {
    channels = in_dims[4];
    in_depth = in_dims[1];
    in_height = in_dims[2];
    in_width = in_dims[3];
    out_depth = out_dims[1];
    out_height = out_dims[2];
    out_width = out_dims[3];
  }

  if (mode == "reflect") {
    PADDLE_ENFORCE_GT(
        in_depth,
        pads[4],
        errors::InvalidArgument("The depth of Input(X)'s dimension should be "
                                "greater than pad_front"
                                " in reflect mode"
                                ", but received depth(%d) and pad_front(%d).",
                                in_depth,
                                pads[4]));
    PADDLE_ENFORCE_GT(
        in_depth,
        pads[5],
        errors::InvalidArgument("The depth of Input(X)'s dimension should be "
                                "greater than pad_back"
                                " in reflect mode"
                                ", but received depth(%d) and pad_back(%d).",
                                in_depth,
                                pads[5]));

    PADDLE_ENFORCE_GT(
        in_height,
        pads[2],
        errors::InvalidArgument("The height of Input(X)'s dimension should be "
                                "greater than pad_top"
                                " in reflect mode"
                                ", but received depth(%d) and pad_top(%d).",
                                in_height,
                                pads[2]));
    PADDLE_ENFORCE_GT(
        in_height,
        pads[3],
        errors::InvalidArgument("The height of Input(X)'s dimension should be "
                                "greater than pad_bottom"
                                " in reflect mode"
                                ", but received depth(%d) and pad_bottom(%d).",
                                in_height,
                                pads[3]));

    PADDLE_ENFORCE_GT(
        in_width,
        pads[0],
        errors::InvalidArgument("The width of Input(X)'s dimension should be "
                                "greater than pad_left"
                                " in reflect mode"
                                ", but received depth(%d) and pad_left(%d).",
                                in_width,
                                pads[0]));
    PADDLE_ENFORCE_GT(
        in_width,
        pads[1],
        errors::InvalidArgument("The width of Input(X)'s dimension should be "
                                "greater than pad_right"
                                " in reflect mode"
                                ", but received depth(%d) and pad_right(%d).",
                                in_width,
                                pads[1]));
  } else if (mode == "circular" || mode == "replicate") {
    PADDLE_ENFORCE_NE(in_depth * in_height * in_width,
                      0,
                      errors::InvalidArgument(
                          "The input tensor size can not be 0 for circular "
                          "or replicate padding mode."));
  }

  const int pad_left = pads[0];
  const int pad_top = pads[2];
  const int pad_front = pads[4];
  const int num = in_dims[0];

  auto stream = dev_ctx.stream();
  int block = PADDLE_CUDA_NUM_THREADS;
  const int out_size = out->numel();
  int grid = (out_size + block - 1) / block;

  if (data_format == "NCDHW") {
    if (mode == "reflect") {
      Pad3DReflectNCDHW<T><<<grid, block, 0, stream>>>(out_size,
                                                       in_data,
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
                                                       out_data);
    } else if (mode == "replicate") {
      Pad3DReplicateNCDHW<T><<<grid, block, 0, stream>>>(out_size,
                                                         in_data,
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
                                                         out_data);
    } else if (mode == "circular") {
      Pad3DCircularNCDHW<T><<<grid, block, 0, stream>>>(out_size,
                                                        in_data,
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
                                                        out_data);
    } else {
      Pad3DConstNCDHW<T><<<grid, block, 0, stream>>>(out_size,
                                                     in_data,
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
                                                     value,
                                                     out_data);
    }
  } else {
    if (mode == "reflect") {
      Pad3DReflectNDHWC<T><<<grid, block, 0, stream>>>(out_size,
                                                       in_data,
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
                                                       out_data);
    } else if (mode == "replicate") {
      Pad3DReplicateNDHWC<T><<<grid, block, 0, stream>>>(out_size,
                                                         in_data,
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
                                                         out_data);
    } else if (mode == "circular") {
      Pad3DCircularNDHWC<T><<<grid, block, 0, stream>>>(out_size,
                                                        in_data,
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
                                                        out_data);
    } else {
      Pad3DConstNDHWC<T><<<grid, block, 0, stream>>>(out_size,
                                                     in_data,
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
                                                     value,
                                                     out_data);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(pad3d,
                   GPU,
                   ALL_LAYOUT,
                   phi::Pad3dKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
