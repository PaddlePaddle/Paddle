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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
void ConstPad3DGradNCDHW(T* d_in_data,
                         const T* d_out_data,
                         const int in_depth,
                         const int in_height,
                         const int in_width,
                         const int out_depth UNUSED,
                         const int out_height,
                         const int out_width,
                         const int pad_front,
                         const int pad_top,
                         const int pad_left,
                         const int out_d,
                         const int out_h,
                         const int out_w) {
  int in_d = out_d - pad_front;
  int in_h = out_h - pad_top;
  int in_w = out_w - pad_left;
  if (!(in_d < 0 || in_h < 0 || in_w < 0 || in_d >= in_depth ||
        in_h >= in_height || in_w >= in_width)) {
    d_in_data[in_d * in_height * in_width + in_h * in_width + in_w] =
        d_out_data[out_d * out_height * out_width + out_h * out_width + out_w];
  }
}

template <typename T>
void ConstPad3DGradNDHWC(T* d_in_data,
                         const T* d_out_data,
                         const int channels,
                         const int in_depth,
                         const int in_height,
                         const int in_width,
                         const int out_depth UNUSED,
                         const int out_height,
                         const int out_width,
                         const int pad_front,
                         const int pad_top,
                         const int pad_left,
                         const int out_d,
                         const int out_h,
                         const int out_w) {
  int in_d = out_d - pad_front;
  int in_h = out_h - pad_top;
  int in_w = out_w - pad_left;

  const int out_index =
      (out_d * out_height * out_width + out_h * out_width + out_w) * channels;
  if (!(in_d < 0 || in_h < 0 || in_w < 0 || in_d >= in_depth ||
        in_h >= in_height || in_w >= in_width)) {
    const int in_index =
        (in_d * in_height * in_width + in_h * in_width + in_w) * channels;
    for (int c = 0; c < channels; ++c) {
      d_in_data[in_index + c] = d_out_data[out_index + c];
    }
  }
}

template <typename T>
void ReflectPad3DGradNCDHW(T* d_in_data,
                           const T* d_out_data,
                           const int in_depth,
                           const int in_height,
                           const int in_width,
                           const int out_depth UNUSED,
                           const int out_height,
                           const int out_width,
                           const int pad_front,
                           const int pad_top,
                           const int pad_left,
                           const int out_d,
                           const int out_h,
                           const int out_w) {
  int in_d = out_d - pad_front;
  int in_h = out_h - pad_top;
  int in_w = out_w - pad_left;

  in_d = std::max(in_d, -in_d);                     // reflect by 0
  in_d = std::min(in_d, 2 * in_depth - in_d - 2);   // reflect by in_depth
  in_h = std::max(in_h, -in_h);                     // reflect by 0
  in_h = std::min(in_h, 2 * in_height - in_h - 2);  // reflect by in_height
  in_w = std::max(in_w, -in_w);                     // reflect by 0
  in_w = std::min(in_w, 2 * in_width - in_w - 2);   // reflect by in_width

  d_in_data[in_d * in_height * in_width + in_h * in_width + in_w] +=
      d_out_data[out_d * out_height * out_width + out_h * out_width + out_w];
}

template <typename T>
void ReflectPad3DGradNDHWC(T* d_in_data,
                           const T* d_out_data,
                           const int channels,
                           const int in_depth,
                           const int in_height,
                           const int in_width,
                           const int out_depth UNUSED,
                           const int out_height,
                           const int out_width,
                           const int pad_front,
                           const int pad_top,
                           const int pad_left,
                           const int out_d,
                           const int out_h,
                           const int out_w) {
  int in_d = out_d - pad_front;
  int in_h = out_h - pad_top;
  int in_w = out_w - pad_left;

  in_d = std::max(in_d, -in_d);
  in_d = std::min(in_d, 2 * in_depth - in_d - 2);
  in_h = std::max(in_h, -in_h);
  in_h = std::min(in_h, 2 * in_height - in_h - 2);
  in_w = std::max(in_w, -in_w);
  in_w = std::min(in_w, 2 * in_width - in_w - 2);

  const int out_index =
      (out_d * out_height * out_width + out_h * out_width + out_w) * channels;
  const int in_index =
      (in_d * in_height * in_width + in_h * in_width + in_w) * channels;
  for (int c = 0; c < channels; ++c) {
    d_in_data[in_index + c] += d_out_data[out_index + c];
  }
}

template <typename T>
void ReplicatePad3DGradNCDHW(T* d_in_data,
                             const T* d_out_data,
                             const int in_depth,
                             const int in_height,
                             const int in_width,
                             const int out_depth UNUSED,
                             const int out_height,
                             const int out_width,
                             const int pad_front,
                             const int pad_top,
                             const int pad_left,
                             const int out_d,
                             const int out_h,
                             const int out_w) {
  int in_d = std::min(in_depth - 1, std::max(out_d - pad_front, 0));
  int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
  int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));

  d_in_data[in_d * in_height * in_width + in_h * in_width + in_w] +=
      d_out_data[out_d * out_height * out_width + out_h * out_width + out_w];
}

template <typename T>
void ReplicatePad3DGradNDHWC(T* d_in_data,
                             const T* d_out_data,
                             const int channels,
                             const int in_depth,
                             const int in_height,
                             const int in_width,
                             const int out_depth UNUSED,
                             const int out_height,
                             const int out_width,
                             const int pad_front,
                             const int pad_top,
                             const int pad_left,
                             const int out_d,
                             const int out_h,
                             const int out_w) {
  int in_d = std::min(in_depth - 1, std::max(out_d - pad_front, 0));
  int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
  int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));

  const int out_index =
      (out_d * out_height * out_width + out_h * out_width + out_w) * channels;
  const int in_index =
      (in_d * in_height * in_width + in_h * in_width + in_w) * channels;
  for (int c = 0; c < channels; ++c) {
    d_in_data[in_index + c] += d_out_data[out_index + c];
  }
}

template <typename T>
void CircularPad3DGradNCDHW(T* d_in_data,
                            const T* d_out_data,
                            const int in_depth,
                            const int in_height,
                            const int in_width,
                            const int out_depth UNUSED,
                            const int out_height,
                            const int out_width,
                            const int pad_front,
                            const int pad_top,
                            const int pad_left,
                            const int out_d,
                            const int out_h,
                            const int out_w) {
  int in_d = ((out_d - pad_front) % in_depth + in_depth) % in_depth;
  int in_h = ((out_h - pad_top) % in_height + in_height) % in_height;
  int in_w = ((out_w - pad_left) % in_width + in_width) % in_width;
  d_in_data[in_d * in_height * in_width + in_h * in_width + in_w] +=
      d_out_data[out_d * out_height * out_width + out_h * out_width + out_w];
}

template <typename T>
void CircularPad3DGradNDHWC(T* d_in_data,
                            const T* d_out_data,
                            const int channels,
                            const int in_depth,
                            const int in_height,
                            const int in_width,
                            const int out_depth UNUSED,
                            const int out_height,
                            const int out_width,
                            const int pad_front,
                            const int pad_top,
                            const int pad_left,
                            const int out_d,
                            const int out_h,
                            const int out_w) {
  int in_d = ((out_d - pad_front) % in_depth + in_depth) % in_depth;
  int in_h = ((out_h - pad_top) % in_height + in_height) % in_height;
  int in_w = ((out_w - pad_left) % in_width + in_width) % in_width;

  const int out_index =
      (out_d * out_height * out_width + out_h * out_width + out_w) * channels;
  const int in_index =
      (in_d * in_height * in_width + in_h * in_width + in_w) * channels;
  for (int c = 0; c < channels; ++c) {
    d_in_data[in_index + c] += d_out_data[out_index + c];
  }
}

template <typename T>
void Pad3DGradNCDHW(T* d_in_data,
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
                    const T* d_out_data,
                    void (*pad_func)(T*,
                                     const T*,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int)) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_d = 0; out_d < out_depth; ++out_d) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
          for (int out_w = 0; out_w < out_width; ++out_w) {
            pad_func(d_in_data,
                     d_out_data,
                     in_depth,
                     in_height,
                     in_width,
                     out_depth,
                     out_height,
                     out_width,
                     pad_front,
                     pad_top,
                     pad_left,
                     out_d,
                     out_h,
                     out_w);
          }
        }
      }
      d_in_data += in_depth * in_height * in_width;
      d_out_data += out_depth * out_height * out_width;
    }
  }
}

template <typename T>
void Pad3DGradNDHWC(T* d_in_data,
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
                    const T* d_out_data,
                    void (*pad_func)(T*,
                                     const T*,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int,
                                     const int)) {
  for (int n = 0; n < num; ++n) {
    for (int out_d = 0; out_d < out_depth; ++out_d) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          pad_func(d_in_data,
                   d_out_data,
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
                   out_d,
                   out_h,
                   out_w);
        }
      }
    }
    d_in_data += in_depth * in_height * in_width * channels;
    d_out_data += out_depth * out_height * out_width * channels;
  }
}

template <typename T, typename Context>
void Pad3dGradKernel(const Context& dev_ctx,
                     const DenseTensor& x UNUSED,
                     const DenseTensor& out_grad,
                     const IntArray& paddings,
                     const std::string& mode,
                     float pad_value UNUSED,
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

  const int pad_left = static_cast<int>(pads[0]);
  const int pad_top = static_cast<int>(pads[2]);
  const int pad_front = static_cast<int>(pads[4]);
  const int num = static_cast<int>(d_in_dims[0]);
  if (data_format == "NCDHW") {
    const int channels = static_cast<int>(d_in_dims[1]);
    const int in_depth = static_cast<int>(d_in_dims[2]);
    const int in_height = static_cast<int>(d_in_dims[3]);
    const int in_width = static_cast<int>(d_in_dims[4]);
    const int out_depth = static_cast<int>(d_out_dims[2]);
    const int out_height = static_cast<int>(d_out_dims[3]);
    const int out_width = static_cast<int>(d_out_dims[4]);

    std::map<std::string,
             void (*)(T*,
                      const T*,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int)>
        func_map;

    func_map["reflect"] = ReflectPad3DGradNCDHW;
    func_map["replicate"] = ReplicatePad3DGradNCDHW;
    func_map["circular"] = CircularPad3DGradNCDHW;
    func_map["constant"] = ConstPad3DGradNCDHW;

    Pad3DGradNCDHW(d_in_data,
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
                   d_out_data,
                   func_map[mode]);
  } else {
    const int channels = static_cast<int>(d_in_dims[4]);
    const int in_depth = static_cast<int>(d_in_dims[1]);
    const int in_height = static_cast<int>(d_in_dims[2]);
    const int in_width = static_cast<int>(d_in_dims[3]);
    const int out_depth = static_cast<int>(d_out_dims[1]);
    const int out_height = static_cast<int>(d_out_dims[2]);
    const int out_width = static_cast<int>(d_out_dims[3]);

    std::map<std::string,
             void (*)(T*,
                      const T*,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int,
                      const int)>
        func_map;

    func_map["reflect"] = ReflectPad3DGradNDHWC;
    func_map["replicate"] = ReplicatePad3DGradNDHWC;
    func_map["circular"] = CircularPad3DGradNDHWC;
    func_map["constant"] = ConstPad3DGradNDHWC;

    Pad3DGradNDHWC(d_in_data,
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
                   d_out_data,
                   func_map[mode]);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(pad3d_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::Pad3dGradKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
