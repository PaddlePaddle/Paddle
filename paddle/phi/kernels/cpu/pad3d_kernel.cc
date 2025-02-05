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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
void ConstPad3DFuncNCDHW(const T* in_data,
                         T* out_data,
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
                         const int out_w,
                         const T value) {
  int in_d = out_d - pad_front;
  int in_h = out_h - pad_top;
  int in_w = out_w - pad_left;
  out_data[out_d * out_height * out_width + out_h * out_width + out_w] =
      (in_d < 0 || in_h < 0 || in_w < 0 || in_d >= in_depth ||
       in_h >= in_height || in_w >= in_width)
          ? value
          : in_data[in_d * in_height * in_width + in_h * in_width + in_w];
}

template <typename T>
void ConstPad3DFuncNDHWC(const T* in_data,
                         T* out_data,
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
                         const int out_w,
                         const T value) {
  int in_d = out_d - pad_front;
  int in_h = out_h - pad_top;
  int in_w = out_w - pad_left;
  const int out_index =
      (out_d * out_height * out_width + out_h * out_width + out_w) * channels;
  if (in_d < 0 || in_h < 0 || in_w < 0 || in_d >= in_depth ||
      in_h >= in_height || in_w >= in_width) {
    for (int c = 0; c < channels; ++c) {
      out_data[out_index + c] = value;
    }
  } else {
    const int in_index =
        (in_d * in_height * in_width + in_h * in_width + in_w) * channels;
    for (int c = 0; c < channels; ++c) {
      out_data[out_index + c] = in_data[in_index + c];
    }
  }
}

template <typename T>
void ReflectPad3DFuncNCDHW(const T* in_data,
                           T* out_data,
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
                           const int out_w,
                           const T value UNUSED) {
  int in_d = out_d - pad_front;
  int in_h = out_h - pad_top;
  int in_w = out_w - pad_left;

  in_d = std::max(in_d, -in_d);                     // reflect by 0
  in_d = std::min(in_d, 2 * in_depth - in_d - 2);   // reflect by in_depth
  in_h = std::max(in_h, -in_h);                     // reflect by 0
  in_h = std::min(in_h, 2 * in_height - in_h - 2);  // reflect by in_height
  in_w = std::max(in_w, -in_w);                     // reflect by 0
  in_w = std::min(in_w, 2 * in_width - in_w - 2);   // reflect by in_width

  out_data[out_d * out_height * out_width + out_h * out_width + out_w] =
      in_data[in_d * in_height * in_width + in_h * in_width + in_w];
}

template <typename T>
void ReflectPad3DFuncNDHWC(const T* in_data,
                           T* out_data,
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
                           const int out_w,
                           const T value UNUSED) {
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
    out_data[out_index + c] = in_data[in_index + c];
  }
}

template <typename T>
void ReplicatePad3DFuncNCDHW(const T* in_data,
                             T* out_data,
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
                             const int out_w,
                             const T value UNUSED) {
  int in_d = std::min(in_depth - 1, std::max(out_d - pad_front, 0));
  int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
  int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));

  out_data[out_d * out_height * out_width + out_h * out_width + out_w] =
      in_data[in_d * in_height * in_width + in_h * in_width + in_w];
}

template <typename T>
void ReplicatePad3DFuncNDHWC(const T* in_data,
                             T* out_data,
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
                             const int out_w,
                             const T value UNUSED) {
  int in_d = std::min(in_depth - 1, std::max(out_d - pad_front, 0));
  int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
  int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));

  const int out_index =
      (out_d * out_height * out_width + out_h * out_width + out_w) * channels;
  const int in_index =
      (in_d * in_height * in_width + in_h * in_width + in_w) * channels;
  for (int c = 0; c < channels; ++c) {
    out_data[out_index + c] = in_data[in_index + c];
  }
}

template <typename T>
void CircularPad3DFuncNCDHW(const T* in_data,
                            T* out_data,
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
                            const int out_w,
                            const T value UNUSED) {
  int in_d = ((out_d - pad_front) % in_depth + in_depth) % in_depth;
  int in_h = ((out_h - pad_top) % in_height + in_height) % in_height;
  int in_w = ((out_w - pad_left) % in_width + in_width) % in_width;

  out_data[out_d * out_height * out_width + out_h * out_width + out_w] =
      in_data[in_d * in_height * in_width + in_h * in_width + in_w];
}

template <typename T>
void CircularPad3DFuncNDHWC(const T* in_data,
                            T* out_data,
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
                            const int out_w,
                            const T value UNUSED) {
  int in_d = ((out_d - pad_front) % in_depth + in_depth) % in_depth;
  int in_h = ((out_h - pad_top) % in_height + in_height) % in_height;
  int in_w = ((out_w - pad_left) % in_width + in_width) % in_width;

  const int out_index =
      (out_d * out_height * out_width + out_h * out_width + out_w) * channels;
  const int in_index =
      (in_d * in_height * in_width + in_h * in_width + in_w) * channels;
  for (int c = 0; c < channels; ++c) {
    out_data[out_index + c] = in_data[in_index + c];
  }
}

template <typename T>
void Pad3DNCDHW(const T* in_data,
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
                T* out_data,
                void (*pad_func)(const T*,
                                 T*,
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
                                 const T)) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_d = 0; out_d < out_depth; ++out_d) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
          for (int out_w = 0; out_w < out_width; ++out_w) {
            pad_func(in_data,
                     out_data,
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
                     out_w,
                     value);
          }
        }
      }
      in_data += in_depth * in_height * in_width;
      out_data += out_depth * out_height * out_width;
    }
  }
}

template <typename T>
void Pad3DNDHWC(const T* in_data,
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
                T* out_data,
                void (*pad_func)(const T*,
                                 T*,
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
                                 const int,
                                 const T)) {
  for (int n = 0; n < num; ++n) {
    for (int out_d = 0; out_d < out_depth; ++out_d) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          pad_func(in_data,
                   out_data,
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
                   out_w,
                   value);
        }
      }
    }
    in_data += in_depth * in_height * in_width * channels;
    out_data += out_depth * out_height * out_width * channels;
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
  T value = static_cast<T>(pad_value);
  std::vector<int64_t> pads = paddings.GetData();

  auto in_dims = x.dims();
  const T* in_data = x.data<T>();

  if (data_format == "NCDHW") {
    out->Resize({in_dims[0],
                 in_dims[1],
                 in_dims[2] + pads[4] + pads[5],
                 in_dims[3] + pads[2] + pads[3],
                 in_dims[4] + pads[0] + pads[1]});
  } else {
    out->Resize({in_dims[0],
                 in_dims[1] + pads[4] + pads[5],
                 in_dims[2] + pads[2] + pads[3],
                 in_dims[3] + pads[0] + pads[1],
                 in_dims[4]});
  }

  auto out_dims = out->dims();
  T* out_data = dev_ctx.template Alloc<T>(out);

  int channels = static_cast<int>(in_dims[1]);
  int in_depth = static_cast<int>(in_dims[2]);
  int in_height = static_cast<int>(in_dims[3]);
  int in_width = static_cast<int>(in_dims[4]);
  int out_depth = static_cast<int>(out_dims[2]);
  int out_height = static_cast<int>(out_dims[3]);
  int out_width = static_cast<int>(out_dims[4]);
  if (data_format == "NDHWC") {
    channels = static_cast<int>(in_dims[4]);
    in_depth = static_cast<int>(in_dims[1]);
    in_height = static_cast<int>(in_dims[2]);
    in_width = static_cast<int>(in_dims[3]);
    out_depth = static_cast<int>(out_dims[1]);
    out_height = static_cast<int>(out_dims[2]);
    out_width = static_cast<int>(out_dims[3]);
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

  const int pad_left = static_cast<int>(pads[0]);
  const int pad_top = static_cast<int>(pads[2]);
  const int pad_front = static_cast<int>(pads[4]);
  const int num = static_cast<int>(in_dims[0]);
  if (data_format == "NCDHW") {
    std::map<std::string,
             void (*)(const T*,
                      T*,
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
                      const T)>
        func_map;

    func_map["reflect"] = ReflectPad3DFuncNCDHW;
    func_map["replicate"] = ReplicatePad3DFuncNCDHW;
    func_map["circular"] = CircularPad3DFuncNCDHW;
    func_map["constant"] = ConstPad3DFuncNCDHW;
    Pad3DNCDHW(in_data,
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
               out_data,
               func_map[mode]);
  } else {
    std::map<std::string,
             void (*)(const T*,
                      T*,
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
                      const int,
                      const T)>
        func_map;

    func_map["reflect"] = ReflectPad3DFuncNDHWC;
    func_map["replicate"] = ReplicatePad3DFuncNDHWC;
    func_map["circular"] = CircularPad3DFuncNDHWC;
    func_map["constant"] = ConstPad3DFuncNDHWC;
    Pad3DNDHWC(in_data,
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
               out_data,
               func_map[mode]);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(pad3d,
                   CPU,
                   ALL_LAYOUT,
                   phi::Pad3dKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
