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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
void ConstPad3DFuncNCDHW(const T* in_data, T* out_data, const int in_depth,
                         const int in_height, const int in_width,
                         const int out_depth, const int out_height,
                         const int out_width, const int pad_front,
                         const int pad_top, const int pad_left, const int out_d,
                         const int out_h, const int out_w, const T value) {
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
void ConstPad3DFuncNDHWC(const T* in_data, T* out_data, const int channels,
                         const int in_depth, const int in_height,
                         const int in_width, const int out_depth,
                         const int out_height, const int out_width,
                         const int pad_front, const int pad_top,
                         const int pad_left, const int out_d, const int out_h,
                         const int out_w, const T value) {
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
void ReflectPad3DFuncNCDHW(const T* in_data, T* out_data, const int in_depth,
                           const int in_height, const int in_width,
                           const int out_depth, const int out_height,
                           const int out_width, const int pad_front,
                           const int pad_top, const int pad_left,
                           const int out_d, const int out_h, const int out_w,
                           const T value) {
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
void ReflectPad3DFuncNDHWC(const T* in_data, T* out_data, const int channels,
                           const int in_depth, const int in_height,
                           const int in_width, const int out_depth,
                           const int out_height, const int out_width,
                           const int pad_front, const int pad_top,
                           const int pad_left, const int out_d, const int out_h,
                           const int out_w, const T value) {
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
void ReplicatePad3DFuncNCDHW(const T* in_data, T* out_data, const int in_depth,
                             const int in_height, const int in_width,
                             const int out_depth, const int out_height,
                             const int out_width, const int pad_front,
                             const int pad_top, const int pad_left,
                             const int out_d, const int out_h, const int out_w,
                             const T value) {
  int in_d = std::min(in_depth - 1, std::max(out_d - pad_front, 0));
  int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
  int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));

  out_data[out_d * out_height * out_width + out_h * out_width + out_w] =
      in_data[in_d * in_height * in_width + in_h * in_width + in_w];
}

template <typename T>
void ReplicatePad3DFuncNDHWC(const T* in_data, T* out_data, const int channels,
                             const int in_depth, const int in_height,
                             const int in_width, const int out_depth,
                             const int out_height, const int out_width,
                             const int pad_front, const int pad_top,
                             const int pad_left, const int out_d,
                             const int out_h, const int out_w, const T value) {
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
void CircularPad3DFuncNCDHW(const T* in_data, T* out_data, const int in_depth,
                            const int in_height, const int in_width,
                            const int out_depth, const int out_height,
                            const int out_width, const int pad_front,
                            const int pad_top, const int pad_left,
                            const int out_d, const int out_h, const int out_w,
                            const T value) {
  int in_d = ((out_d - pad_front) % in_depth + in_depth) % in_depth;
  int in_h = ((out_h - pad_top) % in_height + in_height) % in_height;
  int in_w = ((out_w - pad_left) % in_width + in_width) % in_width;

  out_data[out_d * out_height * out_width + out_h * out_width + out_w] =
      in_data[in_d * in_height * in_width + in_h * in_width + in_w];
}

template <typename T>
void CircularPad3DFuncNDHWC(const T* in_data, T* out_data, const int channels,
                            const int in_depth, const int in_height,
                            const int in_width, const int out_depth,
                            const int out_height, const int out_width,
                            const int pad_front, const int pad_top,
                            const int pad_left, const int out_d,
                            const int out_h, const int out_w, const T value) {
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
void Pad3DNCDHW(const T* in_data, const int num, const int channels,
                const int in_depth, const int in_height, const int in_width,
                const int out_depth, const int out_height, const int out_width,
                const int pad_front, const int pad_top, const int pad_left,
                T value, T* out_data,
                void (*pad_func)(const T*, T*, const int, const int, const int,
                                 const int, const int, const int, const int,
                                 const int, const int, const int, const int,
                                 const int, const T)) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_d = 0; out_d < out_depth; ++out_d) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
          for (int out_w = 0; out_w < out_width; ++out_w) {
            pad_func(in_data, out_data, in_depth, in_height, in_width,
                     out_depth, out_height, out_width, pad_front, pad_top,
                     pad_left, out_d, out_h, out_w, value);
          }
        }
      }
      in_data += in_depth * in_height * in_width;
      out_data += out_depth * out_height * out_width;
    }
  }
}

template <typename T>
void Pad3DNDHWC(const T* in_data, const int num, const int channels,
                const int in_depth, const int in_height, const int in_width,
                const int out_depth, const int out_height, const int out_width,
                const int pad_front, const int pad_top, const int pad_left,
                T value, T* out_data,
                void (*pad_func)(const T*, T*, const int, const int, const int,
                                 const int, const int, const int, const int,
                                 const int, const int, const int, const int,
                                 const int, const int, const T)) {
  for (int n = 0; n < num; ++n) {
    for (int out_d = 0; out_d < out_depth; ++out_d) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          pad_func(in_data, out_data, channels, in_depth, in_height, in_width,
                   out_depth, out_height, out_width, pad_front, pad_top,
                   pad_left, out_d, out_h, out_w, value);
        }
      }
    }
    in_data += in_depth * in_height * in_width * channels;
    out_data += out_depth * out_height * out_width * channels;
  }
}

template <typename T>
void ConstPad3DGradNCDHW(T* d_in_data, const T* d_out_data, const int in_depth,
                         const int in_height, const int in_width,
                         const int out_depth, const int out_height,
                         const int out_width, const int pad_front,
                         const int pad_top, const int pad_left, const int out_d,
                         const int out_h, const int out_w) {
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
void ConstPad3DGradNDHWC(T* d_in_data, const T* d_out_data, const int channels,
                         const int in_depth, const int in_height,
                         const int in_width, const int out_depth,
                         const int out_height, const int out_width,
                         const int pad_front, const int pad_top,
                         const int pad_left, const int out_d, const int out_h,
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
void ReflectPad3DGradNCDHW(T* d_in_data, const T* d_out_data,
                           const int in_depth, const int in_height,
                           const int in_width, const int out_depth,
                           const int out_height, const int out_width,
                           const int pad_front, const int pad_top,
                           const int pad_left, const int out_d, const int out_h,
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
void ReflectPad3DGradNDHWC(T* d_in_data, const T* d_out_data,
                           const int channels, const int in_depth,
                           const int in_height, const int in_width,
                           const int out_depth, const int out_height,
                           const int out_width, const int pad_front,
                           const int pad_top, const int pad_left,
                           const int out_d, const int out_h, const int out_w) {
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
void ReplicatePad3DGradNCDHW(T* d_in_data, const T* d_out_data,
                             const int in_depth, const int in_height,
                             const int in_width, const int out_depth,
                             const int out_height, const int out_width,
                             const int pad_front, const int pad_top,
                             const int pad_left, const int out_d,
                             const int out_h, const int out_w) {
  int in_d = std::min(in_depth - 1, std::max(out_d - pad_front, 0));
  int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
  int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));

  d_in_data[in_d * in_height * in_width + in_h * in_width + in_w] +=
      d_out_data[out_d * out_height * out_width + out_h * out_width + out_w];
}

template <typename T>
void ReplicatePad3DGradNDHWC(T* d_in_data, const T* d_out_data,
                             const int channels, const int in_depth,
                             const int in_height, const int in_width,
                             const int out_depth, const int out_height,
                             const int out_width, const int pad_front,
                             const int pad_top, const int pad_left,
                             const int out_d, const int out_h,
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
void CircularPad3DGradNCDHW(T* d_in_data, const T* d_out_data,
                            const int in_depth, const int in_height,
                            const int in_width, const int out_depth,
                            const int out_height, const int out_width,
                            const int pad_front, const int pad_top,
                            const int pad_left, const int out_d,
                            const int out_h, const int out_w) {
  int in_d = ((out_d - pad_front) % in_depth + in_depth) % in_depth;
  int in_h = ((out_h - pad_top) % in_height + in_height) % in_height;
  int in_w = ((out_w - pad_left) % in_width + in_width) % in_width;
  d_in_data[in_d * in_height * in_width + in_h * in_width + in_w] +=
      d_out_data[out_d * out_height * out_width + out_h * out_width + out_w];
}

template <typename T>
void CircularPad3DGradNDHWC(T* d_in_data, const T* d_out_data,
                            const int channels, const int in_depth,
                            const int in_height, const int in_width,
                            const int out_depth, const int out_height,
                            const int out_width, const int pad_front,
                            const int pad_top, const int pad_left,
                            const int out_d, const int out_h, const int out_w) {
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
void Pad3DGradNCDHW(T* d_in_data, const int num, const int channels,
                    const int in_depth, const int in_height, const int in_width,
                    const int out_depth, const int out_height,
                    const int out_width, const int pad_front, const int pad_top,
                    const int pad_left, const T* d_out_data,
                    void (*pad_func)(T*, const T*, const int, const int,
                                     const int, const int, const int, const int,
                                     const int, const int, const int, const int,
                                     const int, const int)) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_d = 0; out_d < out_depth; ++out_d) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
          for (int out_w = 0; out_w < out_width; ++out_w) {
            pad_func(d_in_data, d_out_data, in_depth, in_height, in_width,
                     out_depth, out_height, out_width, pad_front, pad_top,
                     pad_left, out_d, out_h, out_w);
          }
        }
      }
      d_in_data += in_depth * in_height * in_width;
      d_out_data += out_depth * out_height * out_width;
    }
  }
}

template <typename T>
void Pad3DGradNDHWC(T* d_in_data, const int num, const int channels,
                    const int in_depth, const int in_height, const int in_width,
                    const int out_depth, const int out_height,
                    const int out_width, const int pad_front, const int pad_top,
                    const int pad_left, const T* d_out_data,
                    void (*pad_func)(T*, const T*, const int, const int,
                                     const int, const int, const int, const int,
                                     const int, const int, const int, const int,
                                     const int, const int, const int)) {
  for (int n = 0; n < num; ++n) {
    for (int out_d = 0; out_d < out_depth; ++out_d) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          pad_func(d_in_data, d_out_data, channels, in_depth, in_height,
                   in_width, out_depth, out_height, out_width, pad_front,
                   pad_top, pad_left, out_d, out_h, out_w);
        }
      }
    }
    d_in_data += in_depth * in_height * in_width * channels;
    d_out_data += out_depth * out_height * out_width * channels;
  }
}

static inline std::vector<int> GetPaddings(
    const framework::ExecutionContext& context) {
  std::vector<int> paddings(6);
  auto* paddings_t = context.Input<Tensor>("Paddings");
  if (paddings_t) {
    auto paddings_data = paddings_t->data<int>();
    std::memcpy(paddings.data(), paddings_data, paddings.size() * sizeof(int));
  } else {
    auto pads = context.Attr<std::vector<int>>("paddings");
    std::copy(pads.begin(), pads.end(), paddings.data());
  }
  return paddings;
}

template <typename T>
class Pad3dCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    std::vector<int> pads = GetPaddings(context);
    auto mode = context.Attr<std::string>("mode");
    auto data_format = context.Attr<std::string>("data_format");
    T value = static_cast<T>(context.Attr<float>("value"));

    auto* x = context.Input<Tensor>("X");
    auto in_dims = x->dims();
    const T* in_data = x->data<T>();

    auto* out = context.Output<Tensor>("Out");
    if (data_format == "NCDHW") {
      out->Resize({in_dims[0], in_dims[1], in_dims[2] + pads[4] + pads[5],
                   in_dims[3] + pads[2] + pads[3],
                   in_dims[4] + pads[0] + pads[1]});
    } else {
      out->Resize({in_dims[0], in_dims[1] + pads[4] + pads[5],
                   in_dims[2] + pads[2] + pads[3],
                   in_dims[3] + pads[0] + pads[1], in_dims[4]});
    }
    auto out_dims = out->dims();
    T* out_data = out->mutable_data<T>(context.GetPlace());

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
      PADDLE_ENFORCE_GT(in_depth, pads[4],
                        platform::errors::InvalidArgument(
                            "The depth of Input(X)'s dimension should be "
                            "greater than pad_front"
                            " in reflect mode"
                            ", but received depth(%d) and pad_front(%d).",
                            in_depth, pads[4]));
      PADDLE_ENFORCE_GT(in_depth, pads[5],
                        platform::errors::InvalidArgument(
                            "The depth of Input(X)'s dimension should be "
                            "greater than pad_back"
                            " in reflect mode"
                            ", but received depth(%d) and pad_back(%d).",
                            in_depth, pads[5]));

      PADDLE_ENFORCE_GT(in_height, pads[2],
                        platform::errors::InvalidArgument(
                            "The height of Input(X)'s dimension should be "
                            "greater than pad_top"
                            " in reflect mode"
                            ", but received depth(%d) and pad_top(%d).",
                            in_height, pads[2]));
      PADDLE_ENFORCE_GT(in_height, pads[3],
                        platform::errors::InvalidArgument(
                            "The height of Input(X)'s dimension should be "
                            "greater than pad_bottom"
                            " in reflect mode"
                            ", but received depth(%d) and pad_bottom(%d).",
                            in_height, pads[3]));

      PADDLE_ENFORCE_GT(in_width, pads[0],
                        platform::errors::InvalidArgument(
                            "The width of Input(X)'s dimension should be "
                            "greater than pad_left"
                            " in reflect mode"
                            ", but received depth(%d) and pad_left(%d).",
                            in_width, pads[0]));
      PADDLE_ENFORCE_GT(in_width, pads[1],
                        platform::errors::InvalidArgument(
                            "The width of Input(X)'s dimension should be "
                            "greater than pad_right"
                            " in reflect mode"
                            ", but received depth(%d) and pad_right(%d).",
                            in_width, pads[1]));
    } else if (mode == "circular" || mode == "replicate") {
      PADDLE_ENFORCE_NE(in_depth * in_height * in_width, 0,
                        platform::errors::InvalidArgument(
                            "The input tensor size can not be 0 for circular "
                            "or replicate padding mode."));
    }

    const int pad_left = pads[0];
    const int pad_top = pads[2];
    const int pad_front = pads[4];
    const int num = in_dims[0];
    if (data_format == "NCDHW") {
      std::map<std::string,
               void (*)(const T*, T*, const int, const int, const int,
                        const int, const int, const int, const int, const int,
                        const int, const int, const int, const int, const T)>
          func_map;

      func_map["reflect"] = ReflectPad3DFuncNCDHW;
      func_map["replicate"] = ReplicatePad3DFuncNCDHW;
      func_map["circular"] = CircularPad3DFuncNCDHW;
      func_map["constant"] = ConstPad3DFuncNCDHW;
      Pad3DNCDHW(in_data, num, channels, in_depth, in_height, in_width,
                 out_depth, out_height, out_width, pad_front, pad_top, pad_left,
                 value, out_data, func_map[mode]);
    } else {
      std::map<std::string, void (*)(const T*, T*, const int, const int,
                                     const int, const int, const int, const int,
                                     const int, const int, const int, const int,
                                     const int, const int, const int, const T)>
          func_map;

      func_map["reflect"] = ReflectPad3DFuncNDHWC;
      func_map["replicate"] = ReplicatePad3DFuncNDHWC;
      func_map["circular"] = CircularPad3DFuncNDHWC;
      func_map["constant"] = ConstPad3DFuncNDHWC;
      Pad3DNDHWC(in_data, num, channels, in_depth, in_height, in_width,
                 out_depth, out_height, out_width, pad_front, pad_top, pad_left,
                 value, out_data, func_map[mode]);
    }
  }
};

template <typename T>
class Pad3dGradCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    std::vector<int> pads = GetPaddings(context);
    auto mode = context.Attr<std::string>("mode");
    auto data_format = context.Attr<std::string>("data_format");
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_in = context.Output<Tensor>(framework::GradVarName("X"));
    auto d_in_dims = d_in->dims();
    auto d_out_dims = d_out->dims();
    const T* d_out_data = d_out->data<T>();
    T* d_in_data = d_in->mutable_data<T>(context.GetPlace());
    pten::funcs::SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(context.template device_context<platform::CPUDeviceContext>(),
             d_in, static_cast<T>(0));
    const int pad_left = pads[0];
    const int pad_top = pads[2];
    const int pad_front = pads[4];
    const int num = d_in_dims[0];
    if (data_format == "NCDHW") {
      const int channels = d_in_dims[1];
      const int in_depth = d_in_dims[2];
      const int in_height = d_in_dims[3];
      const int in_width = d_in_dims[4];
      const int out_depth = d_out_dims[2];
      const int out_height = d_out_dims[3];
      const int out_width = d_out_dims[4];

      std::map<std::string,
               void (*)(T*, const T*, const int, const int, const int,
                        const int, const int, const int, const int, const int,
                        const int, const int, const int, const int)>
          func_map;

      func_map["reflect"] = ReflectPad3DGradNCDHW;
      func_map["replicate"] = ReplicatePad3DGradNCDHW;
      func_map["circular"] = CircularPad3DGradNCDHW;
      func_map["constant"] = ConstPad3DGradNCDHW;

      Pad3DGradNCDHW(d_in_data, num, channels, in_depth, in_height, in_width,
                     out_depth, out_height, out_width, pad_front, pad_top,
                     pad_left, d_out_data, func_map[mode]);
    } else {
      const int channels = d_in_dims[4];
      const int in_depth = d_in_dims[1];
      const int in_height = d_in_dims[2];
      const int in_width = d_in_dims[3];
      const int out_depth = d_out_dims[1];
      const int out_height = d_out_dims[2];
      const int out_width = d_out_dims[3];

      std::map<std::string,
               void (*)(T*, const T*, const int, const int, const int,
                        const int, const int, const int, const int, const int,
                        const int, const int, const int, const int, const int)>
          func_map;

      func_map["reflect"] = ReflectPad3DGradNDHWC;
      func_map["replicate"] = ReplicatePad3DGradNDHWC;
      func_map["circular"] = CircularPad3DGradNDHWC;
      func_map["constant"] = ConstPad3DGradNDHWC;

      Pad3DGradNDHWC(d_in_data, num, channels, in_depth, in_height, in_width,
                     out_depth, out_height, out_width, pad_front, pad_top,
                     pad_left, d_out_data, func_map[mode]);
    }
  }
};

class Pad3dOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Pad3d");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Pad3d");

    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dim.size(), 5,
                      platform::errors::InvalidArgument(
                          "The size of Input(X)'s dimension should be equal to "
                          "5, but received %d. ",
                          x_dim.size()));

    std::vector<int64_t> out_dims(x_dim.size());
    auto data_format = ctx->Attrs().Get<std::string>("data_format");
    out_dims[0] = x_dim[0];
    if (ctx->HasInput("Paddings")) {
      auto paddings_dim = ctx->GetInputDim("Paddings");
      PADDLE_ENFORCE_EQ(paddings_dim.size(), 1,
                        platform::errors::InvalidArgument(
                            "Size of Input(Paddings)'s dimension should be "
                            "equal to 1, but received %d.",
                            paddings_dim.size()));
      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(paddings_dim[0], 6,
                          platform::errors::InvalidArgument(
                              "Shape of Input(Paddings) should be equal to "
                              "[6], but received [%d].",
                              paddings_dim[0]));
      }
      out_dims[1] = x_dim[1];
      out_dims[2] = x_dim[2];
      out_dims[3] = x_dim[3];
    } else {
      auto paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
      PADDLE_ENFORCE_EQ(
          paddings.size(), 6,
          platform::errors::InvalidArgument(
              "Size of paddings should be equal to 4, but received %d.",
              static_cast<int>(paddings.size())));
      if (data_format == "NCDHW") {
        out_dims[1] = x_dim[1];  // channel
        out_dims[2] = ((!ctx->IsRuntime()) && (x_dim[2] < 0))
                          ? x_dim[2]
                          : (x_dim[2] + paddings[4] + paddings[5]);  // depth

        out_dims[3] = ((!ctx->IsRuntime()) && (x_dim[3] < 0))
                          ? x_dim[3]
                          : (x_dim[3] + paddings[2] + paddings[3]);  // height

        out_dims[4] = ((!ctx->IsRuntime()) && (x_dim[4] < 0))
                          ? x_dim[4]
                          : (x_dim[4] + paddings[0] + paddings[1]);  // width
      } else {                                                       // NDHWC
        out_dims[4] = x_dim[4];                                      // channel

        out_dims[1] = ((!ctx->IsRuntime()) && (x_dim[1] < 0))
                          ? x_dim[1]
                          : (x_dim[1] + paddings[4] + paddings[5]);  // depth
        out_dims[2] = ((!ctx->IsRuntime()) && (x_dim[2] < 0))
                          ? x_dim[2]
                          : (x_dim[2] + paddings[2] + paddings[3]);  // height
        out_dims[3] = ((!ctx->IsRuntime()) && (x_dim[3] < 0))
                          ? x_dim[3]
                          : (x_dim[3] + paddings[0] + paddings[1]);  // width
      }
    }

    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class Pad3dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of pad3d op. "
             "The input should be a 5-D tensor with formate NCDHW or NDHWC.");
    AddOutput("Out",
              "The output of pad3d op. "
              "A tensor with the same shape as X.");
    AddInput("Paddings",
             "A 1-D tensor to describe the padding rules."
             "paddings=[0, 1, 2, 3, 4, 5] means "
             "padding 0 column to left, 1 column to right, "
             "2 row to top, 3 row to bottom, 4 depth to front "
             "and 5 depth to back. Size of paddings must be 6.")
        .AsDispensable();
    AddAttr<std::vector<int>>(
        "paddings",
        "(vector<int>) "
        "A list<int> to describe the padding rules."
        "paddings=[0, 1, 2, 3, 4, 5] means "
        "padding 0 column to left, 1 column to right, "
        "2 row to top, 3 row to bottom, 4 depth to front "
        "and 5 depth to back. Size of paddings must be 6.");
    AddAttr<float>("value",
                   "(float, default 0.0) "
                   "The value to fill the padded areas in constant mode.")
        .SetDefault(0.0f);
    AddAttr<std::string>(
        "mode",
        "(string, default constant) "
        "Four modes: constant(default), reflect, replicate, circular.")
        .SetDefault("constant");
    AddAttr<std::string>(
        "data_format",
        "(string, default NCDHW) Only used in "
        "An optional string from: \"NDHWC\", \"NCDHW\". "
        "Defaults to \"NDHWC\". Specify the data format of the input data.")
        .SetDefault("NCDHW");
    AddComment(R"DOC(
Pad3d Operator.
Pad 3-d images according to 'paddings' and 'mode'. 
If mode is 'reflect', paddings[0] and paddings[1] must be no greater
than width-1. The height and depth dimension have the same condition.

Given that X is a channel of image from input:

X = [[[[[1, 2, 3],
     [4, 5, 6]]]]]

Case 0:

paddings = [2, 2, 1, 1, 0, 0],
mode = 'constant'
pad_value = 0

Out = [[[[[0. 0. 0. 0. 0. 0. 0.]
          [0. 0. 1. 2. 3. 0. 0.]
          [0. 0. 4. 5. 6. 0. 0.]
          [0. 0. 0. 0. 0. 0. 0.]]]]]

Case 1:

paddings = [2, 2, 1, 1, 0, 0],
mode = 'reflect'

Out = [[[[[6. 5. 4. 5. 6. 5. 4.]
          [3. 2. 1. 2. 3. 2. 1.]
          [6. 5. 4. 5. 6. 5. 4.]
          [3. 2. 1. 2. 3. 2. 1.]]]]]

Case 2:

paddings = [2, 2, 1, 1, 0, 0],
mode = 'replicate'

Out = [[[[[1. 1. 1. 2. 3. 3. 3.]
          [1. 1. 1. 2. 3. 3. 3.]
          [4. 4. 4. 5. 6. 6. 6.]
          [4. 4. 4. 5. 6. 6. 6.]]]]]

Case 3:

paddings = [2, 2, 1, 1, 0, 0],
mode = 'circular'

Out = [[[[[5. 6. 4. 5. 6. 4. 5.]
          [2. 3. 1. 2. 3. 1. 2.]
          [5. 6. 4. 5. 6. 4. 5.]
          [2. 3. 1. 2. 3. 1. 2.]]]]]

)DOC");
  }
};

class Pad3dOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Pad3d@Grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "Pad3d@Grad");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class Pad3dOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> bind) const override {
    bind->SetInput("X", this->Input("X"));
    if (this->HasInput("Paddings")) {
      bind->SetInput("Paddings", this->Input("Paddings"));
    }
    bind->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    bind->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    bind->SetAttrMap(this->Attrs());
    bind->SetType("pad3d_grad");
  }
};

template <typename T>
class Pad3dOpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    if (this->HasInput("Paddings")) {
      grad_op->SetInput("Paddings", this->Input("Paddings"));
    }
    grad_op->SetType("pad3d");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(Pad3dOpGradNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(pad3d, ops::Pad3dOp, ops::Pad3dOpMaker,
                  ops::Pad3dOpGradMaker<paddle::framework::OpDesc>,
                  ops::Pad3dOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(pad3d_grad, ops::Pad3dOpGrad,
                  ops::Pad3dOpDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::Pad3dOpDoubleGradMaker<paddle::imperative::OpBase>,
                  ops::Pad3dOpGradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(pad3d, ops::Pad3dCPUKernel<float>,
                       ops::Pad3dCPUKernel<double>, ops::Pad3dCPUKernel<int>,
                       ops::Pad3dCPUKernel<int64_t>);
REGISTER_OP_CPU_KERNEL(pad3d_grad, ops::Pad3dGradCPUKernel<float>,
                       ops::Pad3dGradCPUKernel<double>);
