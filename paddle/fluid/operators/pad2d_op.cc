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
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
void Pad2DConstNCHW(const T* in_data, const int num, const int channels,
                    const int in_height, const int in_width,
                    const int out_height, const int out_width,
                    const int pad_top, const int pad_left, T value,
                    T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          int in_h = out_h - pad_top;
          int in_w = out_w - pad_left;
          out_data[out_h * out_width + out_w] =
              (in_h < 0 || in_w < 0 || in_h >= in_height || in_w >= in_width)
                  ? value
                  : in_data[in_h * in_width + in_w];
        }
      }
      in_data += in_height * in_width;
      out_data += out_height * out_width;
    }
  }
}

template <typename T>
void Pad2DConstNHWC(const T* in_data, const int num, const int channels,
                    const int in_height, const int in_width,
                    const int out_height, const int out_width,
                    const int pad_top, const int pad_left, T value,
                    T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int out_h = 0; out_h < out_height; ++out_h) {
      for (int out_w = 0; out_w < out_width; ++out_w) {
        int in_h = out_h - pad_top;
        int in_w = out_w - pad_left;
        const int out_index = (out_h * out_width + out_w) * channels;
        if (in_h < 0 || in_w < 0 || in_h >= in_height || in_w >= in_width) {
          for (int c = 0; c < channels; ++c) {
            out_data[out_index + c] = value;
          }
        } else {
          const int in_index = (in_h * in_width + in_w) * channels;
          for (int c = 0; c < channels; ++c) {
            out_data[out_index + c] = in_data[in_index + c];
          }
        }
      }
    }
    in_data += in_height * in_width * channels;
    out_data += out_height * out_width * channels;
  }
}

template <typename T>
void Pad2DReflectNCHW(const T* in_data, const int num, const int channels,
                      const int in_height, const int in_width,
                      const int out_height, const int out_width,
                      const int pad_top, const int pad_left, T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          int in_h = out_h - pad_top;
          int in_w = out_w - pad_left;
          in_h = std::max(in_h, -in_h);  // reflect by 0
          in_h =
              std::min(in_h, 2 * in_height - in_h - 2);  // reflect by in_height
          in_w = std::max(in_w, -in_w);                  // reflect by 0
          in_w =
              std::min(in_w, 2 * in_width - in_w - 2);  // reflect by in_width
          out_data[out_h * out_width + out_w] = in_data[in_h * in_width + in_w];
        }
      }
      in_data += in_height * in_width;
      out_data += out_height * out_width;
    }
  }
}

template <typename T>
void Pad2DReflectNHWC(const T* in_data, const int num, const int channels,
                      const int in_height, const int in_width,
                      const int out_height, const int out_width,
                      const int pad_top, const int pad_left, T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int out_h = 0; out_h < out_height; ++out_h) {
      for (int out_w = 0; out_w < out_width; ++out_w) {
        const int out_index = (out_h * out_width + out_w) * channels;
        int in_h = out_h - pad_top;
        int in_w = out_w - pad_left;
        in_h = std::max(in_h, -in_h);
        in_h = std::min(in_h, 2 * in_height - in_h - 2);
        in_w = std::max(in_w, -in_w);
        in_w = std::min(in_w, 2 * in_width - in_w - 2);
        const int in_index = (in_h * in_width + in_w) * channels;

        for (int c = 0; c < channels; ++c) {
          out_data[out_index + c] = in_data[in_index + c];
        }
      }
    }
    in_data += in_height * in_width * channels;
    out_data += out_height * out_width * channels;
  }
}

template <typename T>
void Pad2DEdgeNCHW(const T* in_data, const int num, const int channels,
                   const int in_height, const int in_width,
                   const int out_height, const int out_width, const int pad_top,
                   const int pad_left, T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
          int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));
          out_data[out_h * out_width + out_w] = in_data[in_h * in_width + in_w];
        }
      }
      in_data += in_height * in_width;
      out_data += out_height * out_width;
    }
  }
}

template <typename T>
void Pad2DEdgeNHWC(const T* in_data, const int num, const int channels,
                   const int in_height, const int in_width,
                   const int out_height, const int out_width, const int pad_top,
                   const int pad_left, T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int out_h = 0; out_h < out_height; ++out_h) {
      for (int out_w = 0; out_w < out_width; ++out_w) {
        const int out_index = (out_h * out_width + out_w) * channels;
        int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
        int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));
        const int in_index = (in_h * in_width + in_w) * channels;
        for (int c = 0; c < channels; ++c) {
          out_data[out_index + c] = in_data[in_index + c];
        }
      }
    }
    in_data += in_height * in_width * channels;
    out_data += out_height * out_width * channels;
  }
}

template <typename T>
void Pad2DGradConstNCHW(T* d_in_data, const int num, const int channels,
                        const int in_height, const int in_width,
                        const int out_height, const int out_width,
                        const int pad_top, const int pad_left,
                        const T* d_out_data) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          int in_h = out_h - pad_top;
          int in_w = out_w - pad_left;
          if (!(in_h < 0 || in_w < 0 || in_h >= in_height ||
                in_w >= in_width)) {
            d_in_data[in_h * in_width + in_w] =
                d_out_data[out_h * out_width + out_w];
          }
        }
      }
      d_in_data += in_height * in_width;
      d_out_data += out_height * out_width;
    }
  }
}

template <typename T>
void Pad2DGradConstNHWC(T* d_in_data, const int num, const int channels,
                        const int in_height, const int in_width,
                        const int out_height, const int out_width,
                        const int pad_top, const int pad_left,
                        const T* d_out_data) {
  for (int n = 0; n < num; ++n) {
    for (int out_h = 0; out_h < out_height; ++out_h) {
      for (int out_w = 0; out_w < out_width; ++out_w) {
        int in_h = out_h - pad_top;
        int in_w = out_w - pad_left;
        const int out_index = (out_h * out_width + out_w) * channels;
        if (!(in_h < 0 || in_w < 0 || in_h >= in_height || in_w >= in_width)) {
          const int in_index = (in_h * in_width + in_w) * channels;
          for (int c = 0; c < channels; ++c) {
            d_in_data[in_index + c] = d_out_data[out_index + c];
          }
        }
      }
    }
    d_in_data += in_height * in_width * channels;
    d_out_data += out_height * out_width * channels;
  }
}

template <typename T>
void Pad2DGradReflectNCHW(T* d_in_data, const int num, const int channels,
                          const int in_height, const int in_width,
                          const int out_height, const int out_width,
                          const int pad_top, const int pad_left,
                          const T* d_out_data) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          int in_h = out_h - pad_top;
          int in_w = out_w - pad_left;
          in_h = std::max(in_h, -in_h);  // reflect over 0
          in_h = std::min(in_h,
                          2 * in_height - in_h - 2);  // reflect over in_height
          in_w = std::max(in_w, -in_w);               // reflect over 0
          in_w =
              std::min(in_w, 2 * in_width - in_w - 2);  // reflect over in_width
          d_in_data[in_h * in_width + in_w] +=
              d_out_data[out_h * out_width + out_w];
        }
      }
      d_in_data += in_height * in_width;
      d_out_data += out_height * out_width;
    }
  }
}

template <typename T>
void Pad2DGradReflectNHWC(T* d_in_data, const int num, const int channels,
                          const int in_height, const int in_width,
                          const int out_height, const int out_width,
                          const int pad_top, const int pad_left,
                          const T* d_out_data) {
  for (int n = 0; n < num; ++n) {
    for (int out_h = 0; out_h < out_height; ++out_h) {
      for (int out_w = 0; out_w < out_width; ++out_w) {
        const int out_index = (out_h * out_width + out_w) * channels;
        int in_h = out_h - pad_top;
        int in_w = out_w - pad_left;
        in_h = std::max(in_h, -in_h);
        in_h = std::min(in_h, 2 * in_height - in_h - 2);
        in_w = std::max(in_w, -in_w);
        in_w = std::min(in_w, 2 * in_width - in_w - 2);
        const int in_index = (in_h * in_width + in_w) * channels;
        for (int c = 0; c < channels; ++c) {
          d_in_data[in_index + c] += d_out_data[out_index + c];
        }
      }
    }
    d_in_data += in_height * in_width * channels;
    d_out_data += out_height * out_width * channels;
  }
}

template <typename T>
void Pad2DGradEdgeNCHW(T* d_in_data, const int num, const int channels,
                       const int in_height, const int in_width,
                       const int out_height, const int out_width,
                       const int pad_top, const int pad_left,
                       const T* d_out_data) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
          int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));
          d_in_data[in_h * in_width + in_w] +=
              d_out_data[out_h * out_width + out_w];
        }
      }
      d_in_data += in_height * in_width;
      d_out_data += out_height * out_width;
    }
  }
}

template <typename T>
void Pad2DGradEdgeNHWC(T* d_in_data, const int num, const int channels,
                       const int in_height, const int in_width,
                       const int out_height, const int out_width,
                       const int pad_top, const int pad_left,
                       const T* d_out_data) {
  for (int n = 0; n < num; ++n) {
    for (int out_h = 0; out_h < out_height; ++out_h) {
      for (int out_w = 0; out_w < out_width; ++out_w) {
        const int out_index = (out_h * out_width + out_w) * channels;
        int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
        int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));
        const int in_index = (in_h * in_width + in_w) * channels;
        for (int c = 0; c < channels; ++c) {
          d_in_data[in_index + c] += d_out_data[out_index + c];
        }
      }
    }
    d_in_data += in_height * in_width * channels;
    d_out_data += out_height * out_width * channels;
  }
}

static inline void GetPaddings(int* paddings,
                               const framework::ExecutionContext& context) {
  auto* paddings_t = context.Input<Tensor>("Paddings");
  if (paddings_t) {
    auto paddings_data = paddings_t->data<int>();
    paddings[0] = paddings_data[0];
    paddings[1] = paddings_data[1];
    paddings[2] = paddings_data[2];
    paddings[3] = paddings_data[3];
  } else {
    auto pads = context.Attr<std::vector<int>>("paddings");
    std::copy(pads.begin(), pads.end(), paddings);
  }
}

template <typename T>
class Pad2dCPUKernel : public framework::OpKernel<T> {
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
    if (data_format == "NCHW") {
      out->Resize({in_dims[0], in_dims[1], in_dims[2] + pads[0] + pads[1],
                   in_dims[3] + pads[2] + pads[3]});
    } else {
      out->Resize({in_dims[0], in_dims[1] + pads[0] + pads[1],
                   in_dims[2] + pads[2] + pads[3], in_dims[3]});
    }
    auto out_dims = out->dims();
    T* out_data = out->mutable_data<T>(context.GetPlace());

    const int pad_top = pads[0];
    const int pad_left = pads[2];
    const int num = in_dims[0];
    if (data_format == "NCHW") {
      const int channels = in_dims[1];
      const int in_height = in_dims[2];
      const int in_width = in_dims[3];
      const int out_height = out_dims[2];
      const int out_width = out_dims[3];
      if (mode == "reflect") {
        Pad2DReflectNCHW(in_data, num, channels, in_height, in_width,
                         out_height, out_width, pad_top, pad_left, out_data);
      } else if (mode == "edge") {
        Pad2DEdgeNCHW(in_data, num, channels, in_height, in_width, out_height,
                      out_width, pad_top, pad_left, out_data);
      } else {
        Pad2DConstNCHW(in_data, num, channels, in_height, in_width, out_height,
                       out_width, pad_top, pad_left, value, out_data);
      }
    } else {
      const int channels = in_dims[3];
      const int in_height = in_dims[1];
      const int in_width = in_dims[2];
      const int out_height = out_dims[1];
      const int out_width = out_dims[2];
      if (mode == "reflect") {
        Pad2DReflectNHWC(in_data, num, channels, in_height, in_width,
                         out_height, out_width, pad_top, pad_left, out_data);
      } else if (mode == "edge") {
        Pad2DEdgeNHWC(in_data, num, channels, in_height, in_width, out_height,
                      out_width, pad_top, pad_left, out_data);
      } else {
        Pad2DConstNHWC(in_data, num, channels, in_height, in_width, out_height,
                       out_width, pad_top, pad_left, value, out_data);
      }
    }
  }
};

template <typename T>
class Pad2dGradCPUKernel : public framework::OpKernel<T> {
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
    pten::funcs::SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(context.template device_context<platform::CPUDeviceContext>(),
             d_in, static_cast<T>(0));
    const int pad_top = pads[0];
    const int pad_left = pads[2];
    const int num = d_in_dims[0];
    if (data_format == "NCHW") {
      const int channels = d_in_dims[1];
      const int in_height = d_in_dims[2];
      const int in_width = d_in_dims[3];
      const int out_height = d_out_dims[2];
      const int out_width = d_out_dims[3];
      if (mode == "reflect") {
        Pad2DGradReflectNCHW(d_in_data, num, channels, in_height, in_width,
                             out_height, out_width, pad_top, pad_left,
                             d_out_data);
      } else if (mode == "edge") {
        Pad2DGradEdgeNCHW(d_in_data, num, channels, in_height, in_width,
                          out_height, out_width, pad_top, pad_left, d_out_data);
      } else {
        Pad2DGradConstNCHW(d_in_data, num, channels, in_height, in_width,
                           out_height, out_width, pad_top, pad_left,
                           d_out_data);
      }
    } else {
      const int channels = d_in_dims[3];
      const int in_height = d_in_dims[1];
      const int in_width = d_in_dims[2];
      const int out_height = d_out_dims[1];
      const int out_width = d_out_dims[2];
      if (mode == "reflect") {
        Pad2DGradReflectNHWC(d_in_data, num, channels, in_height, in_width,
                             out_height, out_width, pad_top, pad_left,
                             d_out_data);
      } else if (mode == "edge") {
        Pad2DGradEdgeNHWC(d_in_data, num, channels, in_height, in_width,
                          out_height, out_width, pad_top, pad_left, d_out_data);
      } else {
        Pad2DGradConstNHWC(d_in_data, num, channels, in_height, in_width,
                           out_height, out_width, pad_top, pad_left,
                           d_out_data);
      }
    }
  }
};

class Pad2dOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Pad2d");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Pad2d");

    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dim.size(), 4,
                      platform::errors::InvalidArgument(
                          "The size of Input(X)'s dimension should be equal to "
                          "4, but received %d. ",
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
        PADDLE_ENFORCE_EQ(paddings_dim[0], 4,
                          platform::errors::InvalidArgument(
                              "Shape of Input(Paddings) should be equal to "
                              "[4], but received [%d].",
                              paddings_dim[0]));
      }
      out_dims[1] = x_dim[1];
      out_dims[2] = x_dim[2];
      out_dims[3] = x_dim[3];
    } else {
      auto paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
      PADDLE_ENFORCE_EQ(
          paddings.size(), 4,
          platform::errors::InvalidArgument(
              "Size of paddings should be equal to 4, but received %d.",
              static_cast<int>(paddings.size())));
      if (data_format == "NCHW") {
        out_dims[1] = x_dim[1];  // channel
        out_dims[2] = ((!ctx->IsRuntime()) && (x_dim[2] < 0))
                          ? x_dim[2]
                          : (x_dim[2] + paddings[0] + paddings[1]);  // height
        out_dims[3] = ((!ctx->IsRuntime()) && (x_dim[3] < 0))
                          ? x_dim[3]
                          : (x_dim[3] + paddings[2] + paddings[3]);  // width
      } else {                                                       // NHWC
        out_dims[3] = x_dim[3];                                      // channel
        out_dims[1] = ((!ctx->IsRuntime()) && (x_dim[1] < 0))
                          ? x_dim[1]
                          : (x_dim[1] + paddings[0] + paddings[1]);  // height
        out_dims[2] = ((!ctx->IsRuntime()) && (x_dim[2] < 0))
                          ? x_dim[2]
                          : (x_dim[2] + paddings[2] + paddings[3]);  // width
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

class Pad2dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of pad2d op. "
             "The input should be a 4-D tensor with formate NCHW or NHWC.");
    AddOutput("Out",
              "The output of pad2d op. "
              "A tensor with the same shape as X.");
    AddInput("Paddings",
             "A 1-D tensor to describe the padding rules."
             "paddings=[0, 1, 2, 3] means "
             "padding 0 row to top, 1 row to bottom, 2 columns to left "
             "and 3 columns to right. Size of paddings must be 4.")
        .AsDispensable();
    AddAttr<std::vector<int>>(
        "paddings",
        "(vector<int>) "
        "A list<int> to describe the padding rules."
        "paddings=[0, 1, 2, 3] means "
        "padding 0 row to top, 1 row to bottom, 2 columns to left "
        "and 3 columns to right. Size of paddings must be 4.");
    AddAttr<float>("pad_value",
                   "(float, default 0.0) "
                   "The value to fill the padded areas in constant mode.")
        .SetDefault(0.0f);
    AddAttr<std::string>("mode",
                         "(float, default constant) "
                         "Three modes: constant(default), reflect, edge.")
        .SetDefault("constant");
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the input data.")
        .SetDefault("NCHW");
    AddComment(R"DOC(
Pad2d Operator.
Pad 2-d images according to 'paddings' and 'mode'. 
If mode is 'reflect', paddings[0] and paddings[1] must be no greater
than height-1. And the width dimension has the same condition.

Given that X is a channel of image from input:

X = [[1, 2, 3],
     [4, 5, 6]]

Case 0:

paddings = [0, 1, 2, 3],
mode = 'constant'
pad_value = 0

Out = [[0, 0, 1, 2, 3, 0, 0, 0]
       [0, 0, 4, 5, 6, 0, 0, 0]
       [0, 0, 0, 0, 0, 0, 0, 0]]

Case 1:

paddings = [0, 1, 2, 1],
mode = 'reflect'

Out = [[3, 2, 1, 2, 3, 2]
       [6, 5, 4, 5, 6, 5]
       [3, 2, 1, 2, 3, 2]]

Case 2:

paddings = [0, 1, 2, 1],
mode = 'edge'

Out = [[1, 1, 1, 2, 3, 3]
       [4, 4, 4, 5, 6, 6]
       [4, 4, 4, 5, 6, 6]]
)DOC");
  }
};

class Pad2dOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Pad2d@Grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "Pad2d@Grad");

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
class Pad2dOpGradMaker : public framework::SingleGradOpMaker<T> {
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
    bind->SetType("pad2d_grad");
  }
};

// TODO(zjl): Paddings can also be skipped!
DECLARE_NO_NEED_BUFFER_VARS_INFERER(Pad2dOpGradNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(pad2d, ops::Pad2dOp, ops::Pad2dOpMaker,
                  ops::Pad2dOpGradMaker<paddle::framework::OpDesc>,
                  ops::Pad2dOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(pad2d_grad, ops::Pad2dOpGrad,
                  ops::Pad2dOpGradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(pad2d, ops::Pad2dCPUKernel<float>,
                       ops::Pad2dCPUKernel<double>, ops::Pad2dCPUKernel<int>,
                       ops::Pad2dCPUKernel<int64_t>);
REGISTER_OP_CPU_KERNEL(pad2d_grad, ops::Pad2dGradCPUKernel<float>,
                       ops::Pad2dGradCPUKernel<double>);
