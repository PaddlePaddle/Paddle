/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;

template <typename T>
void TemporalShiftFwNCHW(const T* input, T* output, const int ntchw,
                         const int tchw, const int chw, const int hw,
                         const int t, const int c1, const int c2) {
  int src_it = 0;
  for (int i = 0; i < ntchw; i++) {
    int it = (i % tchw) / chw;
    int ic = (i % chw) / hw;

    if (ic < c1) {
      src_it = it - 1;
    } else if (ic < c2) {
      src_it = it + 1;
    } else {
      src_it = it;
    }

    if (src_it < 0 || src_it >= t) {
      output[i] = 0;
    } else {
      output[i] = input[i + (src_it - it) * chw];
    }
  }
}

template <typename T>
void TemporalShiftFwNHWC(const T* input, T* output, const int nthwc,
                         const int thwc, const int hwc, const int t,
                         const int c, const int c1, const int c2) {
  int src_it = 0;
  for (int i = 0; i < nthwc; i++) {
    int it = (i % thwc) / hwc;
    int ic = i % c;

    if (ic < c1) {
      src_it = it - 1;
    } else if (ic < c2) {
      src_it = it + 1;
    } else {
      src_it = it;
    }

    if (src_it < 0 || src_it >= t) {
      output[i] = 0;
    } else {
      output[i] = input[i + (src_it - it) * hwc];
    }
  }
}

template <typename T>
void TemporalShiftBwNCHW(const T* output_grad, T* input_grad, const int ntchw,
                         const int tchw, const int chw, const int hw,
                         const int t, const int c1, const int c2) {
  int src_it = 0;
  for (int i = 0; i < ntchw; i++) {
    int it = (i % tchw) / chw;
    int ic = (i % chw) / hw;

    if (ic < c1) {
      src_it = it + 1;
    } else if (ic < c2) {
      src_it = it - 1;
    } else {
      src_it = it;
    }

    if (src_it >= 0 && src_it < t) {
      input_grad[i] = output_grad[i + (src_it - it) * chw];
    } else {
      input_grad[i] = 0;
    }
  }
}

template <typename T>
void TemporalShiftBwNHWC(const T* output_grad, T* input_grad, const int nthwc,
                         const int thwc, const int hwc, const int t,
                         const int c, const int c1, const int c2) {
  int src_it = 0;
  for (int i = 0; i < nthwc; i++) {
    int it = (i % thwc) / hwc;
    int ic = i % c;

    if (ic < c1) {
      src_it = it + 1;
    } else if (ic < c2) {
      src_it = it - 1;
    } else {
      src_it = it;
    }

    if (src_it >= 0 && src_it < t) {
      input_grad[i] = output_grad[i + (src_it - it) * hwc];
    } else {
      input_grad[i] = 0;
    }
  }
}

template <typename T>
class TemporalShiftKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    int t = ctx.Attr<int>("seg_num");
    float shift_ratio = ctx.Attr<float>("shift_ratio");
    const std::string data_format_str = ctx.Attr<std::string>("data_format");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_format_str);

    const int nt = input->dims()[0];
    const int c = (data_layout == DataLayout::kNCHW ? input->dims()[1]
                                                    : input->dims()[3]);
    const int h = (data_layout == DataLayout::kNCHW ? input->dims()[2]
                                                    : input->dims()[1]);
    const int w = (data_layout == DataLayout::kNCHW ? input->dims()[3]
                                                    : input->dims()[2]);

    const int hw = h * w;
    const int chw = c * hw;
    const int tchw = t * chw;
    const int ntchw = nt * chw;

    const int c1 = static_cast<int>(c * shift_ratio);
    const int c2 = static_cast<int>(c * 2 * shift_ratio);

    framework::DDim out_dims =
        (data_layout == DataLayout::kNCHW ? phi::make_ddim({nt, c, h, w})
                                          : phi::make_ddim({nt, h, w, c}));
    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(out_dims, ctx.GetPlace());

    if (data_layout == DataLayout::kNCHW) {
      TemporalShiftFwNCHW<T>(input_data, output_data, ntchw, tchw, chw, hw, t,
                             c1, c2);
    } else {
      TemporalShiftFwNHWC<T>(input_data, output_data, ntchw, tchw, chw, t, c,
                             c1, c2);
    }
  }
};

template <typename T>
class TemporalShiftGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    int t = ctx.Attr<int>("seg_num");
    float shift_ratio = ctx.Attr<float>("shift_ratio");
    const std::string data_format_str = ctx.Attr<std::string>("data_format");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_format_str);

    const int nt = output_grad->dims()[0];
    const int c = (data_layout == DataLayout::kNCHW ? output_grad->dims()[1]
                                                    : output_grad->dims()[3]);
    const int h = (data_layout == DataLayout::kNCHW ? output_grad->dims()[2]
                                                    : output_grad->dims()[1]);
    const int w = (data_layout == DataLayout::kNCHW ? output_grad->dims()[3]
                                                    : output_grad->dims()[2]);

    const int hw = h * w;
    const int chw = c * hw;
    const int tchw = t * chw;
    const int ntchw = nt * chw;

    const int c1 = static_cast<int>(c * shift_ratio);
    const int c2 = static_cast<int>(c * 2 * shift_ratio);

    framework::DDim in_grad_dims =
        (data_layout == DataLayout::kNCHW ? phi::make_ddim({nt, c, h, w})
                                          : phi::make_ddim({nt, h, w, c}));
    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data =
        input_grad->mutable_data<T>(in_grad_dims, ctx.GetPlace());

    if (data_layout == DataLayout::kNCHW) {
      TemporalShiftBwNCHW<T>(output_grad_data, input_grad_data, ntchw, tchw,
                             chw, hw, t, c1, c2);
    } else {
      TemporalShiftBwNHWC<T>(output_grad_data, input_grad_data, ntchw, tchw,
                             chw, t, c, c1, c2);
    }
  }
};

}  // namespace operators
}  // namespace paddle
