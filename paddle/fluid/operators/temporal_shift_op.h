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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static HOSTDEVICE inline int GetEntryIndex(int in, int it, int ic, int ih,
                                           int iw, const int tchw,
                                           const int chw, const int hw,
                                           const int w) {
  return in * tchw + it * chw + ic * hw + ih * w + iw;
}

template <typename T>
class TemporalShiftKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    int t = ctx.Attr<int>("seg_num");
    float shift_ratio = ctx.Attr<float>("shift_ratio");

    const int nt = input->dims()[0];
    const int c = input->dims()[1];
    const int h = input->dims()[2];
    const int w = input->dims()[3];

    const int c1 = static_cast<int>(c * shift_ratio);
    const int c2 = static_cast<int>(c * 2 * shift_ratio);

    const int hw = h * w;
    const int chw = c * hw;
    const int tchw = t * chw;

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>({nt, c, h, w}, ctx.GetPlace());

    int src_it = 0;
    for (int i = 0; i < output->numel(); i++) {
      int in = i / tchw;
      int it = (i % tchw) / chw;
      int ic = (i % chw) / hw;
      int ih = (i % hw) / w;
      int iw = i % w;

      if (ic < c1) {
        src_it = it - 1;
      } else if (ic < c2) {
        src_it = it + 1;
      } else {
        src_it = it;
      }

      if (src_it < 0 || src_it >= t) {
        output_data[i] = 0;
      } else {
        int src_idx = GetEntryIndex(in, src_it, ic, ih, iw, tchw, chw, hw, w);
        output_data[i] = input_data[src_idx];
      }
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

    const int nt = output_grad->dims()[0];
    const int c = output_grad->dims()[1];
    const int h = output_grad->dims()[2];
    const int w = output_grad->dims()[3];

    const int c1 = static_cast<int>(c * shift_ratio);
    const int c2 = static_cast<int>(c * 2 * shift_ratio);

    const int hw = h * w;
    const int chw = c * hw;
    const int tchw = t * chw;

    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data =
        input_grad->mutable_data<T>({nt, c, h, w}, ctx.GetPlace());
    memset(input_grad_data, 0, input_grad->numel() * sizeof(T));

    int src_it = 0;
    for (int i = 0; i < output_grad->numel(); i++) {
      int in = i / tchw;
      int it = (i % tchw) / chw;
      int ic = (i % chw) / hw;
      int ih = (i % hw) / w;
      int iw = i % w;

      if (ic < c1) {
        src_it = it - 1;
      } else if (ic < c2) {
        src_it = it + 1;
      } else {
        src_it = it;
      }

      if (src_it >= 0 && src_it < t) {
        int src_idx = GetEntryIndex(in, src_it, ic, ih, iw, tchw, chw, hw, w);
        input_grad_data[src_idx] = output_grad_data[i];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
