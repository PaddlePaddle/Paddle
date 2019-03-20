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

#pragma once

#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
struct FindAbsMaxFunctor {
  void operator()(const DeviceContext& ctx, const T* in, const int num, T* out);
};

template <typename DeviceContext, typename T>
struct ClipAndFakeQuantFunctor {
  void operator()(const DeviceContext& ctx, const framework::Tensor& in,
                  const framework::Tensor& scale, const int bin_cnt,
                  framework::Tensor* out);
};

template <typename DeviceContext, typename T>
struct FindRangeAbsMaxFunctor {
  void operator()(const DeviceContext& ctx, const framework::Tensor& cur_scale,
                  const framework::Tensor& last_scale,
                  const framework::Tensor& iter, const int window_size,
                  framework::Tensor* scales_arr, framework::Tensor* out_scale);
};

template <typename DeviceContext, typename T>
struct FindMovingAverageAbsMaxFunctor {
  void operator()(const DeviceContext& ctx, const framework::Tensor& in_accum,
                  const framework::Tensor& in_state,
                  const framework::Tensor& cur_scale,
                  framework::Tensor* out_state, framework::Tensor* out_accum,
                  framework::Tensor* out_scale);
};

template <typename DeviceContext, typename T>
class FakeQuantizeAbsMaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto* out_scale = context.Output<framework::Tensor>("OutScale");
    T* out_s = out_scale->mutable_data<T>(context.GetPlace());

    int bit_length = context.Attr<int>("bit_length");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;

    auto& dev_ctx = context.template device_context<DeviceContext>();
    const T* in_data = in->data<T>();
    FindAbsMaxFunctor<DeviceContext, T>()(dev_ctx, in_data, in->numel(), out_s);
    ClipAndFakeQuantFunctor<DeviceContext, T>()(dev_ctx, *in, *out_scale,
                                                bin_cnt, out);
  }
};

template <typename DeviceContext, typename T>
class FakeChannelWiseQuantizeAbsMaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");

    auto* out = context.Output<framework::Tensor>("Out");
    auto* out_scales = context.Output<framework::Tensor>("OutScales");
    T* out_scales_data = out_scales->mutable_data<T>(context.GetPlace());
    out->mutable_data<T>(context.GetPlace());

    int bit_length = context.Attr<int>("bit_length");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;

    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto find_abs_max = FindAbsMaxFunctor<DeviceContext, T>();
    for (int64_t i = 0; i < in->dims()[0]; i++) {
      framework::Tensor one_channel = in->Slice(i, i + 1);
      const T* one_channel_data = one_channel.data<T>();
      find_abs_max(dev_ctx, one_channel_data, one_channel.numel(),
                   &out_scales_data[i]);
    }
    auto clip_quant = ClipAndFakeQuantFunctor<DeviceContext, T>();
    for (int64_t i = 0; i < in->dims()[0]; i++) {
      framework::Tensor one_channel_in = in->Slice(i, i + 1);
      framework::Tensor one_channel_out = out->Slice(i, i + 1);
      framework::Tensor one_channel_scale = out_scales->Slice(i, i + 1);
      clip_quant(dev_ctx, one_channel_in, one_channel_scale, bin_cnt,
                 &one_channel_out);
    }
  }
};

template <typename DeviceContext, typename T>
class FakeQuantizeRangeAbsMaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* in_scale = context.Input<framework::Tensor>("InScale");

    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    bool is_test = context.Attr<bool>("is_test");
    int bit_length = context.Attr<int>("bit_length");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;
    auto& dev_ctx = context.template device_context<DeviceContext>();

    // testing
    if (is_test) {
      ClipAndFakeQuantFunctor<DeviceContext, T>()(dev_ctx, *in, *in_scale,
                                                  bin_cnt, out);
      return;
    }

    // training
    auto* out_scale = context.Output<framework::Tensor>("OutScale");
    auto* out_scales = context.Output<framework::Tensor>("OutScales");
    auto* iter = context.Input<framework::Tensor>("Iter");

    int window_size = context.Attr<int>("window_size");
    out_scale->mutable_data<T>(context.GetPlace());

    framework::Tensor cur_scale;
    T* cur_scale_data = cur_scale.mutable_data<T>({1}, context.GetPlace());
    FindAbsMaxFunctor<DeviceContext, T>()(dev_ctx, in->data<T>(), in->numel(),
                                          cur_scale_data);
    FindRangeAbsMaxFunctor<DeviceContext, T>()(dev_ctx, cur_scale, *in_scale,
                                               *iter, window_size, out_scales,
                                               out_scale);
    ClipAndFakeQuantFunctor<DeviceContext, T>()(dev_ctx, *in, *out_scale,
                                                bin_cnt, out);
  }
};

template <typename DeviceContext, typename T>
class FakeQuantizeMovingAverageAbsMaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* in_scale = context.Input<framework::Tensor>("InScale");
    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    bool is_test = context.Attr<bool>("is_test");
    int bit_length = context.Attr<int>("bit_length");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;
    auto& dev_ctx = context.template device_context<DeviceContext>();

    // testing
    if (is_test) {
      ClipAndFakeQuantFunctor<DeviceContext, T>()(dev_ctx, *in, *in_scale,
                                                  bin_cnt, out);
      return;
    }

    // training
    auto* in_accum = context.Input<framework::Tensor>("InAccum");
    auto* in_state = context.Input<framework::Tensor>("InState");
    auto& allocator =
        platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx);
    auto cur_scale = allocator.Allocate(1 * sizeof(T));
    T* cur_scale_data = static_cast<T*>(cur_scale->ptr());

    FindAbsMaxFunctor<DeviceContext, T>()(dev_ctx, in->data<T>(), in->numel(),
                                          cur_scale_data);

    auto* out_state = context.Output<framework::Tensor>("OutState");
    auto* out_accum = context.Output<framework::Tensor>("OutAccum");
    auto* out_scale = context.Output<framework::Tensor>("OutScale");
    out_state->mutable_data<T>(context.GetPlace());
    out_accum->mutable_data<T>(context.GetPlace());
    out_scale->mutable_data<T>(context.GetPlace());
    float moving_rate = context.Attr<float>("moving_rate");

    FindMovingAverageAbsMaxFunctor<DeviceContext, T>()(
        dev_ctx, *in_accum, *in_state, cur_scale_data, moving_rate, out_state,
        out_accum, out_scale);

    ClipAndFakeQuantFunctor<DeviceContext, T>()(dev_ctx, *in, *out_scale,
                                                bin_cnt, out);
  }
};

}  // namespace operators
}  // namespace paddle
