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
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

template <typename T>
inline HOSTDEVICE T inverse(T s) {
  T eps = static_cast<T>(1e-6);
  T one = static_cast<T>(1.0);
  return s <= static_cast<T>(1e-30) ? one / (s + eps) : one / s;
}

template <typename T>
inline HOSTDEVICE T roundWithTiesToEven(T x) {
  T xLower = floor(x);
  T xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  T dLower = x - xLower;
  T dUpper = xUpper - x;
  return static_cast<T>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename T>
class QuantTensorFunctor {
 public:
  explicit QuantTensorFunctor(const T bin_cnt, const T inv_s)
      : bin_cnt_(bin_cnt), inv_s_(inv_s) {}
  HOSTDEVICE T operator()(const T x) const {
    T out = bin_cnt_ * inv_s_ * x;
    out = roundWithTiesToEven(out);
    T max_bound = bin_cnt_;
    T min_bound = -bin_cnt_ - static_cast<T>(1);
    out = out > max_bound ? max_bound : out;
    out = out < min_bound ? min_bound : out;
    return out;
  }

 private:
  T bin_cnt_;
  T inv_s_;
};

template <typename DeviceContext, typename T>
struct FindAbsMaxFunctor {
  void operator()(const DeviceContext &ctx, const T *in, const int num, T *out);
};

template <typename DeviceContext, typename T>
struct ClipAndFakeQuantFunctor {
  void operator()(const DeviceContext &ctx,
                  const phi::DenseTensor &in,
                  const phi::DenseTensor &scale,
                  const int bin_cnt,
                  const int round_type,
                  phi::DenseTensor *out);
};

template <typename DeviceContext, typename T>
struct ClipAndFakeQuantDequantFunctor {
  void operator()(const DeviceContext &ctx,
                  const phi::DenseTensor &in,
                  const phi::DenseTensor &scale,
                  const int bin_cnt,
                  int round_type,
                  phi::DenseTensor *out);
};

template <typename DeviceContext, typename T>
struct FindRangeAbsMaxFunctor {
  void operator()(const DeviceContext &ctx,
                  const phi::DenseTensor &cur_scale,
                  const phi::DenseTensor &last_scale,
                  const phi::DenseTensor &iter,
                  const int window_size,
                  phi::DenseTensor *scales_arr,
                  phi::DenseTensor *out_scale);
};

template <typename DeviceContext, typename T>
struct FindChannelAbsMaxFunctor {
  void operator()(const DeviceContext &ctx,
                  const phi::DenseTensor &in_tensor,
                  const int quant_axis,
                  T *out_abs_max);
};

template <typename DeviceContext, typename T>
struct ChannelClipAndFakeQuantFunctor {
  void operator()(const DeviceContext &ctx,
                  const phi::DenseTensor &in,
                  const phi::DenseTensor &scale,
                  const int bin_cnt,
                  const int round_type,
                  const int quant_axis,
                  phi::DenseTensor *out);
};

template <typename DeviceContext, typename T>
struct ChannelClipFakeQuantDequantFunctor {
  void operator()(const DeviceContext &ctx,
                  const phi::DenseTensor &in,
                  const phi::DenseTensor &scale,
                  const int bin_cnt,
                  int round_type,
                  const int quant_axis,
                  phi::DenseTensor *out);
};

template <typename DeviceContext, typename T>
struct FindMovingAverageAbsMaxFunctor {
  void operator()(const DeviceContext &ctx,
                  const phi::DenseTensor &in_accum,
                  const phi::DenseTensor &in_state,
                  const T *cur_scale,
                  const float rate,
                  phi::DenseTensor *out_state,
                  phi::DenseTensor *out_accum,
                  phi::DenseTensor *out_scale);
};

template <typename DeviceContext, typename T>
class FakeAbsMaxKernelBase : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<phi::DenseTensor>("X");
    auto *out = context.Output<phi::DenseTensor>("Out");
    auto *out_scale = context.Output<phi::DenseTensor>("OutScale");
    T *out_s = out_scale->mutable_data<T>(context.GetPlace());

    int bit_length = context.Attr<int>("bit_length");
    int round_type = context.Attr<int>("round_type");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;

    auto &dev_ctx = context.template device_context<DeviceContext>();
    const T *in_data = in->data<T>();
    FindAbsMaxFunctor<DeviceContext, T>()(dev_ctx, in_data, in->numel(), out_s);
    RunClipFunctor(dev_ctx, *in, *out_scale, bin_cnt, round_type, out);
  }

  virtual ~FakeAbsMaxKernelBase() = default;

 protected:
  virtual void RunClipFunctor(const DeviceContext &dev_ctx,
                              const phi::DenseTensor &in,
                              const phi::DenseTensor &scale,
                              int bin_cnt,
                              int round_type,
                              phi::DenseTensor *out) const = 0;
};

template <typename DeviceContext, typename T>
class FakeQuantizeAbsMaxKernel : public FakeAbsMaxKernelBase<DeviceContext, T> {
 protected:
  void RunClipFunctor(const DeviceContext &dev_ctx,
                      const phi::DenseTensor &in,
                      const phi::DenseTensor &scale,
                      int bin_cnt,
                      int round_type,
                      phi::DenseTensor *out) const override {
    ClipAndFakeQuantFunctor<DeviceContext, T>()(
        dev_ctx, in, scale, bin_cnt, round_type, out);
  }
};

template <typename DeviceContext, typename T>
class FakeQuantizeDequantizeAbsMaxKernel
    : public FakeAbsMaxKernelBase<DeviceContext, T> {
 protected:
  void RunClipFunctor(const DeviceContext &dev_ctx,
                      const phi::DenseTensor &in,
                      const phi::DenseTensor &scale,
                      int bin_cnt,
                      int round_type,
                      phi::DenseTensor *out) const override {
    ClipAndFakeQuantDequantFunctor<DeviceContext, T>()(
        dev_ctx, in, scale, bin_cnt, round_type, out);
  }
};

template <typename DeviceContext, typename T>
class FakeChannelWiseQuantizeAbsMaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<phi::DenseTensor>("X");

    auto *out = context.Output<phi::DenseTensor>("Out");
    auto *out_scale = context.Output<phi::DenseTensor>("OutScale");
    out->mutable_data<T>(context.GetPlace());

    int bit_length = context.Attr<int>("bit_length");
    int round_type = context.Attr<int>("round_type");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;
    int quant_axis = context.Attr<int>("quant_axis");
    bool is_test = context.Attr<bool>("is_test");

    auto &dev_ctx = context.template device_context<DeviceContext>();
    if (!is_test) {
      T *out_scale_data = out_scale->mutable_data<T>(context.GetPlace());
      FindChannelAbsMaxFunctor<DeviceContext, T>()(
          dev_ctx, *in, quant_axis, out_scale_data);
    }
    ChannelClipAndFakeQuantFunctor<DeviceContext, T>()(
        dev_ctx, *in, *out_scale, bin_cnt, round_type, quant_axis, out);
  }
};

template <typename DeviceContext, typename T>
class FakeChannelWiseQuantizeDequantizeAbsMaxKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<phi::DenseTensor>("X");
    auto *out = context.Output<phi::DenseTensor>("Out");
    auto *out_scale = context.Output<phi::DenseTensor>("OutScale");
    T *out_scale_data = out_scale->mutable_data<T>(context.GetPlace());
    auto &dev_ctx = context.template device_context<DeviceContext>();
    out->mutable_data<T>(dev_ctx.GetPlace());

    int bit_length = context.Attr<int>("bit_length");
    int round_type = context.Attr<int>("round_type");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;
    int quant_axis = context.Attr<int>("quant_axis");

    FindChannelAbsMaxFunctor<DeviceContext, T>()(
        dev_ctx, *in, quant_axis, out_scale_data);

    ChannelClipFakeQuantDequantFunctor<DeviceContext, T>()(
        dev_ctx, *in, *out_scale, bin_cnt, round_type, quant_axis, out);
  }
};

template <typename DeviceContext, typename T>
class FakeQuantizeRangeAbsMaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<phi::DenseTensor>("X");
    auto *in_scale = context.Input<phi::DenseTensor>("InScale");

    auto *out = context.Output<phi::DenseTensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    bool is_test = context.Attr<bool>("is_test");
    int bit_length = context.Attr<int>("bit_length");
    int round_type = context.Attr<int>("round_type");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;
    auto &dev_ctx = context.template device_context<DeviceContext>();

    // testing
    if (is_test) {
      ClipAndFakeQuantFunctor<DeviceContext, T>()(
          dev_ctx, *in, *in_scale, bin_cnt, round_type, out);
      return;
    }

    // training
    auto *out_scale = context.Output<phi::DenseTensor>("OutScale");
    auto *out_scales = context.Output<phi::DenseTensor>("OutScales");
    auto *iter = context.Input<phi::DenseTensor>("Iter");

    int window_size = context.Attr<int>("window_size");
    out_scale->mutable_data<T>(context.GetPlace());

    phi::DenseTensor cur_scale;
    T *cur_scale_data = cur_scale.mutable_data<T>({1}, context.GetPlace());
    FindAbsMaxFunctor<DeviceContext, T>()(
        dev_ctx, in->data<T>(), in->numel(), cur_scale_data);
    FindRangeAbsMaxFunctor<DeviceContext, T>()(dev_ctx,
                                               cur_scale,
                                               *in_scale,
                                               *iter,
                                               window_size,
                                               out_scales,
                                               out_scale);
    ClipAndFakeQuantFunctor<DeviceContext, T>()(
        dev_ctx, *in, *out_scale, bin_cnt, round_type, out);
  }
};

template <typename DeviceContext, typename T>
class FakeMovingAverageAbsMaxKernelBase : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<phi::DenseTensor>("X");
    auto *in_scale = context.Input<phi::DenseTensor>("InScale");
    auto *out = context.Output<phi::DenseTensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    bool is_test = context.Attr<bool>("is_test");
    int bit_length = context.Attr<int>("bit_length");
    int round_type = context.Attr<int>("round_type");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;
    auto &dev_ctx = context.template device_context<DeviceContext>();

    // testing
    if (is_test) {
      RunClipFunctor(dev_ctx, *in, *in_scale, bin_cnt, round_type, out);
      return;
    }

    // training
    auto *in_accum = context.Input<phi::DenseTensor>("InAccum");
    auto *in_state = context.Input<phi::DenseTensor>("InState");

    phi::DenseTensor tmp_scale;
    tmp_scale.Resize(phi::make_dim(1));
    T *cur_scale_data = dev_ctx.template Alloc<T>(&tmp_scale);

    FindAbsMaxFunctor<DeviceContext, T>()(
        dev_ctx, in->data<T>(), in->numel(), cur_scale_data);

    auto *out_state = context.Output<phi::DenseTensor>("OutState");
    auto *out_accum = context.Output<phi::DenseTensor>("OutAccum");
    auto *out_scale = context.Output<phi::DenseTensor>("OutScale");
    out_state->mutable_data<T>(context.GetPlace());
    out_accum->mutable_data<T>(context.GetPlace());
    out_scale->mutable_data<T>(context.GetPlace());
    float moving_rate = context.Attr<float>("moving_rate");

    FindMovingAverageAbsMaxFunctor<DeviceContext, T>()(dev_ctx,
                                                       *in_accum,
                                                       *in_state,
                                                       cur_scale_data,
                                                       moving_rate,
                                                       out_state,
                                                       out_accum,
                                                       out_scale);

    RunClipFunctor(dev_ctx, *in, *out_scale, bin_cnt, round_type, out);
  }

  virtual ~FakeMovingAverageAbsMaxKernelBase() = default;

 protected:
  virtual void RunClipFunctor(const DeviceContext &dev_ctx,
                              const phi::DenseTensor &in,
                              const phi::DenseTensor &in_scale,
                              int bin_cnt,
                              int round_type,
                              phi::DenseTensor *out) const = 0;
};

template <typename DeviceContext, typename T>
class FakeQuantizeMovingAverageAbsMaxKernel
    : public FakeMovingAverageAbsMaxKernelBase<DeviceContext, T> {
 protected:
  void RunClipFunctor(const DeviceContext &dev_ctx,
                      const phi::DenseTensor &in,
                      const phi::DenseTensor &in_scale,
                      int bin_cnt,
                      int round_type,
                      phi::DenseTensor *out) const override {
    ClipAndFakeQuantFunctor<DeviceContext, T>()(
        dev_ctx, in, in_scale, bin_cnt, round_type, out);
  }
};

template <typename DeviceContext, typename T>
class FakeQuantizeDequantizeMovingAverageAbsMaxKernel
    : public FakeMovingAverageAbsMaxKernelBase<DeviceContext, T> {
 protected:
  void RunClipFunctor(const DeviceContext &dev_ctx,
                      const phi::DenseTensor &in,
                      const phi::DenseTensor &in_scale,
                      int bin_cnt,
                      int round_type,
                      phi::DenseTensor *out) const override {
    ClipAndFakeQuantDequantFunctor<DeviceContext, T>()(
        dev_ctx, in, in_scale, bin_cnt, round_type, out);
  }
};

template <typename DeviceContext, typename T>
class MovingAverageAbsMaxScaleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<phi::DenseTensor>("X");
    auto &dev_ctx = context.template device_context<DeviceContext>();

    if (context.HasOutput("Out")) {
      auto *out = context.Output<phi::DenseTensor>("Out");
      out->mutable_data<T>(context.GetPlace());
      framework::TensorCopy(*in, context.GetPlace(), dev_ctx, out);
    }

    bool is_test = context.Attr<bool>("is_test");
    // testing
    if (is_test) {
      return;
    }

    // training
    auto *in_accum = context.Input<phi::DenseTensor>("InAccum");
    auto *in_state = context.Input<phi::DenseTensor>("InState");
    phi::DenseTensor tmp_scale;
    tmp_scale.Resize(phi::make_dim(1));
    T *cur_scale_data = dev_ctx.template Alloc<T>(&tmp_scale);

    FindAbsMaxFunctor<DeviceContext, T>()(
        dev_ctx, in->data<T>(), in->numel(), cur_scale_data);

    auto *out_state = context.Output<phi::DenseTensor>("OutState");
    auto *out_accum = context.Output<phi::DenseTensor>("OutAccum");
    auto *out_scale = context.Output<phi::DenseTensor>("OutScale");
    out_state->mutable_data<T>(context.GetPlace());
    out_accum->mutable_data<T>(context.GetPlace());
    out_scale->mutable_data<T>(context.GetPlace());
    float moving_rate = context.Attr<float>("moving_rate");

    FindMovingAverageAbsMaxFunctor<DeviceContext, T>()(dev_ctx,
                                                       *in_accum,
                                                       *in_state,
                                                       cur_scale_data,
                                                       moving_rate,
                                                       out_state,
                                                       out_accum,
                                                       out_scale);
  }
};

template <typename DeviceContext, typename T>
class StrightThroughEstimatorGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *d_out =
        context.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto x_grad_name = framework::GradVarName("X");
    auto *d_x = context.Output<framework::LoDTensor>(x_grad_name);
    PADDLE_ENFORCE_NOT_NULL(d_x,
                            platform::errors::PreconditionNotMet(
                                "StrightThroughEstimatorGradKernel "
                                "doesn't have the output named %s.",
                                x_grad_name));

    // Initialize dx as same as d_out
    d_x->mutable_data<T>(context.GetPlace());
    framework::TensorCopy(*d_out, context.GetPlace(), d_x);
  }
};

}  // namespace operators
}  // namespace paddle
