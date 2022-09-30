// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj.h"
#include "paddle/phi/kernels/funcs/frame_functor.h"
#include "paddle/phi/kernels/funcs/padding.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class StftKernel : public framework::OpKernel<T> {
 public:
  /*
    Batch Signals (N, T) -> Frames (N, n_fft, num_frames) -> FFTR2C -> (N,
    n_fft/2 + 1, num_frames) or (N, n_fft, num_frames)
  */
  void Compute(const framework::ExecutionContext& ctx) const override {
    using C = paddle::platform::complex<T>;
    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* window = ctx.Input<phi::DenseTensor>("Window");
    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    out->mutable_data<C>(ctx.GetPlace());

    const size_t x_rank = x->dims().size();
    const size_t out_rank = out->dims().size();

    const int n_fft = ctx.Attr<int>("n_fft");
    const int hop_length = ctx.Attr<int>("hop_length");
    const bool normalized = ctx.Attr<bool>("normalized");
    const bool onesided = ctx.Attr<bool>("onesided");

    const int n_frames = out->dims()[out_rank - 1];
    const int seq_length = x->dims()[x_rank - 1];

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    std::vector<int64_t> axes = {1};

    // Frame
    Tensor frames;
    framework::DDim frames_dims(out->dims());
    frames_dims.at(axes.back()) = n_fft;
    frames.mutable_data<T>(frames_dims, ctx.GetPlace());
    phi::funcs::FrameFunctor<DeviceContext, T>()(dev_ctx,
                                                 x,
                                                 &frames,
                                                 seq_length,
                                                 n_fft,
                                                 n_frames,
                                                 hop_length,
                                                 /*is_grad*/ false);

    // Window
    Tensor frames_w;
    frames_w.mutable_data<T>(frames_dims, ctx.GetPlace());
    ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
        ctx, &frames, window, axes.back(), MulFunctor<T>(), &frames_w);

    // FFTR2C
    phi::funcs::FFTNormMode normalization;
    if (normalized) {
      normalization = phi::funcs::get_norm_from_string("ortho", true);
    } else {
      normalization = phi::funcs::get_norm_from_string("backward", true);
    }
    phi::funcs::FFTR2CFunctor<DeviceContext, T, C> fft_r2c_func;

    if (onesided) {
      fft_r2c_func(dev_ctx, frames_w, out, axes, normalization, true);
    } else {
      framework::DDim onesided_dims(out->dims());
      const int64_t onesided_axis_size = out->dims().at(axes.back()) / 2 + 1;
      onesided_dims.at(axes.back()) = onesided_axis_size;
      Tensor onesided_out;
      onesided_out.mutable_data<C>(onesided_dims, ctx.GetPlace());
      fft_r2c_func(dev_ctx, frames_w, &onesided_out, axes, normalization, true);
      phi::funcs::FFTFillConj<DeviceContext, C>(
          dev_ctx, &onesided_out, out, axes);
    }
  }
};

template <typename DeviceContext, typename T>
class StftGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using C = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    const phi::DenseTensor* window = ctx.Input<phi::DenseTensor>("Window");
    const auto* dy = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    const size_t dy_rank = dy->dims().size();
    const size_t dx_rank = dx->dims().size();

    const int n_fft = ctx.Attr<int>("n_fft");
    const int hop_length = ctx.Attr<int>("hop_length");
    const bool normalized = ctx.Attr<bool>("normalized");
    const bool onesided = ctx.Attr<bool>("onesided");
    const int n_frames = dy->dims()[dy_rank - 1];
    const int seq_length = dx->dims()[dx_rank - 1];

    std::vector<int64_t> axes = {1};
    Tensor d_frames_w;
    framework::DDim d_frames_dims(dy->dims());
    d_frames_dims.at(axes.back()) = n_fft;
    d_frames_w.mutable_data<T>(d_frames_dims, ctx.GetPlace());

    Tensor complex_d_frames_w;
    complex_d_frames_w.mutable_data<C>(d_frames_dims, ctx.GetPlace());

    // dy -> d_frames_w
    phi::funcs::FFTNormMode normalization;
    if (normalized) {
      normalization = phi::funcs::get_norm_from_string("ortho", true);
    } else {
      normalization = phi::funcs::get_norm_from_string("backward", true);
    }
    phi::funcs::FFTC2CFunctor<DeviceContext, C, C> fft_c2c_func;

    if (!onesided) {
      fft_c2c_func(
          dev_ctx, *dy, &complex_d_frames_w, axes, normalization, false);
    } else {
      Tensor full_dy;
      full_dy.mutable_data<C>(d_frames_dims, ctx.GetPlace());
      auto zero_length = static_cast<int>(full_dy.dims().at(axes.back()) -
                                          dy->dims().at(axes.back()));
      auto rank = dy->dims().size();

      std::vector<int> pads(rank * 2, 0);
      pads[axes.back() * 2 + 1] = zero_length;

      phi::funcs::PaddingFunctor<DeviceContext, C>(
          rank, dev_ctx, pads, static_cast<C>(0), *dy, &full_dy);
      fft_c2c_func(
          dev_ctx, full_dy, &complex_d_frames_w, axes, normalization, false);
    }
    phi::RealKernel<C>(dev_ctx, complex_d_frames_w, &d_frames_w);

    // d_frames_w -> d_frames
    Tensor d_frames;
    d_frames.mutable_data<T>(d_frames_dims, ctx.GetPlace());
    ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
        ctx, &d_frames_w, window, axes.back(), MulFunctor<T>(), &d_frames);

    // d_frames -> dx
    phi::funcs::FrameFunctor<DeviceContext, T>()(dev_ctx,
                                                 &d_frames,
                                                 dx,
                                                 seq_length,
                                                 n_fft,
                                                 n_frames,
                                                 hop_length,
                                                 /*is_grad*/ true);
  }
};

}  // namespace operators
}  // namespace paddle
