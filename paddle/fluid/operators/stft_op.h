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

#include "paddle/fluid/operators/frame_op.h"
#include "paddle/fluid/operators/spectral_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class StftKernel : public framework::OpKernel<T> {
 public:
  /*
    Batch Signals (N, T) -> Frames (N, n_fft, num_frames) -> FFTR2C -> (N,
    n_fft/2 + 1, num_frames)
  */
  void Compute(const framework::ExecutionContext& ctx) const override {
    using C = paddle::platform::complex<T>;
    const Tensor* x = ctx.Input<Tensor>("X");
    Tensor* out = ctx.Output<Tensor>("Out");
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

    int64_t axis = 1;

    // Frame
    Tensor frames;
    framework::DDim frames_dims(out->dims());
    frames_dims.at(axis) = n_fft;
    frames.mutable_data<T>(frames_dims, ctx.GetPlace());
    FrameFunctor<DeviceContext, T>()(dev_ctx, x, &frames, seq_length, n_fft,
                                     n_frames, hop_length,
                                     /*is_grad*/ false);

    // FFTR2C
    FFTNormMode normalization;
    if (normalized) {
      normalization = get_norm_from_string("ortho", true);
    } else {
      normalization = get_norm_from_string("backward", true);
    }
    FFTR2CFunctor<DeviceContext, T, C> fft_r2c_func;

    if (onesided) {
      fft_r2c_func(dev_ctx, &frames, out, std::vector<int64_t>{axis},
                   normalization, true);
    } else {
      framework::DDim onesided_dims(out->dims());
      const int64_t onesided_axis_size = out->dims().at(axis) / 2 + 1;
      onesided_dims.at(axis) = onesided_axis_size;
      Tensor onesided_out;
      onesided_out.mutable_data<C>(onesided_dims, ctx.GetPlace());
      fft_r2c_func(dev_ctx, &frames, &onesided_out, std::vector<int64_t>{axis},
                   normalization, true);
      fill_conj<DeviceContext, C>(dev_ctx, &onesided_out, out,
                                  std::vector<int64_t>{axis});
    }
  }
};

}  // namespace operators
}  // namespace paddle
