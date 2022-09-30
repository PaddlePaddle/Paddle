/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {
template <typename T, typename WT>
struct TargetAssignFunctor {
  const T* in_;
  const int* match_indices_;
  const size_t* lod_;
  const int mismatch_value_;
  const int64_t N_;
  const int64_t M_;
  const int64_t P_;
  const int64_t K_;

  T* out_;
  WT* out_wt_;

  TargetAssignFunctor(const T* input,
                      const int* match_indices,
                      const size_t* lod,
                      const int mismatch_value,
                      const int64_t N,
                      const int64_t M,
                      const int64_t P,
                      const int64_t K,
                      T* out,
                      WT* out_wt)
      : in_(input),
        match_indices_(match_indices),
        lod_(lod),
        mismatch_value_(mismatch_value),
        N_(N),
        M_(M),
        P_(P),
        K_(K),
        out_(out),
        out_wt_(out_wt) {}

  HOSTDEVICE void operator()(size_t i) const {
    int h = i / M_;
    int w = i - h * M_;

    size_t off = lod_[h];
    int id = match_indices_[i];

    T* out = out_ + i * K_;
    WT* out_wt = out_wt_ + i;

    if (id > -1) {
      int w_off = w % P_;
      const T* in = in_ + ((off + id) * P_ + w_off) * K_;
      for (int64_t k = 0; k < K_; ++k) {
        out[k] = in[k];
      }
      out_wt[0] = static_cast<WT>(1.);
    } else {
      for (int64_t k = 0; k < K_; ++k) {
        out[k] = static_cast<T>(mismatch_value_);
      }
      out_wt[0] = static_cast<WT>(0.);
    }
  }
};

template <typename DeviceContext, typename T, typename WT>
struct NegTargetAssignFunctor {
  void operator()(const platform::DeviceContext& ctx,
                  const int* neg_indices,
                  const size_t* lod,
                  const int N,
                  const int M,
                  const int K,
                  const int mismatch_value,
                  T* out,
                  WT* out_wt) const;
};

template <typename DeviceContext, typename T, typename WT>
class TargetAssignKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* match_indices = ctx.Input<phi::DenseTensor>("MatchIndices");

    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto* out_wt = ctx.Output<phi::DenseTensor>("OutWeight");

    PADDLE_ENFORCE_EQ(x->lod().size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "TargetAssignOp input(X) needs 1 level of LoD"));
    int mismatch_value = ctx.Attr<int>("mismatch_value");

    const T* x_data = x->data<T>();
    const int* match_idx_data = match_indices->data<int>();

    T* out_data = out->mutable_data<T>(ctx.GetPlace());
    WT* out_wt_data = out_wt->mutable_data<WT>(ctx.GetPlace());

    int64_t n = match_indices->dims()[0];
    int64_t m = match_indices->dims()[1];
    int64_t p = x->dims()[1];
    int64_t k = x->dims()[2];

    auto x_lod = x->lod().back();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    paddle::framework::MixVector<size_t> mixv_x_lod(&x_lod);
    size_t* x_lod_data = mixv_x_lod.MutableData(ctx.GetPlace());
#else
    size_t* x_lod_data = x_lod.data();
#endif

    TargetAssignFunctor<T, WT> functor(x_data,
                                       match_idx_data,
                                       x_lod_data,
                                       mismatch_value,
                                       n,
                                       m,
                                       p,
                                       k,
                                       out_data,
                                       out_wt_data);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    mixv_x_lod.CopyToCPU();
#endif

    auto& device_ctx = ctx.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(device_ctx, n * m);
    for_range(functor);

    auto* neg_indices = ctx.Input<framework::LoDTensor>("NegIndices");
    if (neg_indices) {
      PADDLE_ENFORCE_EQ(
          neg_indices->lod().size(),
          1UL,
          platform::errors::InvalidArgument(
              "TargetAssignOp input(NegIndices) needs 1 level of LoD"));
      const int* neg_idx_data = neg_indices->data<int>();
      auto neg_lod = neg_indices->lod().back();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      paddle::framework::MixVector<size_t> mixv_neg_lod(&neg_lod);
      size_t* neg_lod_data = mixv_neg_lod.MutableData(ctx.GetPlace());
#else
      size_t* neg_lod_data = neg_lod.data();
#endif
      NegTargetAssignFunctor<DeviceContext, T, WT> neg_trg_functor;
      neg_trg_functor(device_ctx,
                      neg_idx_data,
                      neg_lod_data,
                      n,
                      m,
                      k,
                      mismatch_value,
                      out_data,
                      out_wt_data);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      mixv_neg_lod.CopyToCPU();
#endif
    }
  }
};

}  // namespace operators
}  // namespace paddle
