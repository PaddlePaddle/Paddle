// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#else
#include <algorithm>
#endif

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename Tx, typename Ty>
struct SequenceMaskForRangeFunctor {
  HOSTDEVICE SequenceMaskForRangeFunctor(const Tx *x, Ty *y, int maxlen)
      : x_(x), y_(y), maxlen_(maxlen) {}

  HOSTDEVICE void operator()(int y_idx) const {
    int x_idx = y_idx / maxlen_;
    int j = y_idx % maxlen_;
    y_[y_idx] = static_cast<Ty>(j < x_[x_idx] ? 1 : 0);
  }

 private:
  const Tx *x_;
  Ty *y_;
  int maxlen_;
};

template <typename DeviceContext, typename Tx>
struct SequenceMaskFunctor {
  SequenceMaskFunctor(const DeviceContext &ctx,
                      const Tx *x,
                      phi::DenseTensor *y,
                      int limits,
                      int maxlen)
      : ctx_(ctx), x_(x), y_(y), limits_(limits), maxlen_(maxlen) {}

  template <typename Ty>
  void apply() const {
    auto *y_data = y_->mutable_data<Ty>(ctx_.GetPlace());
    platform::ForRange<DeviceContext> for_range(ctx_, limits_);
    for_range(SequenceMaskForRangeFunctor<Tx, Ty>(x_, y_data, maxlen_));
  }

 private:
  const DeviceContext &ctx_;
  const Tx *x_;
  phi::DenseTensor *y_;
  int limits_;
  int maxlen_;
};

template <typename DeviceContext, typename Tx>
class SequenceMaskKernel : public framework::OpKernel<Tx> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *y = ctx.Output<phi::DenseTensor>("Y");
    int maxlen = ctx.Attr<int>("maxlen");
    if (ctx.HasInput("MaxLenTensor")) {
      auto max_len_tensor = ctx.Input<phi::DenseTensor>("MaxLenTensor");
      PADDLE_ENFORCE_NOT_NULL(max_len_tensor,
                              platform::errors::InvalidArgument(
                                  "Input(MaxLenTensor) should not be NULL."
                                  "But received Input(MaxLenTensor) is NULL"));
      if (platform::is_gpu_place(max_len_tensor->place())) {
        phi::DenseTensor temp;
        paddle::framework::TensorCopySync(
            *max_len_tensor, platform::CPUPlace(), &temp);
        maxlen = *temp.data<int32_t>();
      } else {
        maxlen = *max_len_tensor->data<int32_t>();
      }

      auto y_dim = phi::vectorize<int>(x->dims());
      y_dim.push_back(maxlen);
      y->Resize(phi::make_ddim(y_dim));

      PADDLE_ENFORCE_GT(
          maxlen,
          0,
          platform::errors::InvalidArgument(
              "Input(MaxLenTensor) value should be greater than 0. But "
              "received Input(MaxLenTensor) value = %d.",
              maxlen));
    }

    auto *x_data = x->data<Tx>();
    auto x_numel = x->numel();
    if (maxlen < 0) {
#if defined(__NVCC__) || defined(__HIPCC__)
      VLOG(10)
          << "SequenceMaskOp on GPU may be slow when maxlen is not provided.";
      maxlen = static_cast<int>(
          thrust::reduce(thrust::device_pointer_cast(x_data),
                         thrust::device_pointer_cast(x_data) + x_numel,
                         static_cast<Tx>(0),
                         thrust::maximum<Tx>()));
#else
      maxlen = static_cast<int>(*std::max_element(x_data, x_data + x_numel));
#endif
      auto y_dim = phi::vectorize<int>(x->dims());
      y_dim.push_back(maxlen);
      y->Resize(phi::make_ddim(y_dim));
    }

    auto out_dtype = static_cast<framework::proto::VarType::Type>(
        ctx.Attr<int>("out_dtype"));
    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    framework::VisitDataType(out_dtype,
                             SequenceMaskFunctor<DeviceContext, Tx>(
                                 dev_ctx, x_data, y, x_numel * maxlen, maxlen));
  }
};

}  // namespace operators
}  // namespace paddle
