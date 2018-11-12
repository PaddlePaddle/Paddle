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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct ReverseSequenceFunctor {
  ReverseSequenceFunctor(const T *x_data_ptr,
                         const int64_t *seq_lengths_data_ptr,
                         int64_t seq_max_len, int64_t batch_size,
                         int64_t embedding_size, T *y_data_ptr)
      : x_data_ptr_(x_data_ptr),
        seq_lengths_data_ptr_(seq_lengths_data_ptr),
        seq_max_len_(seq_max_len),
        batch_size_(batch_size),
        embedding_size_(embedding_size),
        y_data_ptr_(y_data_ptr) {}

  HOSTDEVICE void operator()(size_t idx) const {
    size_t seq_idx = idx / (batch_size_ * embedding_size_);
    size_t batch_idx = (idx / embedding_size_) % batch_size_;

    int64_t seq_len = seq_lengths_data_ptr_[batch_idx];
    PADDLE_ASSERT(seq_len < seq_max_len_);

    size_t y_idx = idx;
    if (seq_idx <= static_cast<size_t>(seq_len)) {
      size_t embed_idx = idx % embedding_size_;
      size_t y_seq = seq_len - seq_idx;
      y_idx = (y_seq * batch_size_ + batch_idx) * embedding_size_ + embed_idx;
    }
    y_data_ptr_[y_idx] = x_data_ptr_[idx];
  }

  const T *x_data_ptr_;
  const int64_t *seq_lengths_data_ptr_;
  const int64_t seq_max_len_;
  const int64_t batch_size_;
  const int64_t embedding_size_;
  T *y_data_ptr_;
};

template <typename DeviceContext, typename T>
class ReverseSequenceOpKernel : public framework::OpKernel<T> {
  using LoDTensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &x = *ctx.Input<LoDTensor>("X");
    auto &seq_lengths = *ctx.Input<LoDTensor>("SeqLen");
    auto *y = ctx.Output<LoDTensor>("Y");

    int64_t seq_max_len = x.dims()[0];
    int64_t batch_size = x.dims()[1];
    int64_t embedding_size = x.dims()[2];

    PADDLE_ENFORCE_EQ(batch_size, seq_lengths.dims()[0]);

    auto x_data_ptr = x.data<T>();
    auto seq_lengths_data_ptr = seq_lengths.data<int64_t>();
    auto *y_data_ptr = y->mutable_data<T>(ctx.GetPlace());

    ReverseSequenceFunctor<T> functor(x_data_ptr, seq_lengths_data_ptr,
                                      seq_max_len, batch_size, embedding_size,
                                      y_data_ptr);

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    size_t limit = static_cast<size_t>(x.numel());

    platform::ForRange<DeviceContext> for_range(dev_ctx, limit);
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle
