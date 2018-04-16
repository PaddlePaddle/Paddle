//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/details/reduce_util.h"
namespace paddle {
namespace framework {
namespace details {

struct ReduceLoDTensor {
  const std::vector<LoDTensor> &src_tensors_;
  LoDTensor &dst_tensor_;

  ReduceLoDTensor(const std::vector<LoDTensor> &src, LoDTensor *dst)
      : src_tensors_(src), dst_tensor_(*dst) {}

  template <typename T>
  void operator()() const {
    PADDLE_ENFORCE(!src_tensors_.empty());
    auto &t0 = src_tensors_[0];
    PADDLE_ENFORCE_NE(t0.numel(), 0);
    dst_tensor_.Resize(t0.dims());
    T *dst = dst_tensor_.mutable_data<T>(platform::CPUPlace());
    std::copy(t0.data<T>(), t0.data<T>() + t0.numel(), dst);

    for (size_t i = 1; i < src_tensors_.size(); ++i) {
      auto &t = src_tensors_[i];
      PADDLE_ENFORCE_EQ(t.dims(), t0.dims());
      PADDLE_ENFORCE_EQ(t.type(), t0.type());
      std::transform(t.data<T>(), t.data<T>() + t.numel(), dst, dst,
                     [](T a, T b) -> T { return a + b; });
    }
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
