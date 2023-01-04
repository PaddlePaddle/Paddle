// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {
namespace funcs {

template <typename T>
class TrilTriuCompute {
 public:
  HOSTDEVICE TrilTriuCompute(const T* in,
                             const int diagonal,
                             const bool lower,
                             const int64_t H,
                             const int64_t W,
                             T* out)
      : in_(in), diagonal_(diagonal), lower_(lower), H_(H), W_(W), out_(out) {}

  HOSTDEVICE void operator()(int64_t idx) {
    const int64_t row = (idx / W_) % H_;
    const int64_t col = idx % W_;
    const bool mask =
        lower_ ? (col - row > diagonal_) : (col - row < diagonal_);
    out_[idx] = mask ? static_cast<T>(0) : in_[idx];
  }

 private:
  const T* in_;
  const int diagonal_;
  const bool lower_;
  const int64_t H_;
  const int64_t W_;
  T* out_;
};
}  // namespace funcs
}  // namespace phi
