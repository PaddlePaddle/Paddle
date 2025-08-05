// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
struct ConjugateTransposeFunctor {
  ConjugateTransposeFunctor(const T* input,
                            T* output,
                            int64_t batch_size,
                            int64_t n)
      : input_(input), output_(output), batch_size_(batch_size), n_(n) {}

  HOSTDEVICE void operator()(int64_t index) const {
    const int64_t n_square = n_ * n_;
    const int64_t batch_idx = index / n_square;
    const int64_t element_idx_in_batch = index % n_square;
    const int64_t row = element_idx_in_batch / n_;
    const int64_t col = element_idx_in_batch % n_;

    // The source element is at (batch_idx, row, col)
    const int64_t src_index = index;

    // The destination element is at (batch_idx, col, row)
    const int64_t dest_index = batch_idx * n_square + col * n_ + row;

    if constexpr (std::is_same_v<T, phi::dtype::complex<float>> ||
                  std::is_same_v<T, phi::dtype::complex<double>>) {
      output_[dest_index] = phi::dtype::conj(input_[src_index]);
    } else {
      // For real numbers, conjugate is a no-op.
      output_[dest_index] = input_[src_index];
    }
  }

 private:
  const T* input_;
  T* output_;
  int64_t batch_size_;
  int64_t n_;
};

}  // namespace funcs
}  // namespace phi
