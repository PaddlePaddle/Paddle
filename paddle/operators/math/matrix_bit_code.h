/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {
namespace math {

/**
 * return the 1-based index of the highest bit set
 *
 * for x > 0:
 * \f[
 *    findLastSet(x) = 1 + \floor*{\log_{2}x}
 * \f]
 */
inline constexpr size_t FindLastSet(size_t x) {
  return std::is_same<size_t, unsigned int>::value
             ? (x ? 8 * sizeof(x) - __builtin_clz(x) : 0)
             : (std::is_same<size_t, unsigned long>::value  // NOLINT
                    ? (x ? 8 * sizeof(x) - __builtin_clzl(x) : 0)
                    : (x ? 8 * sizeof(x) - __builtin_clzll(x) : 0));
}

struct SimpleCode {
  SimpleCode(size_t code, size_t num_classes) : c_(code + num_classes) {}
  inline size_t calc_index(int bit) const { return (c_ >> (bit + 1)) - 1; }
  inline bool calc_bit(int bit) const { return c_ & (1 << bit); }
  inline int get_length() const { return FindLastSet(c_) - 1; }

 private:
  size_t c_;
};

struct SimpleCodeTable {
  explicit SimpleCodeTable(size_t num_classes) : num_classes_(num_classes) {}
  SimpleCode operator()(size_t code) const {
    return SimpleCode(code, num_classes_);
  }
  size_t size() const { return num_classes_; }
  int get_max_code_length() const { return FindLastSet(num_classes_ - 1); }

 private:
  size_t num_classes_;
  int max_code_length_;
};

/* For j < codeLength
    tmat(i, j) += vec(0, index(i, j))
*/
template <typename T>
void AddByBitCode(size_t num_classes, const framework::Tensor& codes,
                  framework::Tensor& tmat, const framework::Tensor& vec);

/* For j < codeLength
    sum(i, 0) = \sum_j bit(i, j) * tmat(i, j)
*/
template <typename T>
void SumByBitCode(size_t num_classes, const framework::Tensor& codes,
                  framework::Tensor& tmat, framework::Tensor& sum, T scale_sum);

/* For j < codeLength
    input.row(i) += tmat(i, j) * weight.row(index(i, j))
*/
template <typename T>
void MulByBitCode(size_t num_classes, const framework::Tensor& codes,
                  framework::Tensor& tmat, const framework::Tensor& weight,
                  const framework::Tensor& input);
}  // namespace math
}  // namespace operators
}  // namespace paddle
