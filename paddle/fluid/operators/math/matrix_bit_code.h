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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

#if defined(_WIN32)
#include <intrin.h>
#include <windows.h>
#endif  // _WIN32

namespace paddle {
namespace operators {
namespace math {
/**
 * SimpleCodeTable class should support 3 functions:
 *
 * size_t size()
 *   return the number of ids
 *
 * int get_max_code_length()
 *   return the maximal code length
 *
 * SimpleCode operator()(size_t i)
 *   return the i-th code. Code class is descriebed below.
 *
 * SimpleCode class should support 3 functions:
 *
 * int get_length()
 *   return the length of the code
 *
 * size_t cal_index(int bit)
 *   bit ranges from 0 to get_length() - 1
 *   return the index for the (1+bit) level parent
 *
 * bool calc_bit(int bit)
 *   return true if the bit level parent is the right child of (1+bit) level
 *   parent
 *
 */

/**
 * return the 1-based index of the highest bit set
 *
 * for x > 0:
 * \f[
 *    FindLastSet(x) = 1 + \floor*{\log_{2}x}
 * \f]
 */
#if !defined(_WIN32)
inline constexpr size_t FindLastSet(size_t x) {
  return std::is_same<size_t, unsigned int>::value
             ? (x ? 8 * sizeof(x) - __builtin_clz(x) : 0)
             : (std::is_same<size_t, unsigned long>::value  // NOLINT
                    ? (x ? 8 * sizeof(x) - __builtin_clzl(x) : 0)
                    : (x ? 8 * sizeof(x) - __builtin_clzll(x) : 0));

#else
// windows don't have built-in clz, ctz function
template <typename T>
inline int ctz(const T& value) {
  DWORD trailing_zero = 0;
  if (_BitScanForward(&trailing_zero, value)) {
    return static_cast<int>(trailing_zero);
  } else {
    return static_cast<int>(0);
  }
}

template <typename T>
inline int clz(const T& value) {
  DWORD leadning_zero = 0;
  if (_BitScanReverse(&leadning_zero, value)) {
    return static_cast<int>(sizeof(T) * 8 - leadning_zero);
  } else {
    return static_cast<int>(0);
  }
}

inline size_t FindLastSet(size_t x) { return sizeof(size_t) * 8 - clz(x); }
#endif  // !_WIN32
}

struct SimpleCode {
  SimpleCode(size_t code, size_t num_classes) : c_(code + num_classes) {}
  /**
   * Here the id of root shoud be 1 rather than 0, thus the encoding of class c
   * is `c + num_classes` and all siblings can get the same weight indice using
   * prefixes.
   * Weight index is the prefixes of encoding, thus leave out the right most
   * bit in calc_index.
   * Binary classification path is the suffixes of encoding, thus leave out the
   * left most bit in calc_bit.
   */
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
};

template <typename T>
class MatrixBitCodeFunctor {
 public:
  explicit MatrixBitCodeFunctor(size_t num_classes, const int64_t* ids)
      : num_classes_(num_classes), ids_(ids) {}
  /* For j < code_length
       tmat(i, j) += vec(0, index(i, j))
  */
  void Add(framework::Tensor* tmat, const framework::Tensor& vec);

  /* For j < code_length
       vec(0, index(i, j)) += tmat(i, j)
  */
  void AddGrad(const framework::Tensor& tmat, framework::Tensor* vec);

  /* For j < code_length
    sum(i, 0) = \sum_j bit(i, j) * tmat(i, j)
  */
  void Sum(const framework::Tensor& tmat, framework::Tensor* sum, T scale_sum);

  /* For j < code_length
       tmat(i, j) -= bit(i, j)
  */
  void Sub(framework::Tensor* tmat);
  /* For j < code_length
       input.row(i) += tmat(i, j) * weight.row(index(i, j))
  */
  void Mul(framework::Tensor* tmat, const framework::Tensor& weight,
           const framework::Tensor& input);

  /* For index(i, j) >= 0:
      weight.row(index(i, j)) += tmat(i, j) * input.row(i)
  */
  void MulGradWeight(const framework::Tensor& tmat, framework::Tensor* weight,
                     const framework::Tensor& input);
  /* For j < code_length
    input.row(i) += tmat(i, j) * weight.row(index(i, j))
  */
  void MulGradError(const framework::Tensor& tmat,
                    const framework::Tensor& weight, framework::Tensor* input);

  size_t num_classes_;
  const int64_t* ids_;
};
}  // namespace math
}  // namespace operators
}  // namespace paddle
