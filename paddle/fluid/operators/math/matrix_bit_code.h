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
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/variant.h"

#if defined(_WIN32)
#include <intrin.h>
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
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
}
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

inline size_t FindLastSet(size_t x) { return 1 + sizeof(size_t) * 8 - clz(x); }
#endif  // !_WIN32
class SimpleCode {
 public:
  SimpleCode(size_t code, size_t num_classes, const int64_t* ids)
      : c_(static_cast<size_t>(ids[code]) + num_classes) {}
  /**
   * Here the id of root should be 1 rather than 0, thus the encoding of class c
   * is `c + num_classes` and all siblings can get the same weight index using
   * prefixes.
   * Weight index is the prefixes of encoding, thus leave out the right most
   * bit in calc_index.
   * Binary classification path is the suffixes of encoding, thus leave out the
   * left most bit in calc_bit.
   */
  size_t calc_index(int bit) const { return (c_ >> (bit + 1)) - 1; }
  bool calc_bit(int bit) const { return c_ & (1 << bit); }
  int get_length() const { return FindLastSet(c_) - 1; }

 private:
  size_t c_;
};

template <typename T>
class CustomCode {
 public:
  CustomCode(const framework::Tensor& path_table,
             const framework::Tensor& path_code, const int64_t* ids,
             int index) {
    seq_len_ = path_table.dims()[1];
    path_table_data_ = path_table.data<T>() + seq_len_ * index;
    path_code_data_ = path_code.data<T>() + seq_len_ * index;
  }
  /**
   * Here the id of root should be 1 rather than 0, thus the encoding of class c
   * is `c + num_classes` and all siblings can get the same weight index using
   * prefixes.
   * Weight index is the prefixes of encoding, thus leave out the right most
   * bit in calc_index.
   * Binary classification path is the suffixes of encoding, thus leave out the
   * left most bit in calc_bit.
   */
  size_t calc_index(int bit) const { return path_table_data_[bit]; }
  bool calc_bit(int bit) const { return path_code_data_[bit]; }

  // NOTE: this function is not thread-safe.
  int get_length() const {
    if (length_ < 0) {
      auto len = seq_len_;
      length_ = static_cast<int>(
          std::find_if(path_table_data_, path_table_data_ + len,
                       [](const T& val) { return val < 0; }) -
          path_table_data_);
    }
    return length_;
  }

 private:
  int64_t seq_len_;
  const T* path_table_data_;
  const T* path_code_data_;
  mutable int length_{-1};
};

class SimpleCodeTable {
 public:
  SimpleCodeTable(size_t num_classes, const int64_t* ids)
      : num_classes_(num_classes), ids_(ids) {}

  SimpleCode get_code(int64_t code) const {
    return SimpleCode(code, num_classes_, ids_);
  }

  size_t size() const { return num_classes_; }
  int get_max_code_length() const { return FindLastSet(num_classes_ - 1); }

 private:
  size_t num_classes_;
  const int64_t* ids_;
};

template <typename T>
class CustomCodeTable {
 public:
  CustomCodeTable(const framework::Tensor& path_table,
                  const framework::Tensor& path_code, const int64_t* ids)
      : ptable_(path_table), pcode_(path_code), ids_(ids) {}

  CustomCode<T> get_code(int64_t code) const {
    return CustomCode<T>(ptable_, pcode_, ids_, code);
  }

  size_t size() const { return static_cast<size_t>(ptable_.dims()[1]); }
  int get_max_code_length() const {
    return static_cast<size_t>(ptable_.dims()[1]);
  }

 private:
  const framework::Tensor& ptable_;
  const framework::Tensor& pcode_;
  const int64_t* ids_;
};

using CodeTable = boost::variant<SimpleCodeTable, CustomCodeTable<int64_t>>;

template <typename T>
class MatrixBitCodeFunctor {
 public:
  MatrixBitCodeFunctor(size_t num_classes, const int64_t* ids)
      : num_classes_(num_classes),
        ids_(ids),
        code_table_(SimpleCodeTable(num_classes, ids)) {}

  MatrixBitCodeFunctor(const framework::Tensor& path_table,
                       const framework::Tensor& path_code, const int64_t* ids)
      : num_classes_(static_cast<size_t>(path_table.dims()[1])),
        ids_(ids),
        code_table_(CustomCodeTable<int64_t>(path_table, path_code, ids)) {}
  /* For j < code_length
       tmat(i, j) += vec(0, index(i, j))
  */
  void Add(const framework::Tensor& vec, framework::Tensor* tmat);

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
  /* For SelectedRows Weight, For index(i, j) >= 0:
      weight.row(index(i, j)) += tmat(i, j) * input.row(i)
  */
  void MulGradWeight(const framework::Tensor& tmat,
                     framework::SelectedRows* weight,
                     const framework::Tensor& input);
  /* For j < code_length
    input.row(i) += tmat(i, j) * weight.row(index(i, j))
  */
  void MulGradError(const framework::Tensor& tmat,
                    const framework::Tensor& weight, framework::Tensor* input);

  size_t num_classes_;
  const int64_t* ids_;
  CodeTable code_table_;
};
}  // namespace math
}  // namespace operators
}  // namespace paddle
