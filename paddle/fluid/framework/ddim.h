/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "paddle/fluid/framework/dim.h"

namespace paddle {
namespace framework {

#define PADDLE_VISIT_DDIM_BASE(rank, callback) \
  case (rank): {                               \
    constexpr auto kRank = (rank);             \
    return (callback);                         \
  }

#define PADDLE_VISIT_DDIM(rank, callback)                                  \
  switch (rank) {                                                          \
    PADDLE_VISIT_DDIM_BASE(0, callback);                                   \
    PADDLE_VISIT_DDIM_BASE(1, callback);                                   \
    PADDLE_VISIT_DDIM_BASE(2, callback);                                   \
    PADDLE_VISIT_DDIM_BASE(3, callback);                                   \
    PADDLE_VISIT_DDIM_BASE(4, callback);                                   \
    PADDLE_VISIT_DDIM_BASE(5, callback);                                   \
    PADDLE_VISIT_DDIM_BASE(6, callback);                                   \
    PADDLE_VISIT_DDIM_BASE(7, callback);                                   \
    PADDLE_VISIT_DDIM_BASE(8, callback);                                   \
    PADDLE_VISIT_DDIM_BASE(9, callback);                                   \
    default:                                                               \
      PADDLE_THROW(platform::errors::Unimplemented(                        \
          "Invalid dimension to be accessed. Now only supports access to " \
          "dimension 0 to 9, but received dimension is %d.",               \
          rank));                                                          \
  }

template <typename T1, typename T2>
inline void dynamic_dim_assign(const T1* in, T2* out, int n) {
  PADDLE_VISIT_DDIM(n, (static_dim_assign<kRank, T1, T2>(in, out)));
}

/**
 * \brief A dynamically sized dimension.
 *
 * The number of dimensions must be between [1, 9].
 */
class DDim {
 public:
  constexpr static int kMaxRank = 9;

  DDim() : rank_(1) { dim_[0] = 0; }

  DDim(const DDim& ddim) : dim_() { CopyFrom(ddim); }

  DDim(const int* d, int n) : rank_(n) {
    dynamic_dim_assign(d, dim_.GetMutable(), n);
  }

  DDim(const int64_t* d, int n) : rank_(n) {
    dynamic_dim_assign(d, dim_.GetMutable(), n);
  }

  template <int D>
  /*implicit*/ DDim(const Dim<D>& in) : rank_(D) {  // NOLINT
    UnsafeCast<D>() = in;
  }

  /*implicit*/ DDim(std::initializer_list<int64_t> init_list)
      : DDim(init_list.begin(), init_list.size()) {}

  inline DDim& operator=(const DDim& ddim) { return CopyFrom(ddim); }

  template <int D>
  inline DDim& operator=(const Dim<D>& dim) {
    rank_ = D;
    UnsafeCast<D>() = dim;
    return *this;
  }

  inline int64_t& operator[](int idx) { return dim_[idx]; }

  inline int64_t operator[](int idx) const { return dim_[idx]; }

  int64_t& at(int idx) {
    PADDLE_ENFORCE_GE(idx, 0,
                      platform::errors::InvalidArgument(
                          "Invalid DDim index to be accessed. The valid index "
                          "is between 0 and %d, but received index is %d.",
                          rank_, idx));
    PADDLE_ENFORCE_LT(idx, rank_,
                      platform::errors::InvalidArgument(
                          "Invalid DDim index to be accessed. The valid index "
                          "is between 0 and %d, but received index is %d.",
                          rank_, idx));
    return dim_[idx];
  }

  int64_t at(int idx) const {
    PADDLE_ENFORCE_GE(idx, 0,
                      platform::errors::InvalidArgument(
                          "Invalid DDim index to be accessed. The valid index "
                          "is between 0 and %d, but received index is %d.",
                          rank_, idx));
    PADDLE_ENFORCE_LT(idx, rank_,
                      platform::errors::InvalidArgument(
                          "Invalid DDim index to be accessed. The valid index "
                          "is between 0 and %d, but received index is %d.",
                          rank_, idx));
    return dim_[idx];
  }

  template <typename Visitor>
  typename std::result_of<Visitor(Dim<0>&)>::type apply_visitor(
      Visitor&& visitor) {
    PADDLE_VISIT_DDIM(rank_, visitor(UnsafeCast<kRank>()));
  }

  template <typename Visitor>
  typename std::result_of<Visitor(const Dim<0>&)>::type apply_visitor(
      Visitor&& visitor) const {
    PADDLE_VISIT_DDIM(rank_, visitor(UnsafeCast<kRank>()));
  }

  bool operator==(const DDim& d) const;

  bool operator!=(const DDim& d) const;

  inline const int64_t* Get() const { return dim_.Get(); }

  inline int64_t* GetMutable() { return dim_.GetMutable(); }

  inline int size() const { return rank_; }

  std::string to_str() const;

  DDim reshape(const std::vector<int>& shape) const;

  DDim transpose(const std::vector<int>& axis) const;

 private:
  template <int D>
  inline Dim<D>& UnsafeCast() {
    static_assert(D >= 0 && D <= kMaxRank, "Invalid rank");
    auto* p = static_cast<void*>(&dim_);
    return *reinterpret_cast<Dim<D>*>(p);
  }

  template <int D>
  inline const Dim<D>& UnsafeCast() const {
    static_assert(D >= 0 && D <= kMaxRank, "Invalid rank");
    auto* p = static_cast<const void*>(&dim_);
    return *reinterpret_cast<const Dim<D>*>(p);
  }

  inline DDim& CopyFrom(const DDim& ddim) {
    PADDLE_VISIT_DDIM(ddim.rank_, (*this = ddim.UnsafeCast<kRank>()));
  }

  friend DDim stride(const DDim& ddim);
  friend DDim stride_numel(const DDim& ddim);

 private:
  Dim<kMaxRank> dim_;
  int rank_;
};

#undef PADDLE_VISIT_DDIM_BASE
#undef PADDLE_VISIT_DDIM

/**
 * \brief Make a DDim from std::vector<int64_t>
 *
 * \param dims An vector of ints. Must be sized between [1, 9]
 */
DDim make_ddim(const std::vector<int64_t>& dims);

DDim make_ddim(const std::vector<int>& dims);

/**
 * \brief Make a DDim from an initializer list
 *
 * \param dims An initializer list of ints. Must be sized between [1, 9]
 *
 */
DDim make_ddim(std::initializer_list<int64_t> dims);

template <typename T = int64_t>
std::vector<T> vectorize(const DDim& ddim) {
  std::vector<T> result(DDim::kMaxRank);
  dynamic_dim_assign(ddim.Get(), result.data(), ddim.size());
  result.resize(ddim.size());
  return result;
}

int64_t product(const DDim& ddim);

bool contain_unknown_dim(const DDim& ddim);

/**
 * \brief Slice a ddim
 *
 * Slice dim with [begin, end).
 * e.g.  DDim d = make_ddim({1,2,3,4,5});
 *       slice_ddim(d, 1, 3); ====> {2,3}
 */
DDim slice_ddim(const DDim& dim, int begin, int end);

/**
 * \brief What is the length of this dimension?
 *
 * \param Dynamic dimension to inspect
 */

int arity(const DDim& ddim);

std::ostream& operator<<(std::ostream&, const DDim&);

// Reshape a tensor to a matrix. The matrix's first dimension(column length)
// will be the product of tensor's first `num_col_dims` dimensions.
DDim flatten_to_2d(const DDim& src, int num_col_dims);

DDim flatten_to_1d(const DDim& src);

DDim stride(const DDim& ddim);

DDim stride_numel(const DDim& ddim);
}  // namespace framework
}  // namespace paddle
