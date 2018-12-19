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
#include <vector>
#include "paddle/fluid/framework/dim.h"

namespace paddle {
namespace framework {

template <typename T1, typename T2>
inline void dynamic_dim_assign(const T1* in, T2* out, int n) {
#define STATIC_DIM_ASSIGN_CASE(rank)          \
  case rank:                                  \
    static_dim_assign<rank, T1, T2>(in, out); \
    return
  switch (n) {
    STATIC_DIM_ASSIGN_CASE(0);
    STATIC_DIM_ASSIGN_CASE(1);
    STATIC_DIM_ASSIGN_CASE(2);
    STATIC_DIM_ASSIGN_CASE(3);
    STATIC_DIM_ASSIGN_CASE(4);
    STATIC_DIM_ASSIGN_CASE(5);
    STATIC_DIM_ASSIGN_CASE(6);
    STATIC_DIM_ASSIGN_CASE(7);
    STATIC_DIM_ASSIGN_CASE(8);
    STATIC_DIM_ASSIGN_CASE(9);
    default:
      PADDLE_THROW("Invalid rank %d", n);
  }
#undef STATIC_DIM_ASSIGN_CASE
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

  template <int D>
  inline DDim& operator=(const Dim<D>& in) {
    rank_ = D;
    UnsafeCast<D>() = in;
    return *this;
  }

  inline int64_t& operator[](int idx) { return dim_[idx]; }

  inline int64_t operator[](int idx) const { return dim_[idx]; }

  inline int64_t& at(int idx) {
    PADDLE_ENFORCE(idx >= 0 && idx < rank_);
    return dim_[idx];
  }

  inline int64_t at(int idx) const {
    PADDLE_ENFORCE(idx >= 0 && idx < rank_);
    return dim_[idx];
  }

  template <typename Visitor>
  typename std::result_of<Visitor(Dim<0>&)>::type apply_visitor(
      Visitor&& visitor);

  template <typename Visitor>
  typename std::result_of<Visitor(const Dim<0>&)>::type apply_visitor(
      Visitor&& visitor) const;

  bool operator==(const DDim& d) const;

  bool operator!=(const DDim& d) const;

  DDim operator+(const DDim& d) const;

  DDim operator*(const DDim& d) const;

  inline const int64_t* Get() const { return dim_.Get(); }

  inline int64_t* GetMutable() { return dim_.GetMutable(); }

  inline int size() const { return rank_; }

 private:
  template <int M>
  inline Dim<M>& UnsafeCast() {
    return const_cast<Dim<M>&>(const_cast<const DDim*>(this)->UnsafeCast<M>());
  }

  template <int M>
  inline const Dim<M>& UnsafeCast() const {
    static_assert(M >= 0 && M <= kMaxRank, "Invalid rank");
    auto* p = static_cast<const void*>(&dim_);
    return *reinterpret_cast<const Dim<M>*>(p);
  }

  friend DDim slice_ddim(const DDim& dim, int begin, int end);
  friend DDim stride(const DDim& ddim);
  friend DDim stride_numel(const DDim& ddim);

  Dim<kMaxRank> dim_;
  int rank_;
};

#define PADDLE_VISIT_DDIM(rank) \
  case rank:                    \
    return visitor(UnsafeCast<rank>())

template <typename Visitor>
typename std::result_of<Visitor(Dim<0>&)>::type DDim::apply_visitor(
    Visitor&& visitor) {
  switch (rank_) {
    PADDLE_VISIT_DDIM(0);
    PADDLE_VISIT_DDIM(1);
    PADDLE_VISIT_DDIM(2);
    PADDLE_VISIT_DDIM(3);
    PADDLE_VISIT_DDIM(4);
    PADDLE_VISIT_DDIM(5);
    PADDLE_VISIT_DDIM(6);
    PADDLE_VISIT_DDIM(7);
    PADDLE_VISIT_DDIM(8);
    PADDLE_VISIT_DDIM(9);
    default:
      PADDLE_THROW("Invalid rank %d", rank_);
  }
}

template <typename Visitor>
typename std::result_of<Visitor(const Dim<0>&)>::type DDim::apply_visitor(
    Visitor&& visitor) const {
  switch (rank_) {
    PADDLE_VISIT_DDIM(0);
    PADDLE_VISIT_DDIM(1);
    PADDLE_VISIT_DDIM(2);
    PADDLE_VISIT_DDIM(3);
    PADDLE_VISIT_DDIM(4);
    PADDLE_VISIT_DDIM(5);
    PADDLE_VISIT_DDIM(6);
    PADDLE_VISIT_DDIM(7);
    PADDLE_VISIT_DDIM(8);
    PADDLE_VISIT_DDIM(9);
    default:
      PADDLE_THROW("Invalid rank %d", rank_);
  }
}
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

int64_t get(const DDim& dim, int idx);
void set(DDim& dim, int idx, int val);  // NOLINT

std::vector<int64_t> vectorize(const DDim& ddim);
std::vector<int> vectorize2int(const DDim& ddim);

int64_t product(const DDim& ddim);

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
