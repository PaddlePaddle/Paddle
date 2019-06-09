//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "paddle/fluid/framework/array.h"
#include "paddle/fluid/platform/assert.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace framework {

// Statically sized, statically indexed dimension
template <int D>
class Dim : public Array<int64_t, D> {
 public:
  static_assert(D >= 0, "D must be not less than 0");

  static constexpr int kRank = D;
  using BaseClass = Array<int64_t, D>;

  inline Dim(int64_t head, const Dim<D - 1>& tail) {
    (*this)[0] = head;
    new (this->GetMutable() + 1) Dim<D - 1>(tail);
  }

  template <typename... Args>
  HOSTDEVICE explicit Dim(int64_t head, Args... args)
      : BaseClass(head, args...) {}

  /** Construct a Dim from a linear index and size.  Uses Fortran order
   * indexing. */
  HOSTDEVICE Dim(int64_t idx, const Dim<D>& size);

  /** Construct a Dim with each dimension set to the given index */
  HOSTDEVICE explicit Dim(int64_t idx) { this->Fill(idx); }

  HOSTDEVICE Dim() = default;

  HOST std::string to_string() const;
};

namespace detail {
template <int kStart, int kEnd, bool kStop>
struct FortranOrderIndexingConstructorFunctor {
  HOSTDEVICE inline static void Run(const int64_t* in, int64_t* idx,
                                    int64_t* out) {
    out[kStart] = (*idx) % in[kStart];
    (*idx) /= in[kStart];
    FortranOrderIndexingConstructorFunctor<kStart + 1, kEnd,
                                           kStart + 1 == kEnd>::Run(in, idx,
                                                                    out);
  }
};

template <int kStart, int kEnd>
struct FortranOrderIndexingConstructorFunctor<kStart, kEnd, true> {
  HOSTDEVICE inline static void Run(const int64_t* in, int64_t* idx,
                                    int64_t* out) {}
};
}  // namespace detail

template <int D>
HOSTDEVICE Dim<D>::Dim(int64_t idx, const Dim<D>& size) {
  detail::FortranOrderIndexingConstructorFunctor<0, D, D == 0>::Run(
      size.Get(), &idx, this->GetMutable());
}

template <int idx, int D>
HOSTDEVICE inline int64_t get(const Dim<D>& dim) {
  return dim[idx];
}

template <int idx, int D>
HOSTDEVICE inline int64_t& get(Dim<D>& dim) {  // NOLINT
  return dim[idx];
}

template <int D>
HOSTDEVICE inline int64_t get(const Dim<D>& dim, int idx) {
  return dim[idx];
}

template <int D>
HOSTDEVICE inline int64_t& get(Dim<D>& dim, int idx) {  // NOLINT
  return dim[idx];
}

// Dot product of two dims
template <int D>
HOSTDEVICE inline int64_t linearize(const Dim<D>& a, const Dim<D>& b) {
  return UnrollProduct<D>::Run(a.Get(), b.Get());
}

// Product of a Dim
template <int D>
HOSTDEVICE inline int64_t product(const Dim<D>& a) {
  return UnrollProduct<D>::Run(a.Get());
}

// Is 0 <= idx_i < size_i for all i?
namespace detail {
template <int kStart, int kEnd, bool kStop>
struct ContainedFunctor {
  HOSTDEVICE static inline bool Run(const int64_t* idx, const int64_t* size) {
    return (idx[kStart] >= 0 && idx[kStart] < size[kStart]) &&
           ContainedFunctor<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(idx,
                                                                       size);
  }
};

template <int kStart, int kEnd>
struct ContainedFunctor<kStart, kEnd, true> {
  HOSTDEVICE static constexpr inline bool Run(const int64_t* idx,
                                              const int64_t* size) {
    return true;
  }
};
}  // namespace detail

template <int D>
HOSTDEVICE inline bool contained(const Dim<D>& idx, const Dim<D>& size) {
  return detail::ContainedFunctor<0, D, D == 0>::Run(idx.Get(), size.Get());
}

/**
 * \brief Compute exclusive prefix-multiply of a Dim.
 */
namespace detail {
template <int kStart, int kEnd, bool kStop>
struct ExPrefixMulFunctor {
  HOSTDEVICE static inline void Run(const int64_t* in, int64_t* out) {
    kStart == 0 ? out[kStart] = 1 : out[kStart] =
                                        out[kStart - 1] * in[kStart - 1];
    detail::ExPrefixMulFunctor<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(in,
                                                                          out);
  }
};

template <int kStart, int kEnd>
struct ExPrefixMulFunctor<kStart, kEnd, true> {
  HOSTDEVICE static inline void Run(const int64_t* in, int64_t* out) {}
};
}  // namespace detail

template <int D>
HOSTDEVICE inline Dim<D> ex_prefix_mul(const Dim<D>& src) {
  Dim<D> ret;
  detail::ExPrefixMulFunctor<0, D, D == 0>::Run(src.Get(), ret.GetMutable());
  return ret;
}

/**
 * Add two dimensions together
 */
template <int D>
HOSTDEVICE inline Dim<D> dim_plus(const Dim<D>& a, const Dim<D>& b) {
  Dim<D> ret;
  UnrollAdd<D>::Run(a.Get(), b.Get(), ret.GetMutable());
  return ret;
}

template <int D>
HOSTDEVICE inline Dim<D> operator+(const Dim<D>& lhs, const Dim<D>& rhs) {
  return dim_plus(lhs, rhs);
}

/**
 * Multiply two dimensions together
 */
template <int D>
HOSTDEVICE inline Dim<D> dim_mult(const Dim<D>& a, const Dim<D>& b) {
  Dim<D> ret;
  UnrollMul<D>::Run(a.Get(), b.Get(), ret.GetMutable());
  return ret;
}

template <int D>
HOSTDEVICE Dim<D> operator*(const Dim<D>& lhs, const Dim<D>& rhs) {
  return dim_mult(lhs, rhs);
}

/**
 * \brief Normalize strides to ensure any dimension with extent 1
 * has stride 0.
 *
 * \param size Dim object containing the size of an array
 * \param stride Dim object containing stride of an array
 * \return Dim object the same size as \p size with normalized strides
 *
 */
namespace detail {
template <int kStart, int kEnd, bool kStop>
struct NormalizeStridesFunctor {
  HOSTDEVICE static void Run(const int64_t* size, const int64_t* stride,
                             int64_t* ret) {
    ret[kStart] = (size[kStart] == 1 ? 0 : stride[kStart]);
    NormalizeStridesFunctor<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(
        size, stride, ret);
  }
};

template <int kStart, int kEnd>
struct NormalizeStridesFunctor<kStart, kEnd, true> {
  HOSTDEVICE static void Run(const int64_t* size, const int64_t* stride,
                             int64_t* ret) {}
};
}  // namespace detail

template <int D>
HOSTDEVICE Dim<D> normalize_strides(const Dim<D>& size, const Dim<D>& stride) {
  Dim<D> ret;
  detail::NormalizeStridesFunctor<0, D, D == 0>::Run(size.Get(), stride.Get(),
                                                     ret.GetMutable());
  return ret;
}

/**
 * Helper function to create a Dim
 *
 * \param idxes The type of Dim constructed depends on the number of params
 *
 */

template <typename... Args>
HOSTDEVICE inline Dim<sizeof...(Args)> make_dim(Args... idxes) {
  return Dim<sizeof...(Args)>(idxes...);
}

// Allows us to output a Dim
template <int D>
inline std::ostream& operator<<(std::ostream& os, const Dim<D>& d) {
  os << d[0];
  for (int i = 1; i < D; ++i) {
    os << ", " << d[i];
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Dim<0>& d) {
  return os;
}

template <int D>
HOST std::string Dim<D>::to_string() const {
  std::stringstream stream;
  stream << *this;
  return stream.str();
}

template <int D>
HOSTDEVICE Dim<D> linear_to_dimension(int linear_index, const Dim<D>& extents) {
  Dim<D> result;

  for (int i = 0; i < D - 1; ++i) {
    result[i] = linear_index % extents[i];
    linear_index /= extents[i];
  }

  result[D - 1] = linear_index;

  return result;
}

template <int D, typename T1, typename T2>
inline void static_dim_assign(const T1* in, T2* out) {
  UnrollAssign<D>::Run(in, out);
}

}  // namespace framework
}  // namespace paddle
