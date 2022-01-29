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

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "paddle/pten/core/hostdevice.h"
#include "paddle/pten/core/utils/array.h"

namespace pten {
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

  /** Construct a Dim with each dimension set to the given index */
  HOSTDEVICE explicit Dim(int64_t idx) { this->Fill(idx); }

  HOSTDEVICE Dim() = default;

  HOST std::string to_string() const;
};

// Product of a Dim
template <int D>
HOSTDEVICE inline int64_t product(const Dim<D>& a) {
  return UnrollProduct<D>::Run(a.Get());
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

template <int D, typename T1, typename T2>
inline void static_dim_assign(const T1* in, T2* out) {
  UnrollAssign<D>::Run(in, out);
}

}  // namespace framework
}  // namespace pten
