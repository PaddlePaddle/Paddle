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

#include "paddle/phi/common/data_type.h"

namespace phi {
namespace dtype {

template <bool B, typename T>
struct cond {
  static constexpr bool value = B;
  using type = T;
};

template <bool B, typename TrueF, typename FalseF>
struct eval_if {
  using type = typename TrueF::type;
};

template <typename TrueF, typename FalseF>
struct eval_if<false, TrueF, FalseF> {
  using type = typename FalseF::type;
};

template <bool B, typename T, typename F>
using eval_if_t = typename eval_if<B, T, F>::type;

template <typename Head, typename... Tail>
struct select {
  using type = eval_if_t<Head::value, Head, select<Tail...>>;
};

template <typename T>
struct select<T> {
  using type = T;
};

template <bool B, typename T>
struct select<cond<B, T>> {
  // last one had better be true!
  static_assert(B, "No match select type!");
  using type = T;
};

template <typename Head, typename... Tail>
using select_t = typename select<Head, Tail...>::type;

// runtime real and complex type conversion

template <typename T>
using Real = select_t<cond<std::is_same<T, complex<float>>::value, float>,
                      cond<std::is_same<T, complex<double>>::value, double>,
                      T>;

template <typename T>
using Complex = select_t<cond<std::is_same<T, float>::value, complex<float>>,
                         cond<std::is_same<T, double>::value, complex<double>>,
                         T>;

inline DataType ToReal(DataType dtype) {
  switch (dtype) {
    case phi::DataType::COMPLEX64:
      return phi::DataType::FLOAT32;
    case phi::DataType::COMPLEX128:
      return phi::DataType::FLOAT64;
    default:
      return dtype;
  }
}

inline DataType ToComplex(DataType dtype) {
  switch (dtype) {
    case phi::DataType::FLOAT32:
      return phi::DataType::COMPLEX64;
    case phi::DataType::FLOAT64:
      return phi::DataType::COMPLEX128;
    default:
      return dtype;
  }
}

}  // namespace dtype
}  // namespace phi
