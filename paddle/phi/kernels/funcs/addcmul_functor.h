// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include <limits>
#include <type_traits>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace funcs {

// Whether value overflows type To, the same as PyTorch: inf and nan can be
// cast to floating-point types, and negative values can wrap around for
// unsigned integer types.
template <typename To, typename From>
bool AddcmulValueOverflows(From value) {
  using Limits = std::numeric_limits<To>;
  if constexpr (std::is_floating_point<From>::value) {
    if (Limits::has_infinity && std::isinf(value)) {
      return false;
    }
    if (!Limits::has_quiet_NaN && std::isnan(value)) {
      return true;
    }
    return value < static_cast<From>(Limits::lowest()) ||
           value > static_cast<From>(Limits::max());
  } else {
    if (!Limits::is_signed && value < 0) {
      return -static_cast<uint64_t>(value) >
             static_cast<uint64_t>(Limits::max());
    }
    return value < Limits::lowest() || value > Limits::max();
  }
}

// Converts value to the computation type of T, raising an error if it
// overflows, the same as PyTorch.
template <typename T>
typename phi::dtype::MPTypeTrait<T>::Type GetAddcmulValue(const Scalar& value) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  using RealType = phi::dtype::Real<MPType>;
  bool overflow = false;
  switch (value.dtype()) {
    case DataType::BOOL:
      break;
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
    case DataType::FLOAT32:
    case DataType::FLOAT64:
      overflow = AddcmulValueOverflows<RealType>(value.to<double>());
      break;
    case DataType::COMPLEX64:
    case DataType::COMPLEX128: {
      auto complex_value = value.to<phi::complex128>();
      overflow =
          (std::is_same<MPType, RealType>::value && complex_value.imag != 0) ||
          AddcmulValueOverflows<RealType>(complex_value.real) ||
          AddcmulValueOverflows<RealType>(complex_value.imag);
      break;
    }
    default:
      overflow = AddcmulValueOverflows<RealType>(value.to<int64_t>());
  }
  PADDLE_ENFORCE_EQ(overflow,
                    false,
                    common::errors::InvalidArgument(
                        "The value %s of addcmul cannot be converted to %s "
                        "without overflow.",
                        value.ToString(),
                        phi::CppTypeToDataType<MPType>::Type()));
  return value.to<MPType>();
}

// out = input + value * tensor1 * tensor2, computed in MPType so that
// float16 and bfloat16 inputs are accumulated in float.
template <typename T>
struct AddcmulFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  explicit AddcmulFunctor(MPType value) : value_(value) {}

  HOSTDEVICE inline T operator()(const T input,
                                 const T tensor1,
                                 const T tensor2) const {
    return static_cast<T>(static_cast<MPType>(input) +
                          value_ * static_cast<MPType>(tensor1) *
                              static_cast<MPType>(tensor2));
  }

 private:
  MPType value_;
};

// Gradient of addcmul w.r.t. one multiplicand: out_grad * conj(value * other),
// where other is the other multiplicand.
template <typename T>
struct AddcmulGradFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  explicit AddcmulGradFunctor(MPType value) : value_(value) {}

  HOSTDEVICE inline T operator()(const T out_grad, const T other) const {
    return static_cast<T>(static_cast<MPType>(out_grad) *
                          (value_ * static_cast<MPType>(other)));
  }

 private:
  MPType value_;
};

template <typename T>
struct AddcmulGradFunctor<phi::dtype::complex<T>> {
  using MPType = phi::dtype::complex<T>;

  explicit AddcmulGradFunctor(MPType value) : value_(value) {}

  HOSTDEVICE inline MPType operator()(const MPType out_grad,
                                      const MPType other) const {
    return out_grad * conj(value_ * other);
  }

 private:
  MPType value_;
};

}  // namespace funcs
}  // namespace phi
