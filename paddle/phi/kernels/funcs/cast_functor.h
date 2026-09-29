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

#include "paddle/common/hostdevice.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/common/float8_e5m2.h"

namespace phi {

template <typename InT, typename OutT>
struct CastOpTransformFunctor {
  HOSTDEVICE OutT operator()(InT in) const { return static_cast<OutT>(in); }
};

template <>
struct CastOpTransformFunctor<::phi::dtype::float8_e5m2, ::phi::complex64> {
  HOSTDEVICE ::phi::complex64 operator()(::phi::dtype::float8_e5m2 in) const {
    return ::phi::complex64(static_cast<float>(in));
  }
};

template <>
struct CastOpTransformFunctor<::phi::dtype::float8_e5m2, ::phi::complex128> {
  HOSTDEVICE ::phi::complex128 operator()(::phi::dtype::float8_e5m2 in) const {
    return ::phi::complex128(static_cast<double>(in));
  }
};

template <>
struct CastOpTransformFunctor<::phi::dtype::float8_e4m3fn, ::phi::complex64> {
  HOSTDEVICE ::phi::complex64 operator()(::phi::dtype::float8_e4m3fn in) const {
    return ::phi::complex64(static_cast<float>(in));
  }
};

template <>
struct CastOpTransformFunctor<::phi::dtype::float8_e4m3fn, ::phi::complex128> {
  HOSTDEVICE ::phi::complex128 operator()(
      ::phi::dtype::float8_e4m3fn in) const {
    return ::phi::complex128(static_cast<double>(in));
  }
};

template <>
struct CastOpTransformFunctor<::phi::dtype::bfloat16, ::phi::complex64> {
  HOSTDEVICE ::phi::complex64 operator()(::phi::dtype::bfloat16 in) const {
    return ::phi::complex64(static_cast<float>(in));
  }
};

template <>
struct CastOpTransformFunctor<::phi::dtype::bfloat16, ::phi::complex128> {
  HOSTDEVICE ::phi::complex128 operator()(::phi::dtype::bfloat16 in) const {
    return ::phi::complex128(static_cast<double>(in));
  }
};

template <>
struct CastOpTransformFunctor<::phi::dtype::float16, ::phi::complex64> {
  HOSTDEVICE ::phi::complex64 operator()(::phi::dtype::float16 in) const {
    return ::phi::complex64(static_cast<float>(in));
  }
};

template <>
struct CastOpTransformFunctor<::phi::dtype::float16, ::phi::complex128> {
  HOSTDEVICE ::phi::complex128 operator()(::phi::dtype::float16 in) const {
    return ::phi::complex128(static_cast<double>(in));
  }
};

}  // namespace phi
