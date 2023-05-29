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

namespace phi {
namespace funcs {

template <typename T, class Enable = void>
struct IsNanFunctor {
  HOSTDEVICE bool operator()(const T& a) const {
#if defined(__CUDACC__) || defined(__HIPCC__)
    return ::isnan(a);
#else
    return std::isnan(a);
#endif
  }
};

template <typename T>
struct IsNanFunctor<T,
                    typename std::enable_if<std::is_integral<T>::value>::type> {
  HOSTDEVICE bool operator()(const T& a UNUSED) const { return false; }
};

// isnan is defined in namespace std in float16.h, but
// on rocm platform, it still got:
// "error: call to 'isnan' is ambiguous".
// So use phi::dtype::isnan here.
template <>
struct IsNanFunctor<phi::dtype::float16, void> {
  HOSTDEVICE bool operator()(const phi::dtype::float16& a) const {
    return phi::dtype::isnan(a);
  }
};

template <>
struct IsNanFunctor<phi::dtype::bfloat16, void> {
  HOSTDEVICE bool operator()(const phi::dtype::bfloat16& a) const {
    return phi::dtype::isnan(a);
  }
};

template <typename T, class Enable = void>
struct IsInfFunctor {
  HOSTDEVICE bool operator()(const T& a) const {
#if defined(__CUDACC__) || defined(__HIPCC__)
    return ::isinf(a);
#else
    return std::isinf(a);
#endif
  }
};

template <typename T>
struct IsInfFunctor<T,
                    typename std::enable_if<std::is_integral<T>::value>::type> {
  HOSTDEVICE bool operator()(const T& a UNUSED) const { return false; }
};

template <>
struct IsInfFunctor<phi::dtype::float16, void> {
  HOSTDEVICE bool operator()(const phi::dtype::float16& a) const {
    return phi::dtype::isinf(a);
  }
};

template <>
struct IsInfFunctor<phi::dtype::bfloat16, void> {
  HOSTDEVICE bool operator()(const phi::dtype::bfloat16& a) const {
    return phi::dtype::isinf(a);
  }
};

template <typename T, class Enable = void>
struct IsFiniteFunctor {
  HOSTDEVICE bool operator()(const T& a) const {
#if defined(__CUDACC__) || defined(__HIPCC__)
    return ::isfinite(a);
#else
    return std::isfinite(a);
#endif
  }
};

template <typename T>
struct IsFiniteFunctor<
    T,
    typename std::enable_if<std::is_integral<T>::value>::type> {
  HOSTDEVICE bool operator()(const T& a UNUSED) const { return true; }
};

template <>
struct IsFiniteFunctor<phi::dtype::float16, void> {
  HOSTDEVICE bool operator()(const phi::dtype::float16& a) const {
    return phi::dtype::isfinite(a);
  }
};

template <>
struct IsFiniteFunctor<phi::dtype::bfloat16, void> {
  HOSTDEVICE bool operator()(const phi::dtype::bfloat16& a) const {
    return phi::dtype::isfinite(a);
  }
};

}  // namespace funcs
}  // namespace phi
