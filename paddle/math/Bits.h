/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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

#include <type_traits>

namespace paddle {

/**
 * From Facebook folly:
 * https://github.com/facebook/folly/blob/master/folly/Bits.h
 *
 * findLastSet: return the 1-based index of the highest bit set
 *
 * for x > 0:
 * \f[
 *    findLastSet(x) = 1 + \floor*{\log_{2}x}
 * \f]
 */
template <class T>
inline constexpr typename std::enable_if<(std::is_integral<T>::value &&
                                          std::is_unsigned<T>::value &&
                                          sizeof(T) <= sizeof(unsigned int)),
                                         unsigned int>::type
findLastSet(T x) {
  return x ? 8 * sizeof(unsigned int) - __builtin_clz(x) : 0;
}

template <class T>
inline constexpr
    typename std::enable_if<(std::is_integral<T>::value &&
                             std::is_unsigned<T>::value &&
                             sizeof(T) > sizeof(unsigned int) &&
                             sizeof(T) <= sizeof(unsigned long)),  // NOLINT
                            unsigned int>::type
    findLastSet(T x) {
  return x ? 8 * sizeof(unsigned long) - __builtin_clzl(x) : 0;  // NOLINT
}

}  // namespace paddle
