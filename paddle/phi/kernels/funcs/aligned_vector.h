/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.1 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.1

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include "paddle/phi/core/hostdevice.h"
#if defined(__xpu__)
#define CHAR_BIT 8
#endif

namespace phi {

// Aligned vector generates vectorized load/store on CUDA.
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];

  HOSTDEVICE inline const T& operator[](int i) const { return val[i]; }
  HOSTDEVICE inline T& operator[](int i) { return val[i]; }
};

template <typename T, int Size>
HOSTDEVICE inline void Load(const T* addr, AlignedVector<T, Size>* vec) {
  const AlignedVector<T, Size>* addr_vec =
      reinterpret_cast<const AlignedVector<T, Size>*>(addr);
  *vec = *addr_vec;
}

template <typename T, int Size>
HOSTDEVICE inline void Store(const AlignedVector<T, Size>& vec, T* addr) {
  AlignedVector<T, Size>* addr_vec =
      reinterpret_cast<AlignedVector<T, Size>*>(addr);
  *addr_vec = vec;
}

/*
* Only the address of input data is the multiplier of 1,2,4, vectorized load
* with corresponding multiplier-value is possible. Moreover, the maximum length
* of vectorized load is 128 bits once. Hence, valid length of vectorized load
* shall be determined under both former constraints.
*/
template <typename T>
int GetVectorizedSize(const T* pointer) {
  constexpr int max_load_bits = 128;
  int valid_vec_size = max_load_bits / CHAR_BIT / sizeof(T);
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec8 = std::alignment_of<AlignedVector<T, 8>>::value;  // NOLINT
  constexpr int vec4 = std::alignment_of<AlignedVector<T, 4>>::value;  // NOLINT
  constexpr int vec2 = std::alignment_of<AlignedVector<T, 2>>::value;  // NOLINT
  if (address % vec8 == 0) {
    /*
    * Currently, decide to deal with no more than 4 data once while adopting
    * vectorization load/store, if performance test shows that dealing with
    * 8 data once in vectorization load/store does get optimized, return code
    * below can be changed into " return std::min(8, valid_vec_size); " .
    */
    return std::min(4, valid_vec_size);
  } else if (address % vec4 == 0) {
    return std::min(4, valid_vec_size);
  } else if (address % vec2 == 0) {
    return std::min(2, valid_vec_size);
  } else {
    return 1;
  }
}

}  // namespace phi
