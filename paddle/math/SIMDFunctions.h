/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <stddef.h>
#include <stdint.h>

namespace paddle {

namespace simd {

namespace naive {
template <typename Type>
inline void addTo(Type* a, const Type* b, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    a[i] += b[i];
  }
}

template <typename Type>
inline void batchAddTo(Type* a, const Type* b[], int batch, size_t len) {
  for (int i = 0; i < batch; ++i) {
    for (size_t j = 0; j < len; ++j) {
      a[j] += b[i][j];
    }
  }
}

/**
 * @note this method is unused in paddle.
 */
template <typename Type>
inline void colMax(Type* result, const Type* data, int dim, int numSamples) {
  for (int d = 0; d < dim; ++d) {
    Type sm = data[d];
    for (int i = 1; i < numSamples; ++i) {
      sm = sm > data[i * dim + d] ? sm : data[i * dim + d];
    }
    result[d] = sm;
  }
}

template <typename Type>
inline void decayL1(Type* dst, Type* src, Type* lr, Type lambda, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    Type& src_val = src[i];
    float nlambda = lr[i] * lambda;
    if (src_val > 0) {
      dst[i] = ((src_val > nlambda) ? (src_val - nlambda) : 0);
    } else {
      dst[i] = ((-src_val > nlambda) ? (src_val + nlambda) : 0);
    }
  }
}

template <class Type>
inline void decayL1(Type* dst, Type* src, Type lambda, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    Type& src_val = src[i];
    if (src_val > 0) {
      dst[i] = ((src_val > lambda) ? (src_val - lambda) : 0);
    } else {
      dst[i] = ((-src_val > lambda) ? (src_val + lambda) : 0);
    }
  }
}
}  // namespace naive

template <typename Type>
inline void addTo(Type* a, const Type* b, size_t len) {
  naive::addTo(a, b, len);
}

template <typename Type>
inline void batchAddTo(Type* a, const Type* b[], int batch, size_t len) {
  naive::batchAddTo(a, b, batch, len);
}

template <typename Type>
inline void colMax(Type* result, const Type* data, int dim, int numSamples) {
  naive::colMax(result, data, dim, numSamples);
}

template <typename Type>
inline void decayL1(Type* dst, Type* src, Type* lr, Type lambda, size_t len) {
  naive::decayL1(dst, src, lr, lambda, len);
}

template <typename Type>
inline void decayL1(Type* dst, Type* src, Type lambda, size_t len) {
  naive::decayL1(dst, src, lambda, len);
}

template <size_t AlignSize>
inline bool isPointerAlign(void* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % AlignSize == 0;
}

inline bool vec_check(size_t len) {
#ifdef __AVX__
  return len % 8 == 0;
#else
  return len % 4 == 0;
#endif
}

namespace internal {
void addToImpl(float* a, const float* b, size_t len);
void batchAddToImpl(float* a, const float* b[], int batch, size_t len);
void colMaxImpl(float* result, const float* data, int dim, int numSamples);
#ifdef __AVX__
void decayL1AvxImpl(float* dst, float* src, float lambda, size_t len);
void decayL1AvxImpl(
    float* dst, float* src, float* lr, float lambda, size_t len);
#endif
}  // namespace internal

template <>
inline void addTo(float* a, const float* b, size_t len) {
#ifdef __SSE3__
  internal::addToImpl(a, b, len);
#else
  naive::addTo(a, b, len);
#endif
}

template <>
inline void batchAddTo(float* a, const float* b[], int batch, size_t len) {
#ifdef __SSE3__
  internal::batchAddToImpl(a, b, batch, len);
#else
  naive::batchAddTo(a, b, batch, len);
#endif
}

template <>
inline void colMax(float* result, const float* data, int dim, int numSamples) {
#ifdef __SSE3__
  internal::colMaxImpl(result, data, dim, numSamples);
#else
  naive::colMax(result, data, dim, numSamples);
#endif
}

template <>
inline void decayL1(float* dst, float* src, float lambda, size_t len) {
#ifdef __AVX__
  internal::decayL1AvxImpl(dst, src, lambda, len);
#else
  naive::decayL1(dst, src, lambda, len);
#endif
}

template <>
inline void decayL1(
    float* dst, float* src, float* lr, float lambda, size_t len) {
#ifdef __AVX__
  internal::decayL1AvxImpl(dst, src, lr, lambda, len);
#else
  naive::decayL1(dst, src, lr, lambda, len);
#endif
}

}  // namespace simd

}  // namespace paddle
