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

#ifndef HL_TENSOR_OPS_H_
#define HL_TENSOR_OPS_H_

#include <cmath>
#include "hl_matrix_type.cuh"

namespace hppl {
namespace unary {

template <class T>
class add_scale {
private:
  const T p;

public:
  INLINE add_scale(const T s) : p(s) {}
  INLINE T operator()(const T a) const { return a + p; }
};

template <class T>
class sub_scale {
private:
  const T p;

public:
  INLINE sub_scale(const T s) : p(s) {}
  INLINE T operator()(const T a) const { return a - p; }
};

template <class T>
class mul_scale {
private:
  const T p;

public:
  INLINE mul_scale(const T s) : p(s) {}
  INLINE T operator()(const T a) const { return a * p; }
};

template <class T>
class div_scale {
private:
  const T p;

public:
  INLINE div_scale(const T s) : p(s) {}
  INLINE T operator()(const T a) const { return a / p; }
};

template <class T>
class neg {
public:
  INLINE T operator()(const T a) const { return -a; }
};

template <class T>
class exp_op {
public:
  INLINE T operator()(const T a) const { return std::exp(a); }
};

template <class T>
class log_op {
public:
  INLINE T operator()(const T a) const { return std::log(a); }
};

template <class T>
class sqrt_op {
public:
  INLINE T operator()(const T a) const { return std::sqrt(a); }
};

template <class T>
class square {
public:
  INLINE T operator()(const T a) const { return a * a; }
};

template <class T>
class reciprocal {
public:
  INLINE T operator()(const T a) const { return T(1) / a; }
};

template <class T>
class abs {
public:
  INLINE T operator()(const T a) const { return a > 0 ? a : -a; }
};

template <class T>
class sign {
public:
  INLINE T operator()(const T a) const { return (a > 0) - (a < 0); }
};

template <class T>
class min {
private:
  const T p;

public:
  INLINE min(const T s) : p(s) {}
  INLINE T operator()(const T a) const { return a > p ? p : a; }
};

template <class T>
class max {
private:
  const T p;

public:
  INLINE max(const T s) : p(s) {}
  INLINE T operator()(const T a) const { return a < p ? p : a; }
};

template <class T>
class pow_op {
private:
  const T p;

public:
  INLINE pow_op(const T s) : p(s) {}
  INLINE T operator()(const T a) const { return std::pow(a, p); }
};

template <class T>
class constant {
private:
  const T p;

public:
  INLINE constant(const T s) : p(s) {}
  INLINE T operator()(int i) const { return p; }
  INLINE T operator()(int i, int j) const { return p; }
};

template <class T>
class cmp_eq {
private:
  const T p;

public:
  INLINE cmp_eq(const T s) : p(s) {}
  INLINE bool operator()(const T a) const { return a == p; }
};

template <class T>
class cmp_ne {
private:
  const T p;

public:
  INLINE cmp_ne(const T s) : p(s) {}
  INLINE bool operator()(const T a) const { return a != p; }
};

template <class T>
class cmp_le {
private:
  const T p;

public:
  INLINE cmp_le(const T s) : p(s) {}
  INLINE bool operator()(const T a) const { return a <= p; }
};

template <class T>
class cmp_lt {
private:
  const T p;

public:
  INLINE cmp_lt(const T s) : p(s) {}
  INLINE bool operator()(const T a) const { return a < p; }
};

template <class T>
class cmp_ge {
private:
  const T p;

public:
  INLINE cmp_ge(const T s) : p(s) {}
  INLINE bool operator()(const T a) const { return a >= p; }
};

template <class T>
class cmp_gt {
private:
  const T p;

public:
  INLINE cmp_gt(const T s) : p(s) {}
  INLINE bool operator()(const T a) const { return a > p; }
};

template <class T>
class and_op {
private:
  const T p;

public:
  INLINE and_op(const T s) : p(s) {}
  INLINE bool operator()(const T a) const { return a && p; }
};

template <class T>
class or_op {
private:
  const T p;

public:
  INLINE or_op(const T s) : p(s) {}
  INLINE bool operator()(const T a) const { return a || p; }
};

}  // namespace unary

namespace binary {
template <class T>
class add {
public:
  INLINE T operator()(const T a, const T b) const { return a + b; }
};

template <class T>
class add_scale {
private:
  const T p1;
  const T p2;

public:
  INLINE add_scale(const T s1, const T s2) : p1(s1), p2(s2) {}
  INLINE T operator()(const T a, const T b) const { return p1 * a + p2 * b; }
};

template <class T>
class sub {
public:
  INLINE T operator()(const T a, const T b) const { return a - b; }
};

template <class T>
class mul {
public:
  INLINE T operator()(const T a, const T b) const { return a * b; }
};

template <class T>
class div {
public:
  INLINE T operator()(const T a, const T b) const { return a / b; }
};

template <class T>
class cmp_eq {
public:
  INLINE bool operator()(const T a, const T b) const { return a == b; }
};

template <class T>
class cmp_ne {
public:
  INLINE bool operator()(const T a, const T b) const { return a != b; }
};

template <class T>
class cmp_le {
public:
  INLINE bool operator()(const T a, const T b) const { return a <= b; }
};

template <class T>
class cmp_lt {
public:
  INLINE bool operator()(const T a, const T b) const { return a < b; }
};

template <class T>
class cmp_ge {
public:
  INLINE bool operator()(const T a, const T b) const { return a >= b; }
};

template <class T>
class cmp_gt {
public:
  INLINE bool operator()(const T a, const T b) const { return a > b; }
};

template <class T>
class and_op {
public:
  INLINE bool operator()(const T a, const T b) const { return a && b; }
};

template <class T>
class or_op {
public:
  INLINE bool operator()(const T a, const T b) const { return a || b; }
};

template <class T>
class min {
public:
  INLINE T operator()(const T a, const T b) const { return a > b ? b : a; }
};

template <class T>
class max {
public:
  INLINE T operator()(const T a, const T b) const { return a < b ? b : a; }
};

#ifdef PADDLE_USE_SSE3
#ifndef PADDLE_TYPE_DOUBLE
template <>
class add<__m128> {
public:
  INLINE __m128 operator()(const __m128 a, const __m128 b) const {
    return _mm_add_ps(a, b);
  }
};

template <>
class add_scale<__m128> {
private:
  const __m128 p1;
  const __m128 p2;

public:
  INLINE add_scale(const __m128 s1, const __m128 s2) : p1(s1), p2(s2) {}
  INLINE __m128 operator()(const __m128 a, const __m128 b) const {
    return _mm_add_ps(_mm_mul_ps(p1, a), _mm_mul_ps(p2, b));
  }
};

template <>
class sub<__m128> {
public:
  INLINE __m128 operator()(const __m128 a, const __m128 b) const {
    return _mm_sub_ps(a, b);
  }
};

template <>
class mul<__m128> {
public:
  INLINE __m128 operator()(const __m128 a, const __m128 b) const {
    return _mm_mul_ps(a, b);
  }
};

template <>
class div<__m128> {
public:
  INLINE __m128 operator()(const __m128 a, const __m128 b) const {
    return _mm_div_ps(a, b);
  }
};

template <>
class min<__m128> {
public:
  INLINE __m128 operator()(const __m128 a, const __m128 b) const {
    return _mm_min_ps(a, b);
  }
};

template <>
class max<__m128> {
public:
  INLINE __m128 operator()(const __m128 a, const __m128 b) const {
    return _mm_max_ps(a, b);
  }
};
#else
template <>
class add<__m128d> {
public:
  INLINE __m128d operator()(const __m128d a, const __m128d b) const {
    return _mm_add_pd(a, b);
  }
};

template <>
class add_scale<__m128d> {
private:
  const __m128d p1;
  const __m128d p2;

public:
  INLINE add_scale(const __m128d s1, const __m128d s2) : p1(s1), p2(s2) {}
  INLINE __m128d operator()(const __m128d a, const __m128d b) const {
    return _mm_add_pd(_mm_mul_pd(p1, a), _mm_mul_pd(p2, b));
  }
};

template <>
class sub<__m128d> {
public:
  INLINE __m128d operator()(const __m128d a, const __m128d b) const {
    return _mm_sub_pd(a, b);
  }
};

template <>
class mul<__m128d> {
public:
  INLINE __m128d operator()(const __m128d a, const __m128d b) const {
    return _mm_mul_pd(a, b);
  }
};

template <>
class div<__m128d> {
public:
  INLINE __m128d operator()(const __m128d a, const __m128d b) const {
    return _mm_div_pd(a, b);
  }
};

template <>
class min<__m128d> {
public:
  INLINE __m128d operator()(const __m128d a, const __m128d b) const {
    return _mm_min_pd(a, b);
  }
};

template <>
class max<__m128d> {
public:
  INLINE __m128d operator()(const __m128d a, const __m128d b) const {
    return _mm_max_pd(a, b);
  }
};
#endif  // PADDLE_TYPE_DOUBLE
#endif  // PADDLE_USE_SSE3

#ifdef PADDLE_USE_NEON
#ifndef PADDLE_TYPE_DOUBLE
template <>
class add<float32x4_t> {
public:
  INLINE float32x4_t operator()(const float32x4_t a,
                                const float32x4_t b) const {
    return vmulq_f32(a, b);
  }
};

template <>
class add_scale<float32x4_t> {
private:
  const float32x4_t p1;
  const float32x4_t p2;

public:
  INLINE add_scale(const float32x4_t s1, const float32x4_t s2)
      : p1(s1), p2(s2) {}
  INLINE float32x4_t operator()(const float32x4_t a,
                                const float32x4_t b) const {
    return vaddq_f32(vmulq_f32(p1, a), vmulq_f32(p2, b));
  }
};

template <>
class sub<float32x4_t> {
public:
  INLINE float32x4_t operator()(const float32x4_t a,
                                const float32x4_t b) const {
    return vsubq_f32(a, b);
  }
};

template <>
class mul<float32x4_t> {
public:
  INLINE float32x4_t operator()(const float32x4_t a,
                                const float32x4_t b) const {
    return vmulq_f32(a, b);
  }
};

template <>
class div<float32x4_t> {
public:
  INLINE float32x4_t operator()(const float32x4_t a,
                                const float32x4_t b) const {
    float32x4_t tmp = vrecpeq_f32(b);
    return vmulq_f32(a, tmp);
  }
};

template <>
class min<float32x4_t> {
public:
  INLINE float32x4_t operator()(const float32x4_t a,
                                const float32x4_t b) const {
    return vminq_f32(a, b);
  }
};

template <>
class max<float32x4_t> {
public:
  INLINE float32x4_t operator()(const float32x4_t a,
                                const float32x4_t b) const {
    return vmaxq_f32(a, b);
  }
};
#else
#error To be implemented
#endif  // PADDLE_TYPE_DOUBLE
#endif  // PADDLE_USE_NEON

}  // namespace binary
}  // namespace hppl

#endif  // HL_TENSOR_OPS_H_
