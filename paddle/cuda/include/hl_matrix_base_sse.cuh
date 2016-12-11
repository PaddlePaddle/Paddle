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


#ifndef HL_MATRIX_BASE_SSE_CUH_
#define HL_MATRIX_BASE_SSE_CUH_

namespace aggregate {
class SSESum {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    return _mm_add_ps(a, b);
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    return _mm_add_pd(a, b);
  }
};

class SSEMax {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    return _mm_max_ps(a, b);
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    return _mm_max_pd(a, b);
  }
};

class SSEMin {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    return _mm_min_ps(a, b);
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    return _mm_min_pd(a, b);
  }
};
}  // namespace aggregate

namespace base {
namespace unary {
class SSEIdentity {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a) const {
    return a;
  }
  INLINE __m128d vecOp(const __m128d a) const {
    return a;
  }
};
}  // namespace unary

namespace binary {
class SSEAdd {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    return _mm_add_ps(a, b);
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    return _mm_add_pd(a, b);
  }
};

class SSEAdd2 {
public:
  static const bool sse = true;
  const real p1;
  const real p2;
  union {__m128 f; __m128d d;} mp1;
  union {__m128 f; __m128d d;} mp2;

public:
  SSEAdd2(const real s1, const real s2) : p1(s1), p2(s2) {
    if (sizeof(real) == sizeof(float)) {
      mp1.f = _mm_set1_ps(p1);
      mp2.f = _mm_set1_ps(p2);
    } else {
      mp1.d = _mm_set1_pd(p1);
      mp2.d = _mm_set1_pd(p2);
    }
  }
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    __m128 tmp1, tmp2;
    tmp1 = _mm_mul_ps(mp1.f, a);
    tmp2 = _mm_mul_ps(mp2.f, b);
    return _mm_add_ps(tmp1, tmp2);
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    __m128d tmp1, tmp2;
    tmp1 = _mm_mul_pd(mp1.d, a);
    tmp2 = _mm_mul_pd(mp2.d, b);
    return _mm_add_pd(tmp1, tmp2);
  }
};

class SSESub {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    return _mm_sub_ps(a, b);
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    return _mm_sub_pd(a, b);
  }
};

class SSEMul {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    return _mm_mul_ps(a, b);
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    return _mm_mul_pd(a, b);
  }
};

class SSEDiv {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    return _mm_div_ps(a, b);
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    return _mm_div_pd(a, b);
  }
};

class SSESquaredDiff {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    return _mm_mul_ps(_mm_sub_ps(a, b), _mm_sub_ps(a, b));
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    return _mm_mul_pd(_mm_sub_pd(a, b), _mm_sub_pd(a, b));
  }
};

class SSEFirst {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    return a;
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    return a;
  }
};

class SSESecond {
public:
  static const bool sse = true;
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    return b;
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    return b;
  }
};

class SSEClassificationError {
public:
  static const bool sse = true;
  const real p;
  union {__m128 f; __m128d d;} mp;
  union {__m128 f; __m128d d;} result;

public:
  explicit SSEClassificationError(const real s) : p(s) {
    if (sizeof(real) == sizeof(float)) {
      mp.f = _mm_set1_ps(p);
      result.f = _mm_set1_ps(1.0f);
    } else {
      mp.d = _mm_set1_pd(p);
      result.d = _mm_set1_pd(1.0);
    }
  }
  INLINE __m128 vecOp(const __m128 a, const __m128 b) const {
    __m128 tmp1 = _mm_cmpgt_ps(a, mp.f);
    __m128 tmp2 = _mm_cmpgt_ps(b, mp.f);
    __m128 tmp3 = _mm_xor_ps(tmp1, tmp2);
    return _mm_and_ps(tmp3, result.f);
  }
  INLINE __m128d vecOp(const __m128d a, const __m128d b) const {
    __m128d tmp1 = _mm_cmpgt_pd(a, mp.d);
    __m128d tmp2 = _mm_cmpgt_pd(b, mp.d);
    __m128d tmp3 = _mm_xor_pd(tmp1, tmp2);
    return _mm_and_pd(tmp3, result.d);
  }
};
}  // namespace binary
}  // namespace base

#endif /* HL_MATRIX_BASE_SSE_CUH_ */
