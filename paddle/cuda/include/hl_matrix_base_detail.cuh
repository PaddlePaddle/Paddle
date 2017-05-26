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

#ifndef HL_MATRIX_BASE_DETAIL_CUH_
#define HL_MATRIX_BASE_DETAIL_CUH_

#include "hl_matrix_type.cuh"

namespace aggregate {
class SSESum {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hl_vec_add(a, b);
  }
};

class SSEMax {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hl_vec_max(a, b);
  }
};

class SSEMin {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hl_vec_min(a, b);
  }
};
}  // namespace aggregate

namespace base {
namespace unary {
class SSEIdentity {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a) const {
    return a;
  }
};
}  // namespace unary

namespace binary {
class SSEAdd {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hl_vec_add(a, b);
  }
};

class SSEAdd2 {
public:
  static const bool sse = VECTOR_SIMD;
  const real p1;
  const real p2;
  vecType mp1;
  vecType mp2;

public:
  SSEAdd2(const real s1, const real s2) : p1(s1), p2(s2) {
    mp1 = hl_vec_set(p1);
    mp2 = hl_vec_set(p2);
  }
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hl_vec_add(hl_vec_mul(mp1, a), hl_vec_mul(mp2, b));
  }
};

class SSESub {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hl_vec_sub(a, b);
  }
};

class SSEMul {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hl_vec_mul(a, b);
  }
};

class SSEDiv {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hl_vec_div(a, b);
  }
};

class SSESquaredDiff {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hl_vec_mul(hl_vec_sub(a, b), hl_vec_sub(a, b));
  }
};

class SSEFirst {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return a;
  }
};

class SSESecond {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return b;
  }
};

class SSEClassificationError {
public:
  static const bool sse = VECTOR_SIMD;
  const real p;
  vecType mp;
  vecType result;

public:
  explicit SSEClassificationError(const real s) : p(s) {
    mp = hl_vec_set(p);
    result = hl_vec_set(1.0f);
  }
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hl_vec_classification_error(a, b, mp, result);
  }
};
}  // namespace binary
}  // namespace base

#endif /* HL_MATRIX_BASE_DETAIL_CUH_ */
