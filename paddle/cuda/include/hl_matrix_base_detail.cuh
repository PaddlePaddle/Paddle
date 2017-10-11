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
#include "hl_tensor_ops.h"

namespace aggregate {
class SSESum {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hppl::binary::add<vecType>()(a, b);
  }
};

class SSEMax {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hppl::binary::max<vecType>()(a, b);
  }
};

class SSEMin {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hppl::binary::min<vecType>()(a, b);
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
    return hppl::binary::add<vecType>()(a, b);
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
    return hppl::binary::add_scale<vecType>(mp1, mp2)(a, b);
  }
};

class SSESub {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hppl::binary::sub<vecType>()(a, b);
  }
};

class SSEMul {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hppl::binary::mul<vecType>()(a, b);
  }
};

class SSEDiv {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return hppl::binary::div<vecType>()(a, b);
  }
};

class SSESquaredDiff {
public:
  static const bool sse = VECTOR_SIMD;
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    vecType tmp = hppl::binary::sub<vecType>()(a, b);
    return hppl::binary::mul<vecType>()(tmp, tmp);
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
