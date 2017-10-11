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


#ifndef HL_MATRIX_BASE_NEON_CUH_
#define HL_MATRIX_BASE_NEON_CUH_

namespace aggregate {
class SSESum {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    return vaddq_f32(a, b);
  }
};

class SSEMax {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    return vmaxq_f32(a, b);
  }
};

class SSEMin {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    return vminq_f32(a, b);
  }
};
}  // namespace aggregate

namespace base {
namespace unary {
class SSEIdentity {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a) const {
    return a;
  }
};
}  // namespace unary

namespace binary {
class SSEAdd {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    return vaddq_f32(a, b);
  }
};

class SSEAdd2 {
public:
  static const bool sse = true;
  const real p1;
  const real p2;
  float32x4_t mp1;
  float32x4_t mp2;

public:
  SSEAdd2(const real s1, const real s2) : p1(s1), p2(s2) {
    mp1 = vdupq_n_f32(p1);
    mp2 = vdupq_n_f32(p2);
  }
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    float32x4_t tmp1, tmp2;
    tmp1 = vmulq_f32(mp1, a);
    tmp2 = vmulq_f32(mp2, b);
    return vaddq_f32(tmp1, tmp2);
  }
};

class SSESub {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    return vsubq_f32(a, b);
  }
};

class SSEMul {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    return vmulq_f32(a, b);
  }
};

class SSEDiv {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    float32x4_t tmp;
    tmp = vrecpeq_f32(b);
    return vmulq_f32(a, tmp);
  }
};

class SSESquaredDiff {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    float32x4_t tmp;
    tmp = vsubq_f32(a, b);
    return vmulq_f32(tmp, tmp);
  }
};

class SSEFirst {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    return a;
  }
};

class SSESecond {
public:
  static const bool sse = true;
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    return b;
  }
};

class SSEClassificationError {
public:
  static const bool sse = true;
  const real p;
  float32x4_t mp;
  uint32x4_t result;

public:
  explicit SSEClassificationError(const real s) : p(s) {
    mp = vdupq_n_f32(p);
    result = vdupq_n_u32(1);
  }
  // TODO: to be check
  INLINE float32x4_t vecOp(const float32x4_t a, const float32x4_t b) const {
    uint32x4_t tmp1 = vcgtq_f32(a, mp);
    uint32x4_t tmp2 = vcgtq_f32(b, mp);
    uint32x4_t tmp3 = veorq_u32(tmp1, tmp2);
    return vcvtq_f32_u32(vandq_u32(tmp3, result));
  }
};
}  // namespace binary
}  // namespace base

#endif /* HL_MATRIX_BASE_NEON_CUH_ */
