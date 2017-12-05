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


#ifndef HL_MATRIX_BASE_CUH_
#define HL_MATRIX_BASE_CUH_

#include "hl_matrix_type.cuh"

class BaseOp {
public:
  static const bool sse = false;
  BaseOp() {}
  explicit BaseOp(const real s1) {}
  explicit BaseOp(const real s1, const real s2) {}
  INLINE vecType vecOp(const vecType a) const {
    return a;
  }
  INLINE vecType vecOp(const vecType a, const vecType b) const {
    return a;
  }
};

#ifdef __CUDA_ARCH__
typedef BaseOp SSESum;
typedef BaseOp SSEMax;
typedef BaseOp SSEMin;
typedef BaseOp SSEIdentity;
typedef BaseOp SSEAdd;
typedef BaseOp SSEAdd2;
typedef BaseOp SSESub;
typedef BaseOp SSEMul;
typedef BaseOp SSEDiv;
typedef BaseOp SSESquaredDiff;
typedef BaseOp SSEFirst;
typedef BaseOp SSESecond;
typedef BaseOp SSEClassificationError;
#else
#include "hl_matrix_base_detail.cuh"
#endif

namespace aggregate {
class sum : public SSESum {
public:
  INLINE real init() { return 0.0f; }
  INLINE real operator()(const real a, const real b) const {
    return a + b;
  }
};

class max : public SSEMax {
public:
  INLINE real init() { return -HL_FLOAT_MAX; }
  INLINE real operator()(const real a, const real b) const {
    return a > b ? a : b;
  }
};

class min : public SSEMin {
public:
  INLINE real init() {return HL_FLOAT_MAX;}
  INLINE real operator()(const real a, const real b) const {
    return a > b ? b : a;
  }
};
}  // namespace aggregate

namespace base {
namespace unary {
class identity : public SSEIdentity {
public:
  INLINE real operator()(const real a) const {
    return a;
  }
};
}  // namespace unary

namespace binary {
class add : public SSEAdd {
public:
  INLINE real operator()(const real a, const real b) const {
    return a + b;
  }
};

class add2 : public SSEAdd2 {
private:
  const real p1;
  const real p2;
public:
  add2(const real s1, const real s2)
    : SSEAdd2(s1, s2), p1(s1), p2(s2) {}
  INLINE real operator()(const real a, const real b) const {
    return p1 * a + p2 * b;
  }
};

class sub : public SSESub {
public:
  INLINE real operator()(const real a, const real b) const {
    return a - b;
  }
};

class mul : public SSEMul {
public:
  INLINE real operator()(const real a, const real b) const {
    return a * b;
  }
};

class div : public SSEDiv {
public:
  INLINE real operator()(const real a, const real b) const  {
    return a / b;
  }
};

class squaredDiff : public SSESquaredDiff {
public:
  INLINE real operator()(const real a, const real b) const {
    return (a - b) * (a - b);
  }
};

class first : public SSEFirst {
public:
  INLINE real operator()(const real a, const real b) const {
    return a;
  }
};

class second : public SSESecond {
public:
  INLINE real operator()(const real a, const real b) const {
    return b;
  }
};

class classificationError : public SSEClassificationError {
private:
  const real p;
public:
  explicit classificationError(const real s)
    : SSEClassificationError(s), p(s) {}
  INLINE real operator()(const real a, const real b) const {
    return ((a > p) == (b > p)) ? 0.0f : 1.0f;
  }
};
}  // namespace binary
}  // namespace base

#endif /* HL_MATRIX_BASE_CUH_ */
