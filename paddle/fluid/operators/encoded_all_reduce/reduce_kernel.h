// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_REDUCE_KERNEL_H_
#define NCCL_REDUCE_KERNEL_H_

//#include "nccl.h"
#include <cstdint>
#include <limits>

template <typename T>
struct FuncNull {
  __device__ T operator()(const T x, const T y) const { return 0; }
};

template <typename T>
struct FuncSum {
  __device__ T operator()(const T x, const T y) const { return x + y; }
};

template <typename T>
struct FuncProd {
  __device__ T operator()(const T x, const T y) const { return x * y; }
};

template <typename T>
struct FuncMax {
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? y : x;
  }
};

template <typename T>
struct FuncMin {
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? x : y;
  }
};

template <>
struct FuncSum<int8_t> {
  union converter {
    uint32_t storage;
    char4 a;
  };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z = 0;
    asm("vadd4.s32.s32.s32 %0, %1, %2, %3;"
        : "=r"(rv)
        : "r"(x), "r"(y), "r"(z));
    return rv;
#elif (__CUDA_ARCH__ >= 500) && (__CUDA_ARCH__ < 700)
    int32_t rv;
    asm("vadd.s32.s32.s32 %0,    %1.b0, %2.b0;    \n\t"
        "vadd.s32.s32.s32 %0.b1, %1.b1, %2.b1, %0;\n\t"
        "vadd.s32.s32.s32 %0.b2, %1.b2, %2.b2, %0;\n\t"
        "vadd.s32.s32.s32 %0.b3, %1.b3, %2.b3, %0;"
        : "=r"(rv)
        : "r"(x), "r"(y));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = cx.a.x + cy.a.x;
    cr.a.y = cx.a.y + cy.a.y;
    cr.a.z = cx.a.z + cy.a.z;
    cr.a.w = cx.a.w + cy.a.w;
    return cr.storage;
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return x + y;
  }
};
template <>
struct FuncSum<uint8_t> {
  union converter {
    uint32_t storage;
    uchar4 a;
  };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z = 0;
    asm("vadd4.u32.u32.u32 %0, %1, %2, %3;"
        : "=r"(rv)
        : "r"(x), "r"(y), "r"(z));
    return rv;
#elif (__CUDA_ARCH__ >= 500) && (__CUDA_ARCH__ < 700)
    int32_t rv;
    asm("vadd.u32.u32.u32 %0,    %1.b0, %2.b0;    \n\t"
        "vadd.u32.u32.u32 %0.b1, %1.b1, %2.b1, %0;\n\t"
        "vadd.u32.u32.u32 %0.b2, %1.b2, %2.b2, %0;\n\t"
        "vadd.u32.u32.u32 %0.b3, %1.b3, %2.b3, %0;"
        : "=r"(rv)
        : "r"(x), "r"(y));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = cx.a.x + cy.a.x;
    cr.a.y = cx.a.y + cy.a.y;
    cr.a.z = cx.a.z + cy.a.z;
    cr.a.w = cx.a.w + cy.a.w;
    return cr.storage;
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return x + y;
  }
};

static __device__ uint32_t mulChar4(const uint32_t x, const uint32_t y) {
/* This can be used both for signed and unsigned 8-bit multiplication */
#if (__CUDA_ARCH__ >= 300)
  uint32_t rv;
  asm("{ .reg .u32 t0, t1, t2, t3;\n\t"
      " vmad.u32.u32.u32 t3, %1.b3, %2.b3, 0;\n\t"
      " vmad.u32.u32.u32 t2, %1.b2, %2.b2, 0;\n\t"
      " shl.b32          t3, t3, 16;\n\t"
      " shl.b32          t2, t2, 16;\n\t"
      " vmad.u32.u32.u32 t1, %1.b1, %2.b1, t3;\n\t"
      " shl.b32          t1, t1, 8;\n\t"
      " vmad.u32.u32.u32 t0, %1.b0, %2.b0, t2;\n\t"
      " and.b32          t1, t1, 0xff00ff00;\n\t"
      " and.b32          t0, t0, 0x00ff00ff;\n\t"
      " or.b32           %0,  t0, t1;\n\t"
      "}"
      : "=r"(rv)
      : "r"(x), "r"(y));
  return rv;
#else
  union converter {
    uint32_t storage;
    char4 a;
  };
  converter cx, cy, cr;
  cx.storage = x;
  cy.storage = y;
  cr.a.x = cx.a.x * cy.a.x;
  cr.a.y = cx.a.y * cy.a.y;
  cr.a.z = cx.a.z * cy.a.z;
  cr.a.w = cx.a.w * cy.a.w;
  return cr.storage;
#endif
}

template <>
struct FuncProd<int8_t> {
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
    return mulChar4(x, y);
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return x * y;
  }
};
template <>
struct FuncProd<uint8_t> {
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
    return mulChar4(x, y);
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return x * y;
  }
};

template <>
struct FuncMax<int8_t> {
  union converter {
    uint32_t storage;
    char4 a;
  };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z = 0;
    asm("vmax4.s32.s32.s32 %0, %1, %2, %3;"
        : "=r"(rv)
        : "r"(x), "r"(y), "r"(z));
    return rv;
#elif (__CUDA_ARCH__ >= 500) && (__CUDA_ARCH__ < 700)
    int32_t rv;
    asm("vmax.s32.s32.s32 %0,    %1.b0, %2.b0;    \n\t"
        "vmax.s32.s32.s32 %0.b1, %1.b1, %2.b1, %0;\n\t"
        "vmax.s32.s32.s32 %0.b2, %1.b2, %2.b2, %0;\n\t"
        "vmax.s32.s32.s32 %0.b3, %1.b3, %2.b3, %0;"
        : "=r"(rv)
        : "r"(x), "r"(y));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = max(cx.a.x, cy.a.x);
    cr.a.y = max(cx.a.y, cy.a.y);
    cr.a.z = max(cx.a.z, cy.a.z);
    cr.a.w = max(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return (x > y) ? x : y;
  }
};
template <>
struct FuncMax<uint8_t> {
  union converter {
    uint32_t storage;
    uchar4 a;
  };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z = 0;
    asm("vmax4.u32.u32.u32 %0, %1, %2, %3;"
        : "=r"(rv)
        : "r"(x), "r"(y), "r"(z));
    return rv;
#elif (__CUDA_ARCH__ >= 500) && (__CUDA_ARCH__ < 700)
    int32_t rv;
    asm("vmax.u32.u32.u32 %0,    %1.b0, %2.b0;    \n\t"
        "vmax.u32.u32.u32 %0.b1, %1.b1, %2.b1, %0;\n\t"
        "vmax.u32.u32.u32 %0.b2, %1.b2, %2.b2, %0;\n\t"
        "vmax.u32.u32.u32 %0.b3, %1.b3, %2.b3, %0;"
        : "=r"(rv)
        : "r"(x), "r"(y));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = max(cx.a.x, cy.a.x);
    cr.a.y = max(cx.a.y, cy.a.y);
    cr.a.z = max(cx.a.z, cy.a.z);
    cr.a.w = max(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return (x > y) ? x : y;
  }
};

template <>
struct FuncMin<int8_t> {
  union converter {
    uint32_t storage;
    char4 a;
  };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z = 0;
    asm("vmin4.s32.s32.s32 %0, %1, %2, %3;"
        : "=r"(rv)
        : "r"(x), "r"(y), "r"(z));
    return rv;
#elif (__CUDA_ARCH__ >= 500) && (__CUDA_ARCH__ < 700)
    int32_t rv;
    asm("vmin.s32.s32.s32 %0,    %1.b0, %2.b0;    \n\t"
        "vmin.s32.s32.s32 %0.b1, %1.b1, %2.b1, %0;\n\t"
        "vmin.s32.s32.s32 %0.b2, %1.b2, %2.b2, %0;\n\t"
        "vmin.s32.s32.s32 %0.b3, %1.b3, %2.b3, %0;"
        : "=r"(rv)
        : "r"(x), "r"(y));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = min(cx.a.x, cy.a.x);
    cr.a.y = min(cx.a.y, cy.a.y);
    cr.a.z = min(cx.a.z, cy.a.z);
    cr.a.w = min(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return (x < y) ? x : y;
  }
};
template <>
struct FuncMin<uint8_t> {
  union converter {
    uint32_t storage;
    uchar4 a;
  };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z = 0;
    asm("vmin4.u32.u32.u32 %0, %1, %2, %3;"
        : "=r"(rv)
        : "r"(x), "r"(y), "r"(z));
    return rv;
#elif (__CUDA_ARCH__ >= 500) && (__CUDA_ARCH__ < 700)
    int32_t rv;
    asm("vmin.u32.u32.u32 %0,    %1.b0, %2.b0;    \n\t"
        "vmin.u32.u32.u32 %0.b1, %1.b1, %2.b1, %0;\n\t"
        "vmin.u32.u32.u32 %0.b2, %1.b2, %2.b2, %0;\n\t"
        "vmin.u32.u32.u32 %0.b3, %1.b3, %2.b3, %0;"
        : "=r"(rv)
        : "r"(x), "r"(y));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = min(cx.a.x, cy.a.x);
    cr.a.y = min(cx.a.y, cy.a.y);
    cr.a.z = min(cx.a.z, cy.a.z);
    cr.a.w = min(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return (x < y) ? x : y;
  }
};

template <>
struct FuncSum<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hadd2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x + fy.x;
    fr.y = fx.y + fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hadd(x, y);
#else
    return __float2half(__half2float(x) + __half2float(y));
#endif
  }
};

template <>
struct FuncProd<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hmul2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x * fy.x;
    fr.y = fx.y * fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hmul(x, y);
#else
    return __float2half(__half2float(x) * __half2float(y));
#endif
  }
};

template <>
struct FuncMax<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fmaxf(fx.x, fy.x);
    fr.y = fmaxf(fx.y, fy.y);
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = fmaxf(fx, fy);
    return __float2half(fm);
  }
};

template <>
struct FuncMin<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fminf(fx.x, fy.x);
    fr.y = fminf(fx.y, fy.y);
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = fminf(fx, fy);
    return __float2half(fm);
  }
};
#endif  // REDUCE_KERNEL_H_
