/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include "DisableCopy.h"

namespace paddle {

class SIMDFlags final {
public:
    DISABLE_COPY(SIMDFlags);

    SIMDFlags();

    static SIMDFlags* instance();

    inline bool isSSE()       { return simd_flags_ & SIMD_SSE;   }
    inline bool isSSE2()      { return simd_flags_ & SIMD_SSE2;  }
    inline bool isSSE3()      { return simd_flags_ & SIMD_SSE3;  }
    inline bool isSSSE3()     { return simd_flags_ & SIMD_SSSE3; }
    inline bool isSSE41()     { return simd_flags_ & SIMD_SSE41; }
    inline bool isSSE42()     { return simd_flags_ & SIMD_SSE42; }
    inline bool isFMA3()      { return simd_flags_ & SIMD_FMA3;  }
    inline bool isFMA4()      { return simd_flags_ & SIMD_FMA4;  }
    inline bool isAVX()       { return simd_flags_ & SIMD_AVX;   }
    inline bool isAVX2()      { return simd_flags_ & SIMD_AVX2;  }
    inline bool isAVX512()    { return simd_flags_ & SIMD_AVX512;}

private:
    enum simd_t {
        SIMD_NONE     = 0,        ///< None
        SIMD_SSE      = 1 << 0,   ///< SSE
        SIMD_SSE2     = 1 << 1,   ///< SSE 2
        SIMD_SSE3     = 1 << 2,   ///< SSE 3
        SIMD_SSSE3    = 1 << 3,   ///< SSSE 3
        SIMD_SSE41    = 1 << 4,   ///< SSE 4.1
        SIMD_SSE42    = 1 << 5,   ///< SSE 4.2
        SIMD_FMA3     = 1 << 6,   ///< FMA 3
        SIMD_FMA4     = 1 << 7,   ///< FMA 4
        SIMD_AVX      = 1 << 8,   ///< AVX
        SIMD_AVX2     = 1 << 9,   ///< AVX 2
        SIMD_AVX512   = 1 << 10,  ///< AVX 512
    };

    /// simd flags
    int simd_flags_ = SIMD_NONE;
};

#define HAS_SSE      SIMDFlags::instance()->isSSE()
#define HAS_SSE2     SIMDFlags::instance()->isSSE2()
#define HAS_SSE3     SIMDFlags::instance()->isSSE3()
#define HAS_SSSE3    SIMDFlags::instance()->isSSSE3()
#define HAS_SSE41    SIMDFlags::instance()->isSSE41()
#define HAS_SSS42    SIMDFlags::instance()->isSSE42()
#define HAS_FMA3     SIMDFlags::instance()->isFMA3()
#define HAS_FMA4     SIMDFlags::instance()->isFMA4()
#define HAS_AVX      SIMDFlags::instance()->isAVX()
#define HAS_AVX2     SIMDFlags::instance()->isAVX2()
#define HAS_AVX512   SIMDFlags::instance()->isAVX512()

}   // namespace paddle
