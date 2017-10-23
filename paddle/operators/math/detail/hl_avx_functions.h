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

#ifndef HL_AVX_FUNCTIONS_H_
#define HL_AVX_FUNCTIONS_H_

#include <immintrin.h>

namespace hppl {
__m256 relu(const __m256 a);
__m256 sigmoid(const __m256 a);
__m256 tanh(const __m256 a);
__m256 linear(const __m256 a);

__m256 relu(const __m256 a, const __m256 b);
__m256 sigmoid(const __m256 a, const __m256 b);
__m256 tanh(const __m256 a, const __m256 b);
__m256 linear(const __m256 a, const __m256 b);
}  // namespace hppl

#endif  // HL_AVX_FUNCTIONS_H_
