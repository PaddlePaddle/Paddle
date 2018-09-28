/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <string>
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

namespace jit = platform::jit;

#define NEW_JITKERNEL_IMPL(src, t, isa, k) \
  p = std::dynamic_pointer_cast<src<t>>(   \
      std::make_shared<src##Impl<t, isa, k>>())

#define SEARCH_BLOCK(src, t, isa)                             \
  if (d < AVX_FLOAT_BLOCK) {                                  \
    NEW_JITKERNEL_IMPL(src, t, isa, kLT8);                    \
  } else if (d == AVX_FLOAT_BLOCK) {                          \
    NEW_JITKERNEL_IMPL(src, t, isa, kEQ8);                    \
  } else if (d > AVX_FLOAT_BLOCK && d < AVX512_FLOAT_BLOCK) { \
    NEW_JITKERNEL_IMPL(src, t, isa, kGT8LT16);                \
  } else if (d == AVX512_FLOAT_BLOCK) {                       \
    NEW_JITKERNEL_IMPL(src, t, isa, kEQ16);                   \
  } else {                                                    \
    NEW_JITKERNEL_IMPL(src, t, isa, kGT16);                   \
  }

#define SEARCH_ISA_BLOCK(src, t)        \
  if (jit::MayIUse(jit::avx512f)) {     \
    SEARCH_BLOCK(src, t, jit::avx512f); \
  } else if (jit::MayIUse(jit::avx2)) { \
    SEARCH_BLOCK(src, t, jit::avx2);    \
  } else if (jit::MayIUse(jit::avx)) {  \
    SEARCH_BLOCK(src, t, jit::avx);     \
  } else {                              \
    SEARCH_BLOCK(src, t, jit::isa_any); \
  }

#define JITKERNEL_WITH_DTYPE(ker_key, ker_class, ker_dtype, dtype_key)     \
  template <>                                                              \
  const std::shared_ptr<ker_class<ker_dtype>>                              \
  KernelPool::Get<ker_class<ker_dtype>>(int d) {                           \
    std::string key = #ker_key #dtype_key + std::to_string(d);             \
    if (kers_.find(key) == kers_.end()) {                                  \
      std::shared_ptr<ker_class<ker_dtype>> p;                             \
      SEARCH_ISA_BLOCK(ker_class, ker_dtype);                              \
      kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});           \
      return p;                                                            \
    }                                                                      \
    return std::dynamic_pointer_cast<ker_class<ker_dtype>>(kers_.at(key)); \
  }

#define REGISTER_JITKERNEL(ker_key, ker_class)        \
  JITKERNEL_WITH_DTYPE(ker_key, ker_class, float, f); \
  JITKERNEL_WITH_DTYPE(ker_key, ker_class, double, d)

#define FOR_EACH_ISA(macro_, block) \
  macro_(jit::avx512f, block);      \
  macro_(jit::avx2, block);         \
  macro_(jit::avx, block);          \
  macro_(jit::isa_any, block)

#define FOR_EACH_BLOCK(macro_, isa) \
  macro_(isa, kLT8);                \
  macro_(isa, kEQ8);                \
  macro_(isa, kGT8LT16);            \
  macro_(isa, kEQ16);               \
  macro_(isa, kGT16)

#define FOR_EACH_ISA_BLOCK(macro_)      \
  FOR_EACH_BLOCK(macro_, jit::avx512f); \
  FOR_EACH_BLOCK(macro_, jit::avx2);    \
  FOR_EACH_BLOCK(macro_, jit::avx);     \
  FOR_EACH_BLOCK(macro_, jit::isa_any)

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
