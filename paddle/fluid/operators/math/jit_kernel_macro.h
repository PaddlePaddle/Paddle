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
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

#define JITKERNEL_DECLARE_STATIC_FUNC                       \
  static inline std::string name(int d) {                   \
    PADDLE_THROW("DType should be either float or double"); \
  }                                                         \
  static inline bool useJIT(int d) { return false; }        \
  static inline bool useMKL(int d) { return false; }

#define JITKERNEL_DEFINE_NAME(ker_key, ker_class)    \
  template <>                                        \
  std::string ker_class##Impl<float>::name(int d) {  \
    std::string key(#ker_key "f");                   \
    if (useJIT(d)) {                                 \
      /* only jit code need record d*/               \
      return key + "jit" + std::to_string(d);        \
    } else if (useMKL(d)) {                          \
      return key + "mkl";                            \
    } else {                                         \
      return key + "any";                            \
    }                                                \
  }                                                  \
  template <>                                        \
  std::string ker_class##Impl<double>::name(int d) { \
    std::string key(#ker_key "d");                   \
    /* jit code do not support double yet*/          \
    if (useMKL(d)) {                                 \
      return key + "mkl";                            \
    } else {                                         \
      return key + "any";                            \
    }                                                \
  }

#define JITKERNEL_DECLARE(ker_class, ker_dtype) \
  template <>                                   \
  std::shared_ptr<const ker_class<ker_dtype>>   \
  KernelPool::Get<ker_class<ker_dtype>, int>(int d)

#define JITKERNEL_FIND_KEY(ker_class, ker_dtype) \
  std::string key = ker_class##Impl<ker_dtype>::name(d)

#define JITKERNEL_IMPL(ker_class, ker_dtype)           \
  p = std::dynamic_pointer_cast<ker_class<ker_dtype>>( \
      std::make_shared<ker_class##Impl<ker_dtype>>(d))

#define REGISTER_JITKERNEL_WITH_DTYPE(ker_class, ker_dtype, marco_declare, \
                                      macro_find_key, macro_impl)          \
  marco_declare(ker_class, ker_dtype) {                                    \
    macro_find_key(ker_class, ker_dtype);                                  \
    if (kers_.find(key) == kers_.end()) {                                  \
      std::shared_ptr<ker_class<ker_dtype>> p;                             \
      macro_impl(ker_class, ker_dtype);                                    \
      kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});           \
      return p;                                                            \
    }                                                                      \
    return std::dynamic_pointer_cast<const ker_class<ker_dtype>>(          \
        kers_.at(key));                                                    \
  }

#define REGISTER_JITKERNEL_ARGS(ker_key, ker_class, marco_define_name,     \
                                marco_declare, macro_find_key, macro_impl) \
  marco_define_name(ker_key, ker_class);                                   \
  REGISTER_JITKERNEL_WITH_DTYPE(ker_class, float, marco_declare,           \
                                macro_find_key, macro_impl);               \
  REGISTER_JITKERNEL_WITH_DTYPE(ker_class, double, marco_declare,          \
                                macro_find_key, macro_impl)

#define REGISTER_JITKERNEL(ker_key, ker_class)                       \
  REGISTER_JITKERNEL_ARGS(ker_key, ker_class, JITKERNEL_DEFINE_NAME, \
                          JITKERNEL_DECLARE, JITKERNEL_FIND_KEY,     \
                          JITKERNEL_IMPL)

namespace jit = platform::jit;
// TODO(TJ): below defines are deprecated, would be remove recently
#define SEARCH_BLOCK(macro_, ker, dtype, isa)              \
  if (d < YMM_FLOAT_BLOCK) {                               \
    macro_(ker, dtype, isa, kLT8);                         \
  } else if (d == YMM_FLOAT_BLOCK) {                       \
    macro_(ker, dtype, isa, kEQ8);                         \
  } else if (d > YMM_FLOAT_BLOCK && d < ZMM_FLOAT_BLOCK) { \
    macro_(ker, dtype, isa, kGT8LT16);                     \
  } else if (d == ZMM_FLOAT_BLOCK) {                       \
    macro_(ker, dtype, isa, kEQ16);                        \
  } else {                                                 \
    macro_(ker, dtype, isa, kGT16);                        \
  }

#define SEARCH_ISA_BLOCK(macro_, ker, dtype)        \
  if (jit::MayIUse(jit::avx512f)) {                 \
    SEARCH_BLOCK(macro_, ker, dtype, jit::avx512f); \
  } else if (jit::MayIUse(jit::avx2)) {             \
    SEARCH_BLOCK(macro_, ker, dtype, jit::avx2);    \
  } else if (jit::MayIUse(jit::avx)) {              \
    SEARCH_BLOCK(macro_, ker, dtype, jit::avx);     \
  } else {                                          \
    SEARCH_BLOCK(macro_, ker, dtype, jit::isa_any); \
  }

#define JITKERNEL_KEY(ker_key, dtype_key) \
  #ker_key #dtype_key + std::to_string(d)

#define JITKERNEL_NEW_IMPL_DEPRECATED(ker, dtype, isa, k) \
  p = std::dynamic_pointer_cast<ker<dtype>>(              \
      std::make_shared<ker##Impl<dtype, isa, k>>(d))

#define JITKERNEL_WITH_DTYPE_DEPRECATED(ker_key, ker_class, ker_dtype,       \
                                        dtype_key, marco_declare, macro_key, \
                                        macro_impl)                          \
  marco_declare(ker_class, ker_dtype) {                                      \
    std::string key = macro_key(ker_key, dtype_key);                         \
    if (kers_.find(key) == kers_.end()) {                                    \
      std::shared_ptr<ker_class<ker_dtype>> p;                               \
      SEARCH_ISA_BLOCK(macro_impl, ker_class, ker_dtype);                    \
      kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});             \
      return p;                                                              \
    }                                                                        \
    return std::dynamic_pointer_cast<const ker_class<ker_dtype>>(            \
        kers_.at(key));                                                      \
  }

#define REGISTER_JITKERNEL_DEPRECATED(ker_key, ker_class)           \
  JITKERNEL_WITH_DTYPE_DEPRECATED(ker_key, ker_class, float, f,     \
                                  JITKERNEL_DECLARE, JITKERNEL_KEY, \
                                  JITKERNEL_NEW_IMPL_DEPRECATED);   \
  JITKERNEL_WITH_DTYPE_DEPRECATED(ker_key, ker_class, double, d,    \
                                  JITKERNEL_DECLARE, JITKERNEL_KEY, \
                                  JITKERNEL_NEW_IMPL_DEPRECATED)

#define REGISTER_JITKERNEL_ARGS_DEPRECATED(ker_key, ker_class, marco_declare,  \
                                           macro_key, macro_impl)              \
  JITKERNEL_WITH_DTYPE_DEPRECATED(ker_key, ker_class, float, f, marco_declare, \
                                  macro_key, macro_impl);                      \
  JITKERNEL_WITH_DTYPE_DEPRECATED(ker_key, ker_class, double, d,               \
                                  marco_declare, macro_key, macro_impl)

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
