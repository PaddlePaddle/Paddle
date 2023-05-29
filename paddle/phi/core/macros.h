/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

namespace phi {

// Disable the copy and assignment operator for a class.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)         \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete
#endif

#define PD_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg) \
  _PD_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)

#define _PD_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                    \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#ifdef __COUNTER__
#define PD_ID __COUNTER__
#else
#define PD_ID __LINE__
#endif

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

#define PD_CONCATENATE(arg1, arg2) PD_CONCATENATE1(arg1, arg2)
#define PD_CONCATENATE1(arg1, arg2) PD_CONCATENATE2(arg1, arg2)
#define PD_CONCATENATE2(arg1, arg2) arg1##arg2
#define PD_EXPAND(x) x

#if defined(__NVCC__) || defined(__HIPCC__)
#define PADDLE_RESTRICT __restrict__
#else
#define PADDLE_RESTRICT
#endif

#ifndef PADDLE_WITH_MUSL
#if defined(__FLT_MAX__)
#define FLT_MAX __FLT_MAX__
#endif  // __FLT_MAX__
#endif  // PADDLE_WITH_MUSL

}  // namespace phi
