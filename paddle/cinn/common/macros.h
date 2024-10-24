// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#pragma once

#if !defined(NDEBUG)
#define CINN_DEBUG
#endif

#define CINN_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;           \
  void operator=(const TypeName&) = delete

#ifndef CINN_NOT_IMPLEMENTED
#define CINN_NOT_IMPLEMENTED \
  PADDLE_THROW(::common::errors::Unimplemented("Not Implemented"));
#endif

#define CINN_RESULT_SHOULD_USE __attribute__((warn_unused_result))

/**
 * A trick to enforce the registry.
 *
 * usage:
 *
 * CINN_REGISTER_HELPER(some_key) {
 *   // register methods
 * }
 *
 * CINN_USE_REGISTER(some_key);
 */
#define CINN_REGISTER_HELPER(symbol__) bool __cinn__##symbol__##__registrar()
#define CINN_USE_REGISTER(symbol__)                                 \
  extern bool __cinn__##symbol__##__registrar();                    \
  [[maybe_unused]] static bool __cinn_extern_registrar_##symbol__ = \
      __cinn__##symbol__##__registrar();

#if __cplusplus >= 201703L
#define CINN_NODISCARD [[nodiscard]]
#else
#define CINN_NODISCARD
#endif

#define DISABLE_COPY_AND_ASSIGN(classname)         \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete

/**
 * check if MACRO is used in GLOBAL NAMESPACE.
 */
#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define USE_FUSION_PASS(pass_name)                                    \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                     \
      __use_cinn_fusion_pass_##pass_name,                             \
      "USE_FUSION_PASS must be called in global namespace");          \
  extern int TouchCinnFusionPassRegistrar_##pass_name();              \
  [[maybe_unused]] static int __use_cinn_fusion_pass_##pass_name##_ = \
      TouchCinnFusionPassRegistrar_##pass_name()
