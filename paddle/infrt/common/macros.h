// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#define INFRT_DEBUG
#endif

#define INFRT_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;            \
  void operator=(const TypeName&) = delete

#ifndef INFRT_NOT_IMPLEMENTED
#define INFRT_NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented";
#endif

#define INFRT_RESULT_SHOULD_USE __attribute__((warn_unused_result))

/**
 * A trick to enforce the registry.
 *
 * usage:
 *
 * INFRT_REGISTER_HELPER(some_key) {
 *   // register methods
 * }
 *
 * INFRT_USE_REGISTER(some_key);
 */
#define INFRT_REGISTER_HELPER(symbol__) bool __infrt__##symbol__##__registrar()
#define INFRT_USE_REGISTER(symbol__)                                 \
  extern bool __infrt__##symbol__##__registrar();                    \
  [[maybe_unused]] static bool __infrt_extern_registrar_##symbol__ = \
      __infrt__##symbol__##__registrar();

#if __cplusplus >= 201703L
#define INFRT_NODISCARD [[nodiscard]]
#else
#define INFRT_NODISCARD
#endif
