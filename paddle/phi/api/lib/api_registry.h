/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/common/macros.h"

namespace paddle {
namespace experimental {

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

/**
 * Now there is no module to call phi's API. When compiling, the function
 * implementation will be optimized. Therefore, the symbol will be exposed
 * manually for the time being.
 *
 * After the dynamic graph calls the API in the future, the logic declared
 * by these macro can be deleted.
 */

// use to declare symbol
#define PD_REGISTER_API(name) \
  PADDLE_API int RegisterSymbolsFor##name() { return 0; }

#define PD_DECLARE_API(name)                        \
  extern PADDLE_API int RegisterSymbolsFor##name(); \
  UNUSED static int use_phi_api_##name = RegisterSymbolsFor##name()

}  // namespace experimental
}  // namespace paddle
