// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef __cplusplus
extern "C" {
#endif

#include "paddle/fluid/pybind/sot/macros.h"

#if SOT_IS_SUPPORTED

#if PY_3_14_PLUS
#include "paddle/fluid/pybind/sot/cpython_internals/internals_3_14.h"
#elif PY_3_13_PLUS
#include "paddle/fluid/pybind/sot/cpython_internals/internals_3_13.h"
#elif PY_3_12_PLUS
#include "paddle/fluid/pybind/sot/cpython_internals/internals_3_12.h"
#elif PY_3_11_PLUS
#include "paddle/fluid/pybind/sot/cpython_internals/internals_3_11.h"
#endif

#endif  // SOT_IS_SUPPORTED

#ifdef __cplusplus
}
#endif
