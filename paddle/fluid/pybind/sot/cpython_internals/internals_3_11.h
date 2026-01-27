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

#include "paddle/fluid/pybind/sot/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

#if (PY_3_11_PLUS && !PY_3_12_PLUS)

#include <Python.h>

#include <internal/pycore_frame.h>

int Internal_PyInterpreterFrame_GetLine(_PyInterpreterFrame *frame);

void Internal_PyFrame_Clear(_PyInterpreterFrame *frame);

int Internal_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame);

#endif  // (PY_3_11_PLUS && !PY_3_12_PLUS)

#ifdef __cplusplus
}
#endif
