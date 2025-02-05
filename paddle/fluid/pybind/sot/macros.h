// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <Python.h>

#define PY_3_8_0_HEX 0x03080000
#define PY_3_9_0_HEX 0x03090000
#define PY_3_10_0_HEX 0x030A0000
#define PY_3_11_0_HEX 0x030B0000
#define PY_3_12_0_HEX 0x030C0000
#define PY_3_13_0_HEX 0x030D0000
#define PY_3_14_0_HEX 0x030E0000

#define PY_3_8_PLUS (PY_VERSION_HEX >= PY_3_8_0_HEX)
#define PY_3_9_PLUS (PY_VERSION_HEX >= PY_3_9_0_HEX)
#define PY_3_10_PLUS (PY_VERSION_HEX >= PY_3_10_0_HEX)
#define PY_3_11_PLUS (PY_VERSION_HEX >= PY_3_11_0_HEX)
#define PY_3_12_PLUS (PY_VERSION_HEX >= PY_3_12_0_HEX)
#define PY_3_13_PLUS (PY_VERSION_HEX >= PY_3_13_0_HEX)
#define PY_3_14_PLUS (PY_VERSION_HEX >= PY_3_14_0_HEX)

#define SOT_NOT_SUPPORTED_VERSION PY_3_14_0_HEX
#define SOT_IS_SUPPORTED (PY_VERSION_HEX < SOT_NOT_SUPPORTED_VERSION)

#if PY_3_13_PLUS
#define PyFrame_GET_CODE(frame) _PyFrame_GetCode(frame)
#else
#define PyFrame_GET_CODE(frame) (frame->f_code)
#endif

#ifdef __cplusplus
}
#endif
