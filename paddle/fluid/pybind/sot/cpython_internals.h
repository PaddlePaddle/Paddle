/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>

#include "paddle/fluid/pybind/sot/macros.h"

#if SOT_IS_SUPPORTED

#if PY_VERSION_HEX >= 0x030b0000
#include <internal/pycore_frame.h>
#endif

#if PY_VERSION_HEX >= 0x030b0000
int Internal_PyInterpreterFrame_GetLine(_PyInterpreterFrame *frame);
static int Internal_PyFrame_OpAlreadyRan(_PyInterpreterFrame *frame,
                                         int opcode,
                                         int oparg);
int Internal_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame);
PyFrameObject *Internal_PyFrame_New_NoTrack(PyCodeObject *code);
PyFrameObject *Internal_PyFrame_MakeAndSetFrameObject(
    _PyInterpreterFrame *frame);
static inline PyFrameObject *Internal_PyFrame_GetFrameObject(
    _PyInterpreterFrame *frame);
static void Internal_take_ownership(PyFrameObject *f,
                                    _PyInterpreterFrame *frame);
void Internal_PyFrame_Clear(_PyInterpreterFrame *frame);

#if PY_VERSION_HEX >= 0x030c0000
void Internal_PyEvalFrameClearAndPop(PyThreadState *tstate,
                                     _PyInterpreterFrame *frame);
_PyInterpreterFrame *Internal_PyThreadState_PushFrame(PyThreadState *tstate,
                                                      size_t size);
void Internal_PyFrame_ClearExceptCode(_PyInterpreterFrame *frame);
#endif

#endif

#endif

#ifdef __cplusplus
}
#endif
