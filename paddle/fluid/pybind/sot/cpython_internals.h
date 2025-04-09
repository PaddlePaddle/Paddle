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

#if PY_3_11_PLUS
#if PY_3_13_PLUS
#define Py_BUILD_CORE
#endif
#include <internal/pycore_frame.h>
#endif

#if PY_3_11_PLUS
#if PY_3_13_PLUS
int Internal_PyUnstable_InterpreterFrame_GetLine(_PyInterpreterFrame *frame);
#else
int Internal_PyInterpreterFrame_GetLine(_PyInterpreterFrame *frame);
#endif
static int Internal_PyFrame_OpAlreadyRan(_PyInterpreterFrame *frame,
                                         int opcode,
                                         int oparg);
#if PY_3_13_PLUS
PyObject *get_framelocals_mapping(_PyInterpreterFrame *frame);
#else
int Internal_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame);
#endif
PyFrameObject *Internal_PyFrame_New_NoTrack(PyCodeObject *code);
PyFrameObject *Internal_PyFrame_MakeAndSetFrameObject(
    _PyInterpreterFrame *frame);
static inline PyFrameObject *Internal_PyFrame_GetFrameObject(
    _PyInterpreterFrame *frame);
static void Internal_take_ownership(PyFrameObject *f,
                                    _PyInterpreterFrame *frame);
void Internal_PyFrame_Clear(_PyInterpreterFrame *frame);

#if PY_3_12_PLUS
#if PY_3_13_PLUS
void Internal_PyEval_FrameClearAndPop(PyThreadState *tstate,
                                      _PyInterpreterFrame *frame);
#else
void Internal_PyEvalFrameClearAndPop(PyThreadState *tstate,
                                     _PyInterpreterFrame *frame);
#endif
_PyInterpreterFrame *Internal_PyThreadState_PushFrame(PyThreadState *tstate,
                                                      size_t size);
void Internal_PyFrame_ClearExceptCode(_PyInterpreterFrame *frame);
#endif

#endif

#endif

#ifdef __cplusplus
}
#endif
