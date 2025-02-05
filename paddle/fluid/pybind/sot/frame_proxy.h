/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/pybind/sot/cpython_internals.h"
#include "paddle/fluid/pybind/sot/macros.h"

#if SOT_IS_SUPPORTED

#if !PY_3_11_PLUS
#include <frameobject.h>
#endif

#if PY_3_11_PLUS

#if PY_3_13_PLUS
#define Py_BUILD_CORE
#endif
#include <internal/pycore_frame.h>

typedef _PyInterpreterFrame FrameObject;

// clang-format off
// Define a proxy PyObject to access _PyInterpreterFrame's properties.
// It will be passed as an argument to the eval frame's callback.
typedef struct PyInterpreterFrameProxy {
  PyObject_HEAD
  _PyInterpreterFrame *frame;
  #if PY_3_13_PLUS
  PyObject* locals;
  #endif
} PyInterpreterFrameProxy;
// clang-format on

PyInterpreterFrameProxy *PyInterpreterFrameProxy_New(
    _PyInterpreterFrame *frame);
PyMODINIT_FUNC PyInit__frame_proxy();

#else
typedef PyFrameObject FrameObject;
#endif

#endif

#ifdef __cplusplus
}
#endif
