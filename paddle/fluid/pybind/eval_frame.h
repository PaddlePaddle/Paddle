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

#include "paddle/fluid/pybind/cpython_internals.h"

#if EVAL_FRAME_IS_SUPPORTED

PyObject *set_eval_frame_py(PyObject *callback);
PyMODINIT_FUNC PyInit__eval_frame();

#endif

#ifdef __cplusplus
}
#endif
