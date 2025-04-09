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
#include "paddle/fluid/pybind/sot/frame_proxy.h"
#include "paddle/fluid/pybind/sot/macros.h"

#if SOT_IS_SUPPORTED

int need_skip(FrameObject* frame);
int is_code_without_graph(PyCodeObject* code);

PyObject* set_with_graph(PyObject* code);
PyObject* setup_codes_with_graph(PyObject* code_tuple);
PyObject* no_skip_codes(PyObject* code_tuple);
PyObject* skip_file_prefix(PyObject* filepath_tuple);

#endif

#ifdef __cplusplus
}
#endif
