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

#include <Python.h>

// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

namespace paddle {
namespace pybind {

PyObject *static_api_add_n(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *static_api_mean(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *static_api_sum(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *static_api_divide(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *static_api_concat(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *static_api_full(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *static_api_data(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *static_api_fetch(PyObject *self, PyObject *args, PyObject *kwargs);

}  // namespace pybind
}  // namespace paddle
