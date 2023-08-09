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

#include <pybind11/pybind11.h>

#include "paddle/fluid/pybind/static_op_function.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace pybind {

static PyObject *mean(PyObject *self, PyObject *args, PyObject *kwargs) {
  return static_api_mean(self, args, kwargs);
}

static PyObject *sum(PyObject *self, PyObject *args, PyObject *kwargs) {
  return static_api_sum(self, args, kwargs);
}

static PyObject *full(PyObject *self, PyObject *args, PyObject *kwargs) {
  return static_api_full(self, args, kwargs);
}

static PyObject *divide(PyObject *self, PyObject *args, PyObject *kwargs) {
  return static_api_divide(self, args, kwargs);
}

static PyMethodDef OpsAPI[] = {{"mean",
                                (PyCFunction)(void (*)(void))mean,
                                METH_VARARGS | METH_KEYWORDS,
                                "C++ interface function for mean."},
                               {"sum",
                                (PyCFunction)(void (*)(void))sum,
                                METH_VARARGS | METH_KEYWORDS,
                                "C++ interface function for sum."},
                               {"divide",
                                (PyCFunction)(void (*)(void))divide,
                                METH_VARARGS | METH_KEYWORDS,
                                "C++ interface function for divide."},
                               {"full",
                                (PyCFunction)(void (*)(void))full,
                                METH_VARARGS | METH_KEYWORDS,
                                "C++ interface function for full."},
                               {nullptr, nullptr, 0, nullptr}};

void BindOpsAPI(pybind11::module *module) {
  if (PyModule_AddFunctions(module->ptr(), OpsAPI) < 0) {
    PADDLE_THROW(phi::errors::Fatal("Add C++ api to core.ops failed!"));
  }
}

}  // namespace pybind
}  // namespace paddle
