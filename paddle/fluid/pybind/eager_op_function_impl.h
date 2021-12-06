// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "pybind11/detail/common.h"

namespace paddle {
namespace pybind {

static PyObject *eager_api_matmul_v2(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("matmul_v2", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("matmul_v2", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("matmul_v2", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = matmul_v2_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elementwise_add(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("elementwise_add", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("elementwise_add", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elementwise_add", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elementwise_add_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sigmoid(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sigmoid", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sigmoid", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = sigmoid_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reduce_sum(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reduce_sum", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reduce_sum", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reduce_sum_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyMethodDef ExtestMethods[] = {
    {"matmul_v2", (PyCFunction)(void (*)(void))eager_api_matmul_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matmul_v2 in dygraph."},
    {"elementwise_add", (PyCFunction)(void (*)(void))eager_api_elementwise_add,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_add in dygraph."},
    {"sigmoid", (PyCFunction)(void (*)(void))eager_api_sigmoid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sigmoid in dygraph."},
    {"reduce_sum", (PyCFunction)(void (*)(void))eager_api_reduce_sum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reduce_sum in dygraph."},
    {nullptr, nullptr, 0, nullptr}};

inline void BindEagerOpFunctions(pybind11::module *module) {
  auto m = module->def_submodule("ops");
  if (PyModule_AddFunctions(m.ptr(), ExtestMethods) < 0) {
    PADDLE_THROW(
        platform::errors::Fatal("Add functions to core.eager.ops failed!"));
  }

  InitOpsAttrTypeMap();
}

}  // namespace pybind
}  // namespace paddle
