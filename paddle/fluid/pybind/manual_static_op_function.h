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

#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {

namespace pybind {
static PyObject *static_api_get_parameter(PyObject *self,
                                          PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add get_parameter op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Parse Attributes
    PyObject *name_obj = PyTuple_GET_ITEM(args, 0);
    std::string name = CastPyArg2String(name_obj, "name", 0);
    // Call ir static api
    auto static_api_out = paddle::dialect::get_parameter(name);

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_set_parameter(PyObject *self,
                                          PyObject *args,
                                          PyObject *kwargs) {
  try {
    VLOG(6) << "Add set_parameter op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get OpResult from args
    PyObject *parameter_obj = PyTuple_GET_ITEM(args, 0);
    auto parameter = CastPyArg2Value(parameter_obj, "parameter", 0);

    // Parse Attributes
    PyObject *name_obj = PyTuple_GET_ITEM(args, 1);
    std::string name = CastPyArg2String(name_obj, "name", 1);
    // Call ir static api
    paddle::dialect::set_parameter(parameter, name);

    Py_RETURN_NONE;
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

PyObject *static_api_full(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add full op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Parse Attributes
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 3);

    phi::DataType dtype = CastPyArg2DataTypeDirectly(dtype_obj, "full", 2);
    Place place = CastPyArg2Place(place_obj, "full", 3);

    if (!PyObject_CheckIROpResult(shape_obj) &&
        !PyObject_CheckIRVectorOfOpResult(shape_obj) &&
        !PyObject_CheckIROpResult(value_obj)) {
      std::vector<int64_t> shape = CastPyArg2Longs(shape_obj, "full", 0);
      float value = CastPyArg2Float(value_obj, "full", 1);
      auto static_api_out = paddle::dialect::full(shape, value, dtype, place);
      return ToPyObject(static_api_out);
    } else {
      pir::Value shape, value;

      if (PyObject_CheckIROpResult(shape_obj)) {
        shape = CastPyArg2Value(shape_obj, "full", 0);
      } else if (PyObject_CheckIRVectorOfOpResult(shape_obj)) {
        std::vector<pir::Value> shape_tmp =
            CastPyArg2VectorOfValue(shape_obj, "full", 0);
        shape = paddle::dialect::stack(shape_tmp, 0);
      } else {
        std::vector<int64_t> shape_tmp = CastPyArg2Longs(shape_obj, "full", 0);
        shape = paddle::dialect::full_int_array(
            shape_tmp, phi::DataType::INT64, phi::CPUPlace());
      }

      if (PyObject_CheckIROpResult(value_obj)) {
        value = CastPyArg2Value(value_obj, "full", 1);
      } else {
        float value_tmp = CastPyArg2Float(value_obj, "full", 1);
        value = paddle::dialect::full(std::vector<int64_t>{1},
                                      value_tmp,
                                      phi::DataType::FLOAT32,
                                      phi::CPUPlace());
      }

      auto static_api_out =
          paddle::dialect::full_with_tensor(shape, value, dtype);
      return ToPyObject(static_api_out);
    }
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyMethodDef ManualOpsAPI[] = {
    {"set_parameter",
     (PyCFunction)(void (*)(void))static_api_set_parameter,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for set_parameter."},
    {"get_parameter",
     (PyCFunction)(void (*)(void))static_api_get_parameter,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for get_parameter."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace pybind

}  // namespace paddle
