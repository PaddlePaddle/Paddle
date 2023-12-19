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
static PyObject *static_api_parameter(PyObject *self,
                                      PyObject *args,
                                      PyObject *kwargs) {
  try {
    VLOG(6) << "Add parameter op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Parse Attributes
    PyObject *name_obj = PyTuple_GET_ITEM(args, 0);
    std::string name = CastPyArg2String(name_obj, "name", 0);
    // Call ir static api
    auto static_api_out = paddle::dialect::parameter(name);

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

static PyObject *static_api_create_array(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add create_array op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get dtype from args
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 0);
    phi::DataType dtype =
        CastPyArg2DataTypeDirectly(dtype_obj, "create_array", 0);

    // Call ir static api
    auto static_api_out = paddle::dialect::create_array(dtype);

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_array_length(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add array_length op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2Value(x_obj, "array_length", 0);

    // Call ir static api
    auto static_api_out = paddle::dialect::array_length(x);

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_array_read(PyObject *self,
                                       PyObject *args,
                                       PyObject *kwargs) {
  try {
    VLOG(6) << "Add array_read op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *array_obj = PyTuple_GET_ITEM(args, 0);
    auto array = CastPyArg2Value(array_obj, "array_read", 0);

    PyObject *i_obj = PyTuple_GET_ITEM(args, 1);
    pir::Value i;
    if (PyObject_CheckIROpResult(i_obj)) {
      i = CastPyArg2Value(i_obj, "array_read", 1);
    } else {
      int64_t i_tmp = CastPyArg2Int(i_obj, "array_read", 1);
      i = paddle::dialect::full(std::vector<int64_t>{1},
                                i_tmp,
                                phi::DataType::INT64,
                                phi::CPUPlace());
    }

    // Call ir static api
    auto static_api_out = paddle::dialect::array_read(array, i);

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_array_write_(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add array_write_ op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *array_obj = PyTuple_GET_ITEM(args, 0);
    auto array = CastPyArg2Value(array_obj, "array_write_", 0);
    PyObject *x_obj = PyTuple_GET_ITEM(args, 1);
    auto x = CastPyArg2Value(x_obj, "array_write_", 1);
    PyObject *i_obj = PyTuple_GET_ITEM(args, 2);
    pir::Value i;
    if (PyObject_CheckIROpResult(i_obj)) {
      i = CastPyArg2Value(i_obj, "array_write_", 2);
    } else {
      int64_t i_tmp = CastPyArg2Int(i_obj, "array_write_", 2);
      i = paddle::dialect::full(std::vector<int64_t>{1},
                                i_tmp,
                                phi::DataType::INT64,
                                phi::CPUPlace());
    }

    // Call ir static api
    auto static_api_out = paddle::dialect::array_write_(array, x, i);

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_array_to_tensor(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  try {
    VLOG(6) << "Add array_to_tensor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    pir::Value x;
    if (PyObject_CheckIROpResult(x_obj)) {
      x = CastPyArg2Value(x_obj, "array_to_tensor", 0);
    } else if (PyObject_CheckIRVectorOfOpResult(x_obj)) {
      std::vector<pir::Value> x_tmp =
          CastPyArg2VectorOfValue(x_obj, "array_to_tensor", 0);
      if (x_tmp.size() != 1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Input x expects only one input, but %d are given.",
            x_tmp.size()));  // NOLINT
      }
      x = x_tmp[0];
    }

    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    auto axis = CastPyArg2Int(axis_obj, "array_to_tensor", 1);

    PyObject *use_stack_obj = PyTuple_GET_ITEM(args, 2);
    auto use_stack = CastPyArg2Boolean(use_stack_obj, "array_to_tensor", 2);

    // Call ir static api
    auto static_api_out = paddle::dialect::array_to_tensor(x, axis, use_stack);

    return ToPyObject(static_api_out);
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
    {"parameter",
     (PyCFunction)(void (*)(void))static_api_parameter,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for parameter."},
    {"create_array",
     (PyCFunction)(void (*)(void))static_api_create_array,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for create_array."},
    {"array_length",
     (PyCFunction)(void (*)(void))static_api_array_length,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for array_length."},
    {"array_read",
     (PyCFunction)(void (*)(void))static_api_array_read,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for array_read."},
    {"array_write_",
     (PyCFunction)(void (*)(void))static_api_array_write_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for array_write_."},
    {"array_to_tensor",
     (PyCFunction)(void (*)(void))static_api_array_to_tensor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for array_to_tensor."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace pybind

}  // namespace paddle
