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

#include "paddle/fluid/pybind/op_function_common.h"

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/operators/ops_extra_info.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/imperative.h"
#include "paddle/phi/common/complex.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/op_result.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace pybind {

class OpAttrTypeMap {
 public:
  static OpAttrTypeMap& Instance() {
    static OpAttrTypeMap g_op_attr_type_map;
    return g_op_attr_type_map;
  }

  std::unordered_map<
      std::string,
      std::unordered_map<std::string, paddle::framework::proto::AttrType>>&
  Map() {
    return ops_attrtype_map_;
  }

 private:
  OpAttrTypeMap() = default;
  std::unordered_map<
      std::string,
      std::unordered_map<std::string, paddle::framework::proto::AttrType>>
      ops_attrtype_map_;
};

extern PyTypeObject* g_vartype_pytype;
extern PyTypeObject* g_data_type_pytype;
extern PyTypeObject* g_blockdesc_pytype;
extern PyTypeObject* p_tensor_type;

bool PyObject_CheckBool(PyObject** obj) { return PyBool_Check(*obj); }

bool PyObject_CheckLongOrToLong(PyObject** obj) {
  if ((PyLong_Check(*obj) && !PyBool_Check(*obj)) ||
      PyObject_TypeCheck(*obj, g_vartype_pytype) ||        // NOLINT
      PyObject_TypeCheck(*obj, g_data_type_pytype) ||      // NOLINT
      (PyObject_TypeCheck(*obj, p_tensor_type) &&          // NOLINT
       (((TensorObject*)(*obj))->tensor.numel() == 1))) {  // NOLINT
    return true;
  }

  if (std::string(((PyTypeObject*)(*obj)->ob_type)->tp_name)  // NOLINT
          .find("numpy.int") != std::string::npos) {
    auto to = PyNumber_Long(*obj);
    if (to) {
      *obj = to;
      return true;
    }
  }

  return false;
}

bool PyObject_CheckFloatOrToFloat(PyObject** obj) {
  // sometimes users provide PyLong or numpy.int64 but attr is float
  if (PyFloat_Check(*obj) || PyLong_Check(*obj) ||
      (PyObject_TypeCheck(*obj, p_tensor_type) &&          // NOLINT
       (((TensorObject*)(*obj))->tensor.numel() == 1))) {  // NOLINT
    return true;
  }
  auto type_name =
      std::string(reinterpret_cast<PyTypeObject*>((*obj)->ob_type)->tp_name);
  VLOG(4) << "type_name: " << type_name;

  if (type_name.find("numpy") != std::string::npos &&
      type_name.find("numpy.complex") == std::string::npos) {
    auto to = PyNumber_Float(*obj);
    if (to) {
      *obj = to;
      return true;
    }
  }
  return false;
}

bool PyObject_CheckComplexOrToComplex(PyObject** obj) {
  if (PyComplex_Check(*obj) ||
      PyObject_TypeCheck(*obj, g_vartype_pytype) ||  // NOLINT
      PyObject_TypeCheck(*obj, p_tensor_type)) {     // NOLINT
    return true;
  }
  if (std::string(((PyTypeObject*)(*obj)->ob_type)->tp_name)  // NOLINT
          .find("numpy.complex") != std::string::npos) {
    return true;
  }
  // consider numpy cfloat & numpy cdouble?
  return false;
}

bool PyObject_CheckString(PyObject* obj) { return PyUnicode_Check(obj); }

bool CastPyArg2Boolean(PyObject* obj,
                       const std::string& op_type,
                       ssize_t arg_pos) {
  if (obj == Py_None || obj == Py_False) {
    return false;  // To be compatible with QA integration testing. Some
                   // test case pass in None.
  } else if (obj == Py_True) {
    return true;
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "bool, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return false;
}

void CastPyArg2AttrBoolean(PyObject* obj,
                           paddle::framework::AttributeMap& attrs,  // NOLINT
                           const std::string& key,
                           const std::string& op_type,
                           ssize_t arg_pos) {
  attrs[key] = CastPyArg2Boolean(obj, op_type, arg_pos);
}

int CastPyArg2Int(PyObject* obj, const std::string& op_type, ssize_t arg_pos) {
  if (PyObject_CheckLongOrToLong(&obj)) {
    return (int)PyLong_AsLong(obj);  // NOLINT
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "int, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return 0;
}

void CastPyArg2AttrInt(PyObject* obj,
                       paddle::framework::AttributeMap& attrs,  // NOLINT
                       const std::string& key,
                       const std::string& op_type,
                       ssize_t arg_pos) {
  attrs[key] = CastPyArg2Int(obj, op_type, arg_pos);
}

int64_t CastPyArg2Long(PyObject* obj,
                       const std::string& op_type,
                       ssize_t arg_pos) {
  if (PyObject_CheckLongOrToLong(&obj)) {
    return (int64_t)PyLong_AsLongLong(obj);  // NOLINT
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "long, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return 0;
}

void CastPyArg2AttrLong(PyObject* obj,
                        paddle::framework::AttributeMap& attrs,  // NOLINT
                        const std::string& key,
                        const std::string& op_type,
                        ssize_t arg_pos) {
  attrs[key] = CastPyArg2Long(obj, op_type, arg_pos);
}

void CastPyArg2AttrScalar(PyObject* obj,
                          paddle::framework::AttributeMap& attrs,  // NOLINT
                          const std::string& key,
                          const std::string& op_type,
                          ssize_t arg_pos) {
  attrs[key] = CastPyArg2Scalar(obj, op_type, arg_pos);
}

float16 CastPyArg2Float16(PyObject* obj,
                          const std::string& op_type,
                          ssize_t arg_pos) {
  return static_cast<float16>(CastPyArg2Double(obj, op_type, arg_pos));
}

float CastPyArg2Float(PyObject* obj,
                      const std::string& op_type,
                      ssize_t arg_pos) {
  return static_cast<float>(CastPyArg2Double(obj, op_type, arg_pos));
}

void CastPyArg2AttrFloat(PyObject* obj,
                         paddle::framework::AttributeMap& attrs,  // NOLINT
                         const std::string& key,
                         const std::string& op_type,
                         ssize_t arg_pos) {
  attrs[key] = CastPyArg2Float(obj, op_type, arg_pos);
}

double CastPyArg2Double(PyObject* obj,
                        const std::string& op_type,
                        ssize_t arg_pos) {
  if (PyObject_CheckFloatOrToFloat(&obj)) {
    return PyFloat_AsDouble(obj);  // NOLINT
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "double, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return 0.0;
}

phi::dtype::complex<float> CastPyArg2Complex(PyObject* obj,
                                             const std::string& op_type,
                                             ssize_t arg_pos) {
  PyTypeObject* type = obj->ob_type;
  auto type_name = std::string(type->tp_name);
  if (PyComplex_Check(obj)) {
    double real = PyComplex_RealAsDouble(obj);
    double imag = PyComplex_ImagAsDouble(obj);
    return phi::dtype::complex<float>(real, imag);  // NOLINT
  } else if (type_name == "numpy.complex64") {
    Py_complex v = PyComplex_AsCComplex(obj);
    return phi::dtype::complex<float>(v.real, v.imag);
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "complex, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return phi::dtype::complex<float>(0, 0);
}

phi::dtype::complex<double> CastPyArg2Complex128(PyObject* obj,
                                                 const std::string& op_type,
                                                 ssize_t arg_pos) {
  if (PyComplex_Check(obj)) {
    double real = PyComplex_RealAsDouble(obj);
    double imag = PyComplex_ImagAsDouble(obj);
    return phi::dtype::complex<double>(real, imag);
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "complex, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return phi::dtype::complex<double>(0, 0);
}

void CastPyArg2AttrDouble(PyObject* obj,
                          paddle::framework::AttributeMap& attrs,  // NOLINT
                          const std::string& key,
                          const std::string& op_type,
                          ssize_t arg_pos) {
  attrs[key] = CastPyArg2Double(obj, op_type, arg_pos);
}

std::string CastPyArg2String(PyObject* obj,
                             const std::string& op_type,
                             ssize_t arg_pos) {
  if (PyObject_CheckString(obj)) {
    Py_ssize_t size = 0;
    const char* data = nullptr;
    data = PyUnicode_AsUTF8AndSize(obj, &size);
    return std::string(data, (size_t)size);  // NOLINT
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "str, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return "";
}

void CastPyArg2AttrString(PyObject* obj,
                          paddle::framework::AttributeMap& attrs,  // NOLINT
                          const std::string& key,
                          const std::string& op_type,
                          ssize_t arg_pos) {
  attrs[key] = CastPyArg2String(obj, op_type, arg_pos);
}

std::vector<bool> CastPyArg2Booleans(PyObject* obj,
                                     const std::string& op_type,
                                     ssize_t arg_pos) {
  std::vector<bool> value;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckBool(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of bool, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckBool(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of bool, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return value;
}

void CastPyArg2AttrBooleans(PyObject* obj,
                            paddle::framework::AttributeMap& attrs,  // NOLINT
                            const std::string& key,
                            const std::string& op_type,
                            ssize_t arg_pos) {
  attrs[key] = CastPyArg2Booleans(obj, op_type, arg_pos);
}

std::vector<int> CastPyArg2Ints(PyObject* obj,
                                const std::string& op_type,
                                ssize_t arg_pos) {
  std::vector<int> value;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    value.reserve(len);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    value.reserve(len);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (PySequence_Check(obj) && !PyObject_TypeCheck(obj, p_tensor_type)) {
    Py_ssize_t len = PySequence_Size(obj);
    value.reserve(len);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PySequence_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back(PyLong_AsLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return value;
}

void CastPyArg2AttrInts(PyObject* obj,
                        paddle::framework::AttributeMap& attrs,  // NOLINT
                        const std::string& key,
                        const std::string& op_type,
                        ssize_t arg_pos) {
  attrs[key] = CastPyArg2Ints(obj, op_type, arg_pos);
}

std::vector<int64_t> CastPyArg2Longs(PyObject* obj,
                                     const std::string& op_type,
                                     ssize_t arg_pos) {
  std::vector<int64_t> value;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back((int64_t)PyLong_AsLongLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back((int64_t)PyLong_AsLongLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (PySequence_Check(obj) && !PyObject_TypeCheck(obj, p_tensor_type)) {
    Py_ssize_t len = PySequence_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PySequence_GetItem(obj, i);
      if (PyObject_CheckLongOrToLong(&item)) {
        value.emplace_back((int64_t)PyLong_AsLongLong(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of int, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else if (PyObject_CheckLongOrToLong(&obj)) {
    return {(int64_t)PyLong_AsLongLong(obj)};  // NOLINT
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return value;
}

void CastPyArg2AttrLongs(PyObject* obj,
                         paddle::framework::AttributeMap& attrs,  // NOLINT
                         const std::string& key,
                         const std::string& op_type,
                         ssize_t arg_pos) {
  attrs[key] = CastPyArg2Longs(obj, op_type, arg_pos);
}

std::vector<float> CastPyArg2Floats(PyObject* obj,
                                    const std::string& op_type,
                                    ssize_t arg_pos) {
  std::vector<float> value;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (PySequence_Check(obj) && !PyObject_TypeCheck(obj, p_tensor_type)) {
    Py_ssize_t len = PySequence_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PySequence_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return value;
}

void CastPyArg2AttrFloats(PyObject* obj,
                          paddle::framework::AttributeMap& attrs,  // NOLINT
                          const std::string& key,
                          const std::string& op_type,
                          ssize_t arg_pos) {
  attrs[key] = CastPyArg2Floats(obj, op_type, arg_pos);
}

std::vector<double> CastPyArg2Float64s(PyObject* obj,
                                       const std::string& op_type,
                                       ssize_t arg_pos) {
  std::vector<double> value;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (PySequence_Check(obj) && !PyObject_TypeCheck(obj, p_tensor_type)) {
    Py_ssize_t len = PySequence_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PySequence_GetItem(obj, i);
      if (PyObject_CheckFloatOrToFloat(&item)) {
        value.emplace_back(PyFloat_AsDouble(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of float, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return value;
}

void CastPyArg2AttrFloat64s(PyObject* obj,
                            paddle::framework::AttributeMap& attrs,  // NOLINT
                            const std::string& key,
                            const std::string& op_type,
                            ssize_t arg_pos) {
  attrs[key] = CastPyArg2Float64s(obj, op_type, arg_pos);
}

void CastPyArg2AttrScalars(PyObject* obj,
                           paddle::framework::AttributeMap& attrs,  // NOLINT
                           const std::string& key,
                           const std::string& op_type,
                           ssize_t arg_pos) {
  attrs[key] = CastPyArg2Scalars(obj, op_type, arg_pos);
}

std::vector<std::string> CastPyArg2Strings(PyObject* obj,
                                           const std::string& op_type,
                                           ssize_t arg_pos) {
  std::vector<std::string> value;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckString(item)) {
        Py_ssize_t size = 0;
        const char* data = nullptr;
        data = PyUnicode_AsUTF8AndSize(item, &size);
        value.emplace_back(std::string(data, (size_t)size));  // NOLINT
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of str, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckString(item)) {
        Py_ssize_t size = 0;
        const char* data = nullptr;
        data = PyUnicode_AsUTF8AndSize(item, &size);
        value.emplace_back(std::string(data, (size_t)size));  // NOLINT
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "list of str, but got %s at pos %d",
            op_type,
            arg_pos + 1,
            ((PyTypeObject*)item->ob_type)->tp_name,  // NOLINT
            i));
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "list or tuple, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  return value;
}

void CastPyArg2AttrStrings(PyObject* obj,
                           paddle::framework::AttributeMap& attrs,  // NOLINT
                           const std::string& key,
                           const std::string& op_type,
                           ssize_t arg_pos) {
  attrs[key] = CastPyArg2Strings(obj, op_type, arg_pos);
}

std::vector<paddle::experimental::Scalar> CastPyArg2Scalars(
    PyObject* obj, const std::string& op_type, ssize_t arg_pos) {
  if (obj == Py_None) {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "a list of int, float, or bool, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  PyTypeObject* type = obj->ob_type;
  auto type_name = std::string(type->tp_name);
  VLOG(4) << "type_name: " << type_name;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    item = PyList_GetItem(obj, 0);
    if (PyObject_CheckFloatOrToFloat(&item)) {
      std::vector<paddle::experimental::Scalar> value;
      for (Py_ssize_t i = 0; i < len; i++) {
        item = PyList_GetItem(obj, i);
        value.emplace_back(
            paddle::experimental::Scalar{PyFloat_AsDouble(item)});
      }
      return value;
    } else if (PyObject_CheckLongOrToLong(&item)) {
      std::vector<paddle::experimental::Scalar> value;
      for (Py_ssize_t i = 0; i < len; i++) {
        item = PyList_GetItem(obj, i);
        value.emplace_back(paddle::experimental::Scalar{
            static_cast<int64_t>(PyLong_AsLong(item))});
      }
      return value;
    } else if (PyObject_CheckComplexOrToComplex(&item)) {
      std::vector<paddle::experimental::Scalar> value;
      for (Py_ssize_t i = 0; i < len; i++) {
        item = PyList_GetItem(obj, i);
        Py_complex v = PyComplex_AsCComplex(item);
        value.emplace_back(
            paddle::experimental::Scalar{std::complex<double>(v.real, v.imag)});
      }
      return value;
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "a list of int, float, complex, or bool, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  // Fake a ScalarArray
  return std::vector<paddle::experimental::Scalar>(
      {paddle::experimental::Scalar(1.0)});
}

void CastPyArg2AttrBlock(PyObject* obj,
                         paddle::framework::AttributeMap& attrs,  // NOLINT
                         const std::string& key,
                         const std::string& op_type,
                         ssize_t arg_pos) {
  ::pybind11::detail::instance* inst =
      (::pybind11::detail::instance*)obj;  // NOLINT

  if (!PyObject_TypeCheck((PyObject*)inst, g_blockdesc_pytype)) {  // NOLINT
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "BlockDesc, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
  void** vh = inst->simple_layout ? inst->simple_value_holder
                                  : &inst->nonsimple.values_and_holders[0];
  attrs[key] = reinterpret_cast<paddle::framework::BlockDesc*&>(vh[0]);
}

void CastPyArg2AttrIRBlock(PyObject* obj,
                           paddle::framework::AttributeMap& attrs,  // NOLINT
                           const std::string& key,
                           const std::string& op_type,
                           ssize_t arg_pos) {
  VLOG(1) << "After Process pir::Block*";
  ::pybind11::detail::instance* inst =
      (::pybind11::detail::instance*)obj;  // NOLINT
  void** vh = inst->simple_layout ? inst->simple_value_holder
                                  : &inst->nonsimple.values_and_holders[0];
  attrs[key] = reinterpret_cast<::pir::Block*&>(vh[0]);
}

void CastPyArg2AttrIRProgram(PyObject* obj,
                             paddle::framework::AttributeMap& attrs,  // NOLINT
                             const std::string& key,
                             const std::string& op_type,
                             ssize_t arg_pos) {
  VLOG(1) << "After Process shared_ptr<pir::Program>";
  ::pybind11::object o = ::pybind11::reinterpret_steal<::pybind11::object>(obj);
  // ::pybind11::object o =
  //     ::pybind11::reinterpret_borrow<::pybind11::object>(obj);
  // ::pybind11::detail::instance* inst =
  //     (::pybind11::detail::instance*)obj;  // NOLINT
  // void** vh = inst->simple_layout ? inst->simple_value_holder
  //                                 :
  //                                 &inst->nonsimple.values_and_holders[0];
  // attrs[key] = reinterpret_cast<std::shared_ptr<::pir::Program>>(vh[0]);
  attrs[key] = o.cast<std::shared_ptr<::pir::Program>>();
}

void CastPyArg2AttrValues(PyObject* obj,
                          paddle::framework::AttributeMap& attrs,  // NOLINT
                          const std::string& key,
                          const std::string& op_type,
                          ssize_t arg_pos) {
  std::vector<::pir::Value> results;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      // TODO(xiongkun): judge Value;
      item = PyList_GetItem(obj, i);
      ::pybind11::detail::instance* inst =
          (::pybind11::detail::instance*)item;  // NOLINT
      void** vh = inst->simple_layout ? inst->simple_value_holder
                                      : &inst->nonsimple.values_and_holders[0];
      ::pir::Value* value = reinterpret_cast<::pir::Value*>(vh[0]);
      results.emplace_back(pir::Value(value->impl()));
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "a list of int, float, complex, or bool, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
  attrs[key] = results;
  VLOG(1) << "Pybind: Cast " << results.size() << " Value Finished.";
}

void ConstructAttrMapFromPyArgs(
    const std::string& op_type,
    PyObject* args,
    ssize_t attr_start,
    ssize_t attr_end,
    paddle::framework::AttributeMap& attrs) {  // NOLINT
  PADDLE_ENFORCE_EQ((attr_end - attr_start) % 2,
                    0,
                    platform::errors::InvalidArgument(
                        "The number of arguments for attributes should be even "
                        "but attr_start = %d, attr_end = %d.",
                        attr_start,
                        attr_end));

  auto attr_type_map = &(OpAttrTypeMap::Instance().Map()[op_type]);

  PyObject* obj = nullptr;
  for (ssize_t arg_pos = attr_start; arg_pos < attr_end; arg_pos += 2) {
    VLOG(1) << "Start Process " << arg_pos;
    Py_ssize_t key_len = 0;
    const char* key_ptr = nullptr;
    obj = PyTuple_GET_ITEM(args, arg_pos);
    if (PyObject_CheckString(obj)) {
      key_ptr = PyUnicode_AsUTF8AndSize(obj, &key_len);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument (position %d) must be str, but got "
          "%s",
          op_type,
          arg_pos,
          ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
    }

    std::string key(key_ptr, (size_t)key_len);  // NOLINT
    VLOG(1) << "Start Process " << key;
    auto iter = attr_type_map->find(key);
    if (iter == attr_type_map->end()) {
      continue;
    }

    obj = PyTuple_GET_ITEM(args, arg_pos + 1);

    switch (iter->second) {
      case paddle::framework::proto::AttrType::INT:
        CastPyArg2AttrInt(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::FLOAT:
        CastPyArg2AttrFloat(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::FLOAT64:
        CastPyArg2AttrDouble(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::STRING:
        CastPyArg2AttrString(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::INTS:
        CastPyArg2AttrInts(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::FLOATS:
        CastPyArg2AttrFloats(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::STRINGS:
        CastPyArg2AttrStrings(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::BOOLEAN:
        CastPyArg2AttrBoolean(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::BOOLEANS:
        CastPyArg2AttrBooleans(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::LONG:
        CastPyArg2AttrLong(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::LONGS:
        CastPyArg2AttrLongs(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::FLOAT64S:
        CastPyArg2AttrFloat64s(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::BLOCK:
        CastPyArg2AttrBlock(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::SCALAR:
        CastPyArg2AttrScalar(obj, attrs, key, op_type, arg_pos);
        break;
      case paddle::framework::proto::AttrType::SCALARS:
        CastPyArg2AttrScalars(obj, attrs, key, op_type, arg_pos);
        break;
      default:
        break;
    }
  }
}

void ConstructAttrMapForRunProgram(
    const std::string& op_type,
    PyObject* args,
    ssize_t attr_start,
    ssize_t attr_end,
    paddle::framework::AttributeMap& attrs) {  // NOLINT
  PADDLE_ENFORCE_EQ((attr_end - attr_start) % 2,
                    0,
                    platform::errors::InvalidArgument(
                        "The number of arguments for attributes should be even "
                        "but attr_start = %d, attr_end = %d.",
                        attr_start,
                        attr_end));

  PyObject* obj = nullptr;
  for (ssize_t arg_pos = attr_start; arg_pos < attr_end; arg_pos += 2) {
    VLOG(1) << "Start Process " << arg_pos;
    Py_ssize_t key_len = 0;
    const char* key_ptr = nullptr;
    obj = PyTuple_GET_ITEM(args, arg_pos);
    if (PyObject_CheckString(obj)) {
      key_ptr = PyUnicode_AsUTF8AndSize(obj, &key_len);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument (position %d) must be str, but got "
          "%s",
          op_type,
          arg_pos,
          ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
    }

    std::string key(key_ptr, (size_t)key_len);  // NOLINT
    VLOG(1) << "Start Process " << key;
    obj = PyTuple_GET_ITEM(args, arg_pos + 1);

    if (std::set<std::string>({"cuda_graph_capture_mode"}).count(key)) {
      CastPyArg2AttrString(obj, attrs, key, op_type, arg_pos);
    } else if (std::set<std::string>({"global_block"}).count(key)) {
      CastPyArg2AttrIRBlock(obj, attrs, key, op_type, arg_pos);
    } else if (std::set<std::string>({"forward_program", "backward_program"})
                   .count(key)) {
      CastPyArg2AttrIRProgram(obj, attrs, key, op_type, arg_pos);
    } else if (std::set<std::string>({"is_test", "use_interpretorcore"})
                   .count(key)) {
      CastPyArg2AttrBoolean(obj, attrs, key, op_type, arg_pos);
    } else if (std::set<std::string>({"start_op_index",
                                      "end_op_index",
                                      "program_id",
                                      "cuda_graph_pool_id"})
                   .count(key)) {
      CastPyArg2AttrLong(obj, attrs, key, op_type, arg_pos);
    } else if (std::set<std::string>({"fx",
                                      "fp",
                                      "fm",
                                      "fo",
                                      "bx",
                                      "no_need_buffers",
                                      "bp",
                                      "bm",
                                      "bo_g",
                                      "bx_g",
                                      "bp_g",
                                      "bo"})
                   .count(key)) {
      CastPyArg2AttrValues(obj, attrs, key, op_type, arg_pos);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s is not defined in this function.", key));  // NOLINT
    }
  }
}

unsigned long GetUnsignedLongFromArgs(  // NOLINT
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable) {
  PyObject* item = PyTuple_GET_ITEM(args, arg_idx);

  if (item == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be long, but got None",
          op_type,
          arg_name,
          arg_idx));
    }
    return 0;
  }

  if (PyObject_CheckLongOrToLong(&item)) {
    return PyLong_AsUnsignedLong(item);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be "
        "long, but got %s",
        op_type,
        arg_name,
        arg_idx,
        ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
  }
}

void InitOpsAttrTypeMap() {
  auto op_info_map = paddle::framework::OpInfoMap::Instance().map();
  for (auto& item : op_info_map) {
    auto op_proto = item.second.proto_;
    if (op_proto == nullptr) {
      continue;
    }
    auto attrs_proto = op_proto->attrs();
    for (auto& attr : attrs_proto) {
      OpAttrTypeMap::Instance().Map()[item.first][attr.name()] = attr.type();
    }
  }
  const auto& extra_attr_maps =
      operators::ExtraInfoUtils::Instance().GetAllExtraAttrsMap();
  for (const auto& extra_attrs : extra_attr_maps) {
    for (auto& attr : extra_attrs.second) {
      OpAttrTypeMap::Instance().Map()[extra_attrs.first][attr.first] =
          static_cast<paddle::framework::proto::AttrType>(attr.second.index() -
                                                          1);
    }
  }
}

ssize_t GetIdxFromCoreOpsInfoMap(
    const std::unordered_map<std::string, std::vector<std::string>>&
        core_ops_info_map,
    const std::string& op_type,
    const std::string& name) {
  // `core_ops_info_map` can be `core_ops_args_info` or `core_ops_returns_info`.
  // `core_ops_args_info`: get index from core_ops_args_info[op_type] according
  // to input name.
  // `core_ops_returns_info`: get index from core_ops_returns_info[op_type]
  // according to return name.
  if (!core_ops_info_map.count(op_type)) {
    PADDLE_THROW(platform::errors::Fatal(
        "Op %s is not found in core_ops_*_info map.", op_type));
  } else {
    auto args_list = core_ops_info_map.at(op_type);
    auto it = std::find(args_list.begin(), args_list.end(), name);
    if (it == args_list.end()) {
      PADDLE_THROW(platform::errors::Fatal(
          "%s is not found in op %s's args.", name, op_type));
    } else {
      return std::distance(args_list.begin(), it);
    }
  }
  return -1;
}

}  // namespace pybind
}  // namespace paddle
