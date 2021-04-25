// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/pybind/imperative.h"

namespace py = pybind11;
namespace paddle {
namespace pybind {

static inline std::shared_ptr<imperative::VarBase> CastPyHandleToVarBase(
    const std::string& op_type, const std::string& arg_name, int arg_idx,
    const py::handle& handle, bool dispensable = false) {
  PyObject* py_obj = handle.ptr();  // get underlying PyObject
  if (!py_obj || py_obj == Py_None) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be Tensor, but got "
          "%s",
          op_type, arg_name, arg_idx, Py_TYPE(py_obj)->tp_name));
    }
    return nullptr;
  }
  try {
    return py::cast<std::shared_ptr<imperative::VarBase>>(py::handle(py_obj));
  } catch (py::cast_error&) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be Tensor, but got "
        "%s",
        op_type, arg_name, arg_idx, Py_TYPE(py_obj)->tp_name));
  }
}

static inline std::vector<std::shared_ptr<imperative::VarBase>>
CastPyHandleToVarBaseList(const std::string& op_type,
                          const std::string& arg_name, int arg_idx,
                          const py::handle& handle, bool dispensable = false) {
  PyObject* py_obj = handle.ptr();  // get underlying PyObject
  if (!py_obj || py_obj == Py_None) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be Tensor, but got "
          "%s",
          op_type, arg_name, arg_idx, Py_TYPE(py_obj)->tp_name));
    }
    return {};
  }
  std::vector<std::shared_ptr<imperative::VarBase>> result;
  if (PyList_Check(py_obj) || PyTuple_Check(py_obj)) {
    auto size = PyTuple_Check(py_obj) ? PyTuple_GET_SIZE(py_obj)
                                      : PyList_GET_SIZE(py_obj);
    for (auto i = 0; i < size; ++i) {
      PyObject* item = PyTuple_Check(py_obj) ? PyTuple_GET_ITEM(py_obj, i)
                                             : PyList_GET_ITEM(py_obj, i);
      if (!item || item == Py_None) {
        result.emplace_back(nullptr);
        continue;
      }
      try {
        result.emplace_back(
            py::cast<std::shared_ptr<imperative::VarBase>>(py::handle(item)));
      } catch (py::cast_error&) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument '%s' (position %d) must be list of "
            "Tensors, but "
            "got %s in list (item %d)",
            op_type, arg_name, arg_idx, Py_TYPE(item)->tp_name, i));
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be list of Tensors, but got "
        "%s",
        op_type, arg_name, arg_idx, Py_TYPE(py_obj)->tp_name));
  }
  return result;
}  // namespace pybind

static inline void ConstructAttrMapFromPyArgs(const std::string& op_type,
                                              int start_idx,
                                              framework::AttributeMap* attrs,
                                              const py::args& args) {
  PADDLE_ENFORCE_EQ(
      args.size() % 2, 0,
      platform::errors::InvalidArgument(
          "The number of arguments for arributes should be even."));
  for (size_t i = 0; i < args.size(); i += 2) {
    std::string name;
    framework::Attribute value;
    try {
      name = args[i].cast<std::string>();
    } catch (std::exception& e) {
      PyObject* py_obj = args[i].ptr();  // get underlying PyObject
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument (position %d) must be str, but got "
          "%s",
          op_type, start_idx + i, Py_TYPE(py_obj)->tp_name));
    }
    try {
      value = args[i + 1].cast<framework::Attribute>();
    } catch (std::exception& e) {
      PyObject* py_obj = args[i + 1].ptr();  // get underlying PyObject
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument (position %d) must be "
          "Attribute type (one of str, bool, int, int64, float, or list of "
          "them), but got %s",
          op_type, start_idx + i + 1, Py_TYPE(py_obj)->tp_name));
    }
    (*attrs)[name] = value;
  }
}

static inline std::vector<std::shared_ptr<imperative::VarBase>>
ConstructDuplicableOutput(const size_t num) {
  auto tracer = imperative::GetCurrentTracer();
  std::vector<std::shared_ptr<imperative::VarBase>> res;
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    auto var_base_name = tracer->GenerateUniqueName();
    res.emplace_back(new imperative::VarBase(var_base_name));
  }
  return res;
}

static inline void HandleViewBetweenInputAndOutput(
    const std::shared_ptr<imperative::VarBase>& input_var,
    const std::shared_ptr<imperative::VarBase>& view_output_var) {
  PADDLE_ENFORCE_EQ(
      input_var->Var().IsInitialized(), true,
      platform::errors::InvalidArgument("Tensor %s has not been initialized!",
                                        input_var->Name()));

  if (input_var->Var().IsType<framework::LoDTensor>()) {
    const auto& input_tensor = input_var->Var().Get<framework::LoDTensor>();
    PADDLE_ENFORCE_EQ(
        input_tensor.IsInitialized(), true,
        platform::errors::InvalidArgument(
            "LoDTensor %s has not been initialized!", input_var->Name()));

    auto* view_output_tensor =
        view_output_var->MutableVar()->GetMutable<framework::LoDTensor>();
    view_output_tensor->ShareDataWith(input_tensor);
    view_output_tensor->ShareInplaceVersionCounterWith(input_tensor);

    VLOG(3) << "Perform View between Output Var(" << view_output_var->Name()
            << ") and Input Var(" << input_var->Name()
            << "), share allocation and inplace version.";
  }
}

extern PyTypeObject* g_VarBase_PyType;

static inline void ConstructAttrMapFromPyArgs(
    const std::string& op_type, PyObject* args, ssize_t attr_start,
    ssize_t attr_end, paddle::framework::AttributeMap& attrs) {  // NOLINT
  PADDLE_ENFORCE_EQ(
      (attr_end - attr_start + 1) % 2, 0,
      platform::errors::InvalidArgument(
          "The number of arguments for arributes should be even."));

  auto attr_type_map =
      paddle::framework::OpInfoMap::Instance().Get(op_type).attrs_type_;

  PyObject* obj = nullptr;
  for (ssize_t arg_pos = attr_start; arg_pos < attr_end + 1; arg_pos += 2) {
    Py_ssize_t key_len;
    const char* key_prt;
    obj = PyTuple_GET_ITEM(args, arg_pos);
    if (PyUnicode_Check(obj)) {
      key_prt = PyUnicode_AsUTF8AndSize(obj, &key_len);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument (position %d) must be str, but got "
          "%s",
          op_type, arg_pos, ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
    }

    std::string key(key_prt, (size_t)key_len);

    obj = PyTuple_GET_ITEM(args, arg_pos + 1);

    auto iter = attr_type_map->find(key);

    if (iter == attr_type_map->end()) {
      continue;
    }

    switch (iter->second) {
      case paddle::framework::proto::AttrType::INT:
        if (PyLong_Check(obj)) {
          attrs[key] = (int)PyLong_AsLong(obj);  // NOLINT
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "int, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      case paddle::framework::proto::AttrType::FLOAT:
        if (PyFloat_Check(obj) || PyLong_Check(obj)) {
          attrs[key] = (float)PyFloat_AsDouble(obj);  // NOLINT
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "float, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      case paddle::framework::proto::AttrType::STRING:
        if (PyUnicode_Check(obj)) {
          Py_ssize_t size;
          const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
          attrs[key] = std::string(data, (size_t)size);
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "str, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      case paddle::framework::proto::AttrType::INTS:
        if (PyList_Check(obj)) {
          Py_ssize_t len = PyList_Size(obj);
          PyObject* item = nullptr;
          std::vector<int> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyList_GetItem(obj, i);
            if (PyLong_Check(item)) {
              value.emplace_back(PyLong_AsLong(item));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of int, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else if (PyTuple_Check(obj)) {
          Py_ssize_t len = PyTuple_Size(obj);
          PyObject* item = nullptr;
          std::vector<int> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyTuple_GetItem(obj, i);
            if (PyLong_Check(item)) {
              value.emplace_back(PyLong_AsLong(item));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of int, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "list or tuple, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      case paddle::framework::proto::AttrType::FLOATS:
        if (PyList_Check(obj)) {
          Py_ssize_t len = PyList_Size(obj);
          PyObject* item = nullptr;
          std::vector<float> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyList_GetItem(obj, i);
            if (PyFloat_Check(item) || PyLong_Check(item)) {
              value.emplace_back(PyFloat_AsDouble(item));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of float, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else if (PyTuple_Check(obj)) {
          Py_ssize_t len = PyTuple_Size(obj);
          PyObject* item = nullptr;
          std::vector<float> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyTuple_GetItem(obj, i);
            if (PyFloat_Check(item) || PyLong_Check(item)) {
              value.emplace_back(PyFloat_AsDouble(item));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of float, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "list or tuple, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      case paddle::framework::proto::AttrType::STRINGS:
        if (PyList_Check(obj)) {
          Py_ssize_t len = PyList_Size(obj);
          PyObject* item = nullptr;
          std::vector<std::string> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyList_GetItem(obj, i);
            if (PyUnicode_Check(item)) {
              Py_ssize_t size;
              const char* data = PyUnicode_AsUTF8AndSize(item, &size);
              value.emplace_back(std::string(data, (size_t)size));  // NOLINT
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of str, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else if (PyTuple_Check(obj)) {
          Py_ssize_t len = PyTuple_Size(obj);
          PyObject* item = nullptr;
          std::vector<std::string> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyTuple_GetItem(obj, i);
            if (PyUnicode_Check(item)) {
              Py_ssize_t size;
              const char* data = PyUnicode_AsUTF8AndSize(item, &size);
              value.emplace_back(std::string(data, (size_t)size));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of str, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "list or tuple, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      case paddle::framework::proto::AttrType::BOOLEAN:
        if (PyBool_Check(obj)) {
          attrs[key] = (bool)PyLong_AsLong(obj);  // NOLINT
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "bool, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      case paddle::framework::proto::AttrType::BOOLEANS:
        if (PyList_Check(obj)) {
          Py_ssize_t len = PyList_Size(obj);
          PyObject* item = nullptr;
          std::vector<bool> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyList_GetItem(obj, i);
            if (PyBool_Check(item)) {
              value.emplace_back(PyLong_AsLong(item));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of bool, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else if (PyTuple_Check(obj)) {
          Py_ssize_t len = PyTuple_Size(obj);
          PyObject* item = nullptr;
          std::vector<bool> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyTuple_GetItem(obj, i);
            if (PyBool_Check(item)) {
              value.emplace_back(PyLong_AsLong(item));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of bool, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "list or tuple, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      case paddle::framework::proto::AttrType::LONG:
        if (PyLong_Check(obj)) {
          attrs[key] = (int64_t)PyLong_AsLong(obj);  // NOLINT
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "long, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      case paddle::framework::proto::AttrType::LONGS:
        if (PyList_Check(obj)) {
          Py_ssize_t len = PyList_Size(obj);
          PyObject* item = nullptr;
          std::vector<int64_t> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyList_GetItem(obj, i);
            if (PyLong_Check(item)) {
              value.emplace_back(PyLong_AsLong(item));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of long, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else if (PyTuple_Check(obj)) {
          Py_ssize_t len = PyTuple_Size(obj);
          PyObject* item = nullptr;
          std::vector<int64_t> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyTuple_GetItem(obj, i);
            if (PyLong_Check(item)) {
              value.emplace_back(PyLong_AsLong(item));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of long, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "list or tuple, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      case paddle::framework::proto::AttrType::FLOAT64S:
        if (PyList_Check(obj)) {
          Py_ssize_t len = PyList_Size(obj);
          PyObject* item = nullptr;
          std::vector<double> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyList_GetItem(obj, i);
            if (PyFloat_Check(item) || PyLong_Check(item)) {
              value.emplace_back(PyFloat_AsDouble(item));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of float, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else if (PyTuple_Check(obj)) {
          Py_ssize_t len = PyTuple_Size(obj);
          PyObject* item = nullptr;
          std::vector<double> value;
          for (Py_ssize_t i = 0; i < len; i++) {
            item = PyTuple_GetItem(obj, i);
            if (PyFloat_Check(item) || PyLong_Check(item)) {
              value.emplace_back(PyFloat_AsDouble(item));
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "%s(): argument (position %d) must be "
                  "list of float, but got %s item",
                  op_type, arg_pos + 1,
                  ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
            }
          }
          attrs[key] = value;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "%s(): argument (position %d) must be "
              "list or tuple, but got %s",
              op_type, arg_pos + 1,
              ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
        }
        break;
      default:
        break;
    }
  }
}

static inline std::shared_ptr<imperative::VarBase> GetVarBaseFromArgs(
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable = false) {
  ::pybind11::detail::instance* inst =
      (::pybind11::detail::instance*)PyTuple_GET_ITEM(args, arg_idx);

  if (inst == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be Tensor, but got None",
          op_type, arg_name, arg_idx));
    }
    return nullptr;
  }

  if (!PyObject_IsInstance((PyObject*)inst,                 // NOLINT
                           (PyObject*)g_VarBase_PyType)) {  // NOLINT
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be Tensor, but got "
        "%s",
        op_type, arg_name, arg_idx,
        ((PyTypeObject*)((PyObject*)inst)->ob_type)->tp_name));  // NOLINT
  }

  return *((std::shared_ptr<paddle::imperative::VarBase>*)
               inst->simple_value_holder);  // NOLINT
}

static inline std::vector<std::shared_ptr<imperative::VarBase>>
GetVarBaseListFromArgs(const std::string& op_type, const std::string& arg_name,
                       PyObject* args, ssize_t arg_idx,
                       bool dispensable = false) {
  PyObject* list = PyTuple_GET_ITEM(args, arg_idx);

  if (list == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensor, but got "
          "None",
          op_type, arg_name, arg_idx));  // NOLINT
    }
    return {};
  }

  std::vector<std::shared_ptr<imperative::VarBase>> result;

  if (PyList_Check(list)) {
    Py_ssize_t len = PyList_Size(list);
    ::pybind11::detail::instance* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = (::pybind11::detail::instance*)PyList_GetItem(list, i);
      if (!PyObject_IsInstance((PyObject*)item,                 // NOLINT
                               (PyObject*)g_VarBase_PyType)) {  // NOLINT
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument '%s' (position %d) must be list of Tensors, but "
            "got list of "
            "%s",
            op_type, arg_name, arg_idx,
            ((PyTypeObject*)((PyObject*)item)->ob_type)->tp_name));  // NOLINT
      }
      result.emplace_back(*((std::shared_ptr<paddle::imperative::VarBase>*)
                                item->simple_value_holder));  // NOLINT
    }
  } else if (PyTuple_Check(list)) {
    Py_ssize_t len = PyTuple_Size(list);
    ::pybind11::detail::instance* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = (::pybind11::detail::instance*)PyTuple_GetItem(list, i);  // NOLINT
      if (!PyObject_IsInstance((PyObject*)item,                        // NOLINT
                               (PyObject*)g_VarBase_PyType)) {         // NOLINT
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s(): argument '%s' (position %d) must be list of Tensors, but "
            "got list of "
            "%s",
            op_type, arg_name, arg_idx,
            ((PyTypeObject*)((PyObject*)item)->ob_type)->tp_name));  // NOLINT
      }
      result.emplace_back(*((std::shared_ptr<paddle::imperative::VarBase>*)
                                item->simple_value_holder));
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be list of Tensors, but got "
        "%s",
        op_type, arg_name, arg_idx,
        ((PyTypeObject*)list->ob_type)->tp_name));  // NOLINT
  }

  return result;
}

static inline unsigned long GetUnsignedLongFromArgs(  // NOLINT
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable = false) {
  PyObject* item = PyTuple_GET_ITEM(args, arg_idx);

  if (item == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be long, but got None",
          op_type, arg_name, arg_idx));
    }
    return 0;
  }

  if (PyLong_Check(item)) {
    return PyLong_AsUnsignedLong(item);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be "
        "long, but got %s",
        op_type, arg_name, arg_idx,
        ((PyTypeObject*)item->ob_type)->tp_name));  // NOLINT
  }
}

static inline PyObject* MakeReturnPyObject(
    const std::shared_ptr<paddle::imperative::VarBase>& out) {
  ::pybind11::detail::instance* result;
  result = (::pybind11::detail::instance*)g_VarBase_PyType->tp_alloc(
      g_VarBase_PyType, 0);
  if (result != NULL) {
    result->simple_layout = 1;
    *((std::shared_ptr<paddle::imperative::VarBase>*)
          result->simple_value_holder) = out;
  }
  return (PyObject*)result;  // NOLINT
}

static inline PyObject* MakeReturnPyObject(
    const std::vector<std::shared_ptr<imperative::VarBase>>& out) {
  PyObject* result = PyList_New((Py_ssize_t)out.size());

  ::pybind11::detail::instance* item = NULL;
  for (size_t i = 0; i < out.size(); i++) {
    item = (::pybind11::detail::instance*)g_VarBase_PyType->tp_alloc(
        g_VarBase_PyType, 0);
    if (item != NULL) {
      item->simple_layout = 1;
      *((std::shared_ptr<paddle::imperative::VarBase>*)
            item->simple_value_holder) = out[i];
    }
    PyList_SET_ITEM(result, (Py_ssize_t)i, (PyObject*)item);  // NOLINT
  }

  return result;
}

template <typename Tuple, size_t N>
struct TupleVarBasesResult {
  static void Run(const Tuple& out, PyObject* result) {
    TupleVarBasesResult<Tuple, N - 1>::Run(out, result);
    PyTuple_SET_ITEM(result, N - 1, MakeReturnPyObject(std::get<N - 1>(out)));
  }
};

template <typename Tuple>
struct TupleVarBasesResult<Tuple, 1> {
  static void Run(const Tuple& out, PyObject* result) {
    PyTuple_SET_ITEM(result, 0, MakeReturnPyObject(std::get<0>(out)));
  }
};

template <typename... Args>
static inline PyObject* MakeReturnPyObject(const std::tuple<Args...>& out) {
  auto len = sizeof...(Args);  // std::tuple_size<decltype(out)>::value;
  PyObject* result = PyTuple_New(len);

  TupleVarBasesResult<decltype(out), sizeof...(Args)>::Run(out, result);

  return result;
}

}  // namespace pybind
}  // namespace paddle

// This include must be the last line
#include "paddle/fluid/pybind/op_function_impl.h"
