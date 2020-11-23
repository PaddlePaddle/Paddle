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
}  // namespace pybind
}  // namespace paddle

// This include must be the last line
#include "paddle/fluid/pybind/op_function_impl.h"
