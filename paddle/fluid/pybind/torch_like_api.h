// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/library.h>

#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;

namespace torch {

class PyTorchStyleOperationInvoker {
 public:
  static py::object invoke_operator_from_python(
      const std::string& qualified_name,
      const py::args& args,
      const py::kwargs& kwargs);

  static std::pair<const CppFunction*, FunctionArgs> get_op_with_args(
      const std::string& qualified_name,
      const py::args& args,
      const py::kwargs& kwargs);

  static py::object to_py_object(const torch::IValue& value);

  static torch::IValue to_ivalue(py::handle obj);

  static py::object create_python_callable(const std::string& qualified_name);

 private:
  static FunctionArgs convert_args_kwargs_to_function_args(
      const py::args& args, const py::kwargs& kwargs);

  static py::object convert_result_to_python(const FunctionResult& result);
};

class PyTorchStyleError : public std::runtime_error {
 public:
  PyTorchStyleError(const std::string& msg)  // NOLINT
      : std::runtime_error(msg) {}
};

inline py::object PyTorchStyleOperationInvoker::invoke_operator_from_python(
    const std::string& qualified_name,
    const py::args& args,
    const py::kwargs& kwargs) {
  try {
    auto [found_op, function_args] =
        get_op_with_args(qualified_name, args, kwargs);

    FunctionResult result;
    {
      py::gil_scoped_release no_gil_guard;
      result = found_op->call_with_args(function_args);
    }

    return convert_result_to_python(result);
  } catch (const std::exception& e) {
    throw PyTorchStyleError("Error in operator '" + qualified_name +
                            "': " + e.what());
  }
}

inline std::pair<const CppFunction*, FunctionArgs>
PyTorchStyleOperationInvoker::get_op_with_args(
    const std::string& qualified_name,
    const py::args& args,
    const py::kwargs& kwargs) {
  auto* op = OperatorRegistry::instance().find_operator(qualified_name);
  if (!op) {
    throw PyTorchStyleError("Operator " + qualified_name + " not found!");
  }

  auto impl_it = op->implementations.find(DispatchKey::CPU);
  if (impl_it == op->implementations.end()) {
    throw PyTorchStyleError("No CPU implementation found for " +
                            qualified_name);
  }

  FunctionArgs function_args =
      convert_args_kwargs_to_function_args(args, kwargs);

  return std::make_pair(&impl_it->second, std::move(function_args));
}

inline py::object PyTorchStyleOperationInvoker::to_py_object(
    const torch::IValue& value) {
  if (value.is_none()) {
    return py::none();
  } else if (value.is_bool()) {
    return py::cast(value.to_bool());
  } else if (value.is_int()) {
    return py::cast(value.to_int());
  } else if (value.is_double()) {
    return py::cast(value.to_double());
  } else if (value.is_string()) {
    return py::cast(value.to_string());
  } else if (value.is_tensor()) {
    return py::reinterpret_borrow<py::object>(
        paddle::pybind::ToPyObject(value.to_tensor()._PD_GetInner()));
  } else {
    throw PyTorchStyleError(
        "Unknown torch::IValue type in toPyObject conversion");
  }
}

inline torch::IValue PyTorchStyleOperationInvoker::to_ivalue(py::handle obj) {
  if (obj.is_none()) {
    return torch::IValue();  // None
  } else if (py::isinstance<py::bool_>(obj)) {
    return torch::IValue(py::cast<bool>(obj));
  } else if (py::isinstance<py::int_>(obj)) {
    return torch::IValue(py::cast<int>(obj));
  } else if (py::isinstance<py::float_>(obj)) {
    return torch::IValue(py::cast<double>(obj));
  } else if (py::isinstance<py::str>(obj)) {
    return torch::IValue(py::cast<std::string>(obj));
  } else if (paddle::pybind::PyCheckTensor(obj.ptr())) {
    return torch::IValue(paddle::pybind::CastPyArg2Tensor(obj.ptr(), 0));
  } else {
    try {
      auto val = py::cast<int>(obj);
      return torch::IValue(val);
    } catch (...) {
      try {
        auto val = py::cast<double>(obj);
        return torch::IValue(val);
      } catch (...) {
        try {
          auto val = py::cast<std::string>(obj);
          return torch::IValue(val);
        } catch (...) {
          throw PyTorchStyleError(
              "Cannot convert Python object to torch::IValue: unsupported "
              "type " +
              std::string(py::str(py::type::of(obj))));
        }
      }
    }
  }
}

inline FunctionArgs
PyTorchStyleOperationInvoker::convert_args_kwargs_to_function_args(
    const py::args& args, const py::kwargs& kwargs) {
  FunctionArgs function_args;

  for (const auto& arg : args) {
    torch::IValue value = to_ivalue(arg);
    function_args.add_arg(std::move(value));
  }

  for (auto item : kwargs) {
    py::str key = item.first.cast<py::str>();
    py::object value_obj = item.second.cast<py::object>();

    torch::IValue value = to_ivalue(value_obj);
    function_args.add_arg(std::move(value));
  }

  return function_args;
}

inline py::object PyTorchStyleOperationInvoker::convert_result_to_python(
    const FunctionResult& result) {
  if (!result.has_value()) {
    return py::none();
  }

  const torch::IValue& value = result.get_value();
  return to_py_object(value);
}

inline py::object PyTorchStyleOperationInvoker::create_python_callable(
    const std::string& qualified_name) {
  return py::cpp_function(
      [qualified_name](py::args args, py::kwargs kwargs) -> py::object {
        return invoke_operator_from_python(qualified_name, args, kwargs);
      },
      py::name(qualified_name.c_str()),
      py::is_method(py::none()));
}

inline py::object get_operation(const std::string& qualified_name) {
  return PyTorchStyleOperationInvoker::create_python_callable(qualified_name);
}
}  // namespace torch

namespace paddle::pybind {

void BindTorchLikeApi(pybind11::module* m) {
  py::class_<torch::IValue>(*m, "IValue")
      .def(py::init<>())
      .def(py::init<int>())
      .def(py::init<double>())
      .def(py::init<bool>())
      .def(py::init<std::string>())
      .def("is_none", &torch::IValue::is_none)
      .def("is_int", &torch::IValue::is_int)
      .def("is_double", &torch::IValue::is_double)
      .def("is_bool", &torch::IValue::is_bool)
      .def("is_string", &torch::IValue::is_string)
      .def("to_int", &torch::IValue::to_int)
      .def("to_double", &torch::IValue::to_double)
      .def("to_bool", &torch::IValue::to_bool)
      .def("to_string", &torch::IValue::to_string)
      .def("__repr__", [](const torch::IValue& v) {
        if (v.is_none()) return std::string("IValue(None)");
        if (v.is_int())
          return std::string("IValue(") + std::to_string(v.to_int()) + ")";
        if (v.is_double())
          return std::string("IValue(") + std::to_string(v.to_double()) + ")";
        if (v.is_bool())
          return std::string("IValue(") + (v.to_bool() ? "True" : "False") +
                 ")";
        if (v.is_string())
          return std::string("IValue(\"") + v.to_string() + "\")";
        return std::string("IValue(unknown)");
      });

  m->def("get_operation",
         &torch::get_operation,
         "Get a PyTorch-style callable for the specified operation",
         py::arg("qualified_name"));
}
}  // namespace paddle::pybind
