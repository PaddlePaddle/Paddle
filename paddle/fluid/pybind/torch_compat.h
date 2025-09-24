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

#include "paddle/common/exception.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/api/include/compat/utils/scalar_type_conversion.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;

namespace torch {

class OperationInvoker {
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

  static FunctionArgs convert_args_kwargs_to_function_args(
      const py::args& args, const py::kwargs& kwargs);

  static py::object convert_result_to_python(const FunctionResult& result);
};

inline py::object OperationInvoker::invoke_operator_from_python(
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
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Error in operator '%s': %s", qualified_name.c_str(), e.what()));
  }
}

inline std::pair<const CppFunction*, FunctionArgs>
OperationInvoker::get_op_with_args(const std::string& qualified_name,
                                   const py::args& args,
                                   const py::kwargs& kwargs) {
  auto* op = OperatorRegistry::instance().find_operator(qualified_name);
  if (!op) {
    PADDLE_THROW(common::errors::NotFound(
        "Operator '%s' not found in the registry", qualified_name.c_str()));
  }

  auto impl_it = op->implementations.find(DispatchKey::CPU);
  if (impl_it == op->implementations.end()) {
    PADDLE_THROW(common::errors::NotFound(
        "No CPU implementation found for operator '%s'",
        qualified_name.c_str()));
  }

  FunctionArgs function_args =
      convert_args_kwargs_to_function_args(args, kwargs);

  return std::make_pair(&impl_it->second, std::move(function_args));
}

inline py::object OperationInvoker::to_py_object(const torch::IValue& value) {
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
  } else if (value.is_list()) {
    auto ivalue_list = value.to_list();
    py::list py_list;
    for (const auto& item : ivalue_list) {
      py_list.append(to_py_object(item));
    }
    return py_list;
  } else if (value.is_tuple()) {
    auto ivalue_tuple = value.to_tuple();
    size_t size = ivalue_tuple.size();
    py::tuple py_tuple(size);
    for (size_t i = 0; i < size; ++i) {
      py_tuple[i] = to_py_object(ivalue_tuple[i]);
    }
    return py_tuple;
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Conversion of torch::IValue to Python object for type %s is not "
        "implemented yet.",
        value.type_string()));
  }
}

inline torch::IValue OperationInvoker::to_ivalue(py::handle obj) {
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
  } else if (paddle::pybind::PyObject_CheckDataType(obj.ptr())) {
    return torch::IValue(compat::_PD_PhiDataTypeToAtenScalarType(
        paddle::pybind::CastPyArg2DataType(obj.ptr(), "to_ivalue", 0)));
  } else if (py::isinstance<py::list>(obj)) {
    auto list = obj.cast<py::list>();
    std::vector<torch::IValue> ivalue_list;
    ivalue_list.reserve(list.size());
    for (auto item : list) {
      ivalue_list.push_back(to_ivalue(item));
    }
    return torch::IValue(ivalue_list);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Conversion of Python object to torch::IValue for type %s is not "
        "implemented yet.",
        std::string(py::str(py::type::of(obj))).c_str()));
  }
}

inline FunctionArgs OperationInvoker::convert_args_kwargs_to_function_args(
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

inline py::object OperationInvoker::convert_result_to_python(
    const FunctionResult& result) {
  if (!result.has_value()) {
    return py::none();
  }

  const torch::IValue& value = result.get_value();
  return to_py_object(value);
}

inline py::object OperationInvoker::create_python_callable(
    const std::string& qualified_name) {
  return py::cpp_function(
      [qualified_name](py::args args, py::kwargs kwargs) -> py::object {
        return invoke_operator_from_python(qualified_name, args, kwargs);
      },
      py::name(qualified_name.c_str()),
      py::is_method(py::none()));
}

class CustomClassProxyInstance {
 public:
  CustomClassProxyInstance(const std::string& qualified_name,
                           const IValue& instance)
      : qualified_name_(qualified_name), instance_(instance) {}

  // Get instance method
  py::object __getattr__(const std::string& method_name) {
    if (ClassRegistry::instance().has_method(qualified_name_, method_name)) {
      return py::cpp_function(
          [this, method_name](py::args args, py::kwargs kwargs) -> py::object {
            FunctionArgs function_args;
            function_args.add_arg(instance_);  // this pointer
            for (auto arg :
                 OperationInvoker::convert_args_kwargs_to_function_args(
                     args, kwargs)) {
              function_args.add_arg(std::move(arg));
            }

            auto result = ClassRegistry::instance().call_method_with_args(
                qualified_name_, method_name, function_args);

            return OperationInvoker::convert_result_to_python(result);
          },
          py::name(method_name.c_str()));
    }

    PADDLE_THROW(common::errors::NotFound("Method '%s' not found in class %s",
                                          method_name.c_str(),
                                          qualified_name_.c_str()));
  }

  const IValue& get_instance() const { return instance_; }

 private:
  std::string qualified_name_;
  IValue instance_;
};

class CustomClassProxy {
 public:
  CustomClassProxy(const std::string& qualified_name)  // NOLINT
      : qualified_name_(qualified_name) {}

  // Create a new instance of the class
  py::object __call__(const py::args& args, const py::kwargs& kwargs) {
    try {
      FunctionArgs function_args =
          OperationInvoker::convert_args_kwargs_to_function_args(args, kwargs);

      // Call the constructor
      auto result = ClassRegistry::instance().call_constructor_with_args(
          qualified_name_, function_args);

      // Wrap the result in a CustomClassProxyInstance
      if (result.has_value()) {
        const IValue& value = result.get_value();
        // Create proxy object for the custom class instance
        return py::cast(CustomClassProxyInstance(qualified_name_, value));
      } else {
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "Constructor did not return an instance"));
      }
    } catch (const std::exception& e) {
      PADDLE_THROW(common::errors::PreconditionNotMet(
          "Failed to construct %s: %s", qualified_name_.c_str(), e.what()));
    }
  }

  // Get static method
  py::object __getattr__(const std::string& method_name) {
    // Check if the method name is a dunder method
    if (method_name.size() >= 2 && method_name.substr(0, 2) == "__") {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Dunder methods are not supported: %s", method_name.c_str()));
    }

    // Check if the class has the static method
    if (ClassRegistry::instance().has_static_method(qualified_name_,
                                                    method_name)) {
      return py::cpp_function(
          [this, method_name](py::args args, py::kwargs kwargs) -> py::object {
            // Convert args and kwargs to FunctionArgs
            FunctionArgs function_args =
                OperationInvoker::convert_args_kwargs_to_function_args(args,
                                                                       kwargs);

            // Call the static method
            auto result =
                ClassRegistry::instance().call_static_method_with_args(
                    qualified_name_, method_name, function_args);

            return OperationInvoker::convert_result_to_python(result);
          },
          py::name(method_name.c_str()));
    }

    PADDLE_THROW(
        common::errors::NotFound("Static method '%s' not found in class %s",
                                 method_name.c_str(),
                                 qualified_name_.c_str()));
  }

 private:
  std::string qualified_name_;
};

inline py::object get_custom_class_python_wrapper(
    const std::string& namespace_name, const std::string& class_name) {
  std::string qualified_name = namespace_name + "::" + class_name;

  if (!ClassRegistry::instance().has_class(qualified_name)) {
    PADDLE_THROW(common::errors::NotFound(
        "Class '%s' not found in the registry", qualified_name.c_str()));
  }

  return py::cast(CustomClassProxy(qualified_name));
}

inline py::object get_operation(const std::string& qualified_name) {
  return OperationInvoker::create_python_callable(qualified_name);
}
}  // namespace torch

namespace paddle::pybind {

void BindTorchCompat(pybind11::module* m) {
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

  py::class_<torch::CustomClassProxy>(*m, "CustomClassProxy")
      .def("__call__", &torch::CustomClassProxy::__call__)
      .def("__getattr__", &torch::CustomClassProxy::__getattr__);

  py::class_<torch::CustomClassProxyInstance>(*m, "CustomClassProxyInstance")
      .def("__getattr__", &torch::CustomClassProxyInstance::__getattr__);

  m->def("_get_operation",
         &torch::get_operation,
         "Get a callable for the specified operation",
         py::arg("qualified_name"));

  m->def("_get_custom_class_python_wrapper",
         &torch::get_custom_class_python_wrapper,
         "Get a Python wrapper for the specified custom class",
         py::arg("namespace_name"),
         py::arg("class_name"));
}
}  // namespace paddle::pybind
