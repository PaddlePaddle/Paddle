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

#include "paddle/fluid/pybind/global_value_getter_setter.h"
#include <mutex>  // NOLINT
#include <stack>
#include <string>
#include <unordered_map>
#include "Python.h"
#include "boost/any.hpp"
#include "gflags/gflags.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/pybind_boost_headers.h"

DECLARE_double(fraction_of_cpu_memory_to_use);
#ifdef PADDLE_WITH_CUDA
DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_double(fraction_of_cuda_pinned_memory_to_use);
#endif

DECLARE_bool(use_mkldnn);
#ifdef PADDLE_WITH_NGRAPH
DECLARE_bool(use_ngraph);
#endif

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

bool TryCastToInteger(PyObject *obj, int64_t *value) {
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(obj)) {
    *value = static_cast<int64_t>(PyInt_AS_LONG(obj));
    return true;
  }
#endif

  if (PyLong_Check(obj)) {
    *value = static_cast<int64_t>(PyLong_AsLong(obj));
    return true;
  }
  return false;
}

template <typename T>
static boost::any PyObjectToCppAny(const py::object &handle) {
  auto *obj = handle.ptr();
  PADDLE_ENFORCE_EQ(obj && obj != Py_None, true);

  if (std::is_same<T, std::string>::value) {
#if PY_MAJOR_VERSION < 3
    PADDLE_ENFORCE_NE(PyString_Check(obj), 0);
#else
    PADDLE_ENFORCE_NE(PyBytes_Check(obj), 0);
#endif
    return py::cast<std::string>(obj);
  }

  if (std::is_same<T, bool>::value) {
    PADDLE_ENFORCE_NE(PyBool_Check(obj), 0);
    return static_cast<bool>(obj == Py_True);
  }

  if (std::is_integral<T>::value) {
    int64_t value;
    bool success = TryCastToInteger(obj, &value);
    PADDLE_ENFORCE_EQ(success, true);
    return value;
  }

  if (std::is_floating_point<T>::value) {
    if (PyFloat_Check(obj)) {
      return static_cast<double>(PyFloat_AS_DOUBLE(obj));
    } else {
      int64_t value;
      bool success = TryCastToInteger(obj, &value);
      PADDLE_ENFORCE_EQ(success, true);
      return static_cast<double>(value);
    }
  }

  return py::cast<T>(handle);
}

class PYBIND11_HIDDEN GlobalValueManager {
 private:
  struct PYBIND11_HIDDEN GetterSetterMethod {
    std::function<py::object(const py::object &)> setter;
    std::function<py::object()> getter;
  };

  template <typename T>
  using CompatibleType = typename std::conditional<
      !std::is_same<T, bool>::value && std::is_integral<T>::value, int64_t,
      typename std::conditional<std::is_floating_point<T>::value, double,
                                T>::type>::type;

 public:
  static GlobalValueManager &Instance() {
    static GlobalValueManager manager;
    return manager;
  }

  py::object Set(const std::string &name, const py::object &obj) const {
    return GetMethod(name).setter(obj);
  }

  py::object Get(const std::string &name) const {
    return GetMethod(name).getter();
  }

 private:
  GlobalValueManager() { RegisterGlobalValues(); }

  void RegisterGlobalValues();

  const GetterSetterMethod &GetMethod(const std::string &name) const {
    auto iter = method_map_.find(name);
    PADDLE_ENFORCE_EQ(iter != method_map_.end(), true,
                      "Unsupported variable name %s", name);
    return iter->second;
  }

 private:
  std::unordered_map<std::string, GetterSetterMethod> method_map_;
};

#define PADDLE_PYBIND_REG_GLOBAL_VALUE(name)                                \
  do {                                                                      \
    PADDLE_ENFORCE_EQ(method_map_.count(#name), 0);                         \
    method_map_[#name].setter = [](const py::object &handle) {              \
      using __TYPE__ = CompatibleType<decltype(name)>;                      \
      auto old_value = name;                                                \
      name = boost::any_cast<__TYPE__>(PyObjectToCppAny<__TYPE__>(handle)); \
      return py::cast(old_value);                                           \
    };                                                                      \
    method_map_[#name].getter = [] { return py::cast(name); };              \
  } while (0)

void GlobalValueManager::RegisterGlobalValues() {
  PADDLE_PYBIND_REG_GLOBAL_VALUE(FLAGS_fraction_of_cpu_memory_to_use);
#ifdef PADDLE_WITH_CUDA
  PADDLE_PYBIND_REG_GLOBAL_VALUE(FLAGS_fraction_of_gpu_memory_to_use);
  PADDLE_PYBIND_REG_GLOBAL_VALUE(FLAGS_fraction_of_cuda_pinned_memory_to_use);
#endif

  PADDLE_PYBIND_REG_GLOBAL_VALUE(FLAGS_use_mkldnn);
#ifdef PADDLE_WITH_NGRAPH
  PADDLE_PYBIND_REG_GLOBAL_VALUE(FLAGS_use_ngraph);
#endif
}

#undef PADDLE_PYBIND_REG_GLOBAL_VALUE

void BindGlobalValueGetterSetters(pybind11::module *module) {
  auto &m = *module;
  m.def("get_global_var", [](const std::string &name) {
    return GlobalValueManager::Instance().Get(name);
  });
  m.def("set_global_var", [](const std::string &name, const py::object &obj) {
    return GlobalValueManager::Instance().Set(name, obj);
  });
}

}  // namespace pybind
}  // namespace paddle
