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
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "gflags/gflags.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"
#include "pybind11/stl.h"

DECLARE_double(eager_delete_tensor_gb);
DECLARE_bool(use_mkldnn);
DECLARE_bool(use_ngraph);
DECLARE_bool(use_system_allocator);
DECLARE_bool(free_idle_chunk);
DECLARE_bool(free_when_no_cache_hit);

namespace paddle {
namespace pybind {

namespace py = pybind11;

class PYBIND11_HIDDEN GlobalVarGetterSetterRegistry {
  DISABLE_COPY_AND_ASSIGN(GlobalVarGetterSetterRegistry);

  GlobalVarGetterSetterRegistry() = default;

 public:
  using Setter = std::function<void(const py::object &)>;
  using Getter = std::function<py::object()>;

  static const GlobalVarGetterSetterRegistry &Instance() { return instance_; }

  static GlobalVarGetterSetterRegistry *MutableInstance() { return &instance_; }

  void RegisterGetter(const std::string &name, Getter func) {
    PADDLE_ENFORCE_EQ(
        getters_.count(name), 0,
        platform::errors::AlreadyExists(
            "Getter of global variable %s has been registered", name));
    PADDLE_ENFORCE_NOT_NULL(func, platform::errors::InvalidArgument(
                                      "Getter of %s should not be null", name));
    getters_[name] = std::move(func);
  }

  void RegisterSetter(const std::string &name, Setter func) {
    PADDLE_ENFORCE_EQ(
        HasGetterMethod(name), true,
        platform::errors::NotFound(
            "Cannot register setter for %s before register getter", name));

    PADDLE_ENFORCE_EQ(
        setters_.count(name), 0,
        platform::errors::AlreadyExists(
            "Setter of global variable %s has been registered", name));
    PADDLE_ENFORCE_NOT_NULL(func, platform::errors::InvalidArgument(
                                      "Setter of %s should not be null", name));
    setters_[name] = std::move(func);
  }

  const Getter &GetterMethod(const std::string &name) const {
    PADDLE_ENFORCE_EQ(
        HasGetterMethod(name), true,
        platform::errors::NotFound("Cannot find global variable %s", name));
    return getters_.at(name);
  }

  py::object GetOrReturnDefaultValue(const std::string &name,
                                     const py::object &default_value) const {
    if (HasGetterMethod(name)) {
      return GetterMethod(name)();
    } else {
      return default_value;
    }
  }

  py::object Get(const std::string &name) const { return GetterMethod(name)(); }

  const Setter &SetterMethod(const std::string &name) const {
    PADDLE_ENFORCE_EQ(
        HasSetterMethod(name), true,
        platform::errors::NotFound("Global variable %s is not writable", name));
    return setters_.at(name);
  }

  void Set(const std::string &name, const py::object &value) const {
    SetterMethod(name)(value);
  }

  bool HasGetterMethod(const std::string &name) const {
    return getters_.count(name) > 0;
  }

  bool HasSetterMethod(const std::string &name) const {
    return setters_.count(name) > 0;
  }

  std::unordered_set<std::string> Keys() const {
    std::unordered_set<std::string> keys;
    keys.reserve(getters_.size());
    for (auto &pair : getters_) {
      keys.insert(pair.first);
    }
    return keys;
  }

 private:
  static GlobalVarGetterSetterRegistry instance_;

  std::unordered_map<std::string, Getter> getters_;
  std::unordered_map<std::string, Setter> setters_;
};

GlobalVarGetterSetterRegistry GlobalVarGetterSetterRegistry::instance_;

static void RegisterGlobalVarGetterSetter();

void BindGlobalValueGetterSetter(pybind11::module *module) {
  RegisterGlobalVarGetterSetter();

  py::class_<GlobalVarGetterSetterRegistry>(*module,
                                            "GlobalVarGetterSetterRegistry")
      .def("__getitem__", &GlobalVarGetterSetterRegistry::Get)
      .def("__setitem__", &GlobalVarGetterSetterRegistry::Set)
      .def("__contains__", &GlobalVarGetterSetterRegistry::HasGetterMethod)
      .def("keys", &GlobalVarGetterSetterRegistry::Keys)
      .def("get", &GlobalVarGetterSetterRegistry::GetOrReturnDefaultValue,
           py::arg("key"), py::arg("default") = py::cast<py::none>(Py_None));

  module->def("globals", &GlobalVarGetterSetterRegistry::Instance,
              py::return_value_policy::reference);
}

#define REGISTER_GLOBAL_VAR_GETTER_ONLY(var)                        \
  GlobalVarGetterSetterRegistry::MutableInstance()->RegisterGetter( \
      #var, []() -> py::object { return py::cast(var); })

#define REGISTER_GLOBAL_VAR_SETTER_ONLY(var)                          \
  GlobalVarGetterSetterRegistry::MutableInstance()->RegisterSetter(   \
      #var, [](const py::object &obj) {                               \
        using ValueType = std::remove_reference<decltype(var)>::type; \
        var = py::cast<ValueType>(obj);                               \
      })

#define REGISTER_GLOBAL_VAR_GETTER_SETTER(var) \
  REGISTER_GLOBAL_VAR_GETTER_ONLY(var);        \
  REGISTER_GLOBAL_VAR_SETTER_ONLY(var)

static void RegisterGlobalVarGetterSetter() {
  REGISTER_GLOBAL_VAR_GETTER_ONLY(FLAGS_use_mkldnn);
  REGISTER_GLOBAL_VAR_GETTER_ONLY(FLAGS_use_ngraph);
  REGISTER_GLOBAL_VAR_GETTER_SETTER(FLAGS_eager_delete_tensor_gb);
  REGISTER_GLOBAL_VAR_GETTER_SETTER(FLAGS_use_system_allocator);
  REGISTER_GLOBAL_VAR_GETTER_ONLY(FLAGS_free_idle_chunk);
  REGISTER_GLOBAL_VAR_GETTER_ONLY(FLAGS_free_when_no_cache_hit);
}

}  // namespace pybind
}  // namespace paddle
