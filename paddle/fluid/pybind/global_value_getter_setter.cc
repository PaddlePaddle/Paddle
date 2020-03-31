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
#include <cctype>
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
DECLARE_bool(use_system_allocator);
DECLARE_bool(free_idle_chunk);
DECLARE_bool(free_when_no_cache_hit);
#ifdef PADDLE_WITH_CUDA
DECLARE_uint64(gpu_memory_limit_mb);
DECLARE_bool(cudnn_deterministic);
#endif
DECLARE_string(allocator_strategy);
DECLARE_bool(enable_parallel_graph);

namespace paddle {
namespace pybind {

namespace py = pybind11;

class PYBIND11_HIDDEN GlobalVarGetterSetterRegistry {
  DISABLE_COPY_AND_ASSIGN(GlobalVarGetterSetterRegistry);

  GlobalVarGetterSetterRegistry() = default;

 public:
  using Getter = std::function<py::object()>;
  using Setter = std::function<void(const py::object &)>;

  template <typename T>
  static Getter CreateGetter(const T &var) {
    return [&]() -> py::object { return py::cast(var); };
  }

  template <typename T>
  static Setter CreateSetter(T *var) {
    return [var](const py::object &obj) { *var = py::cast<T>(obj); };
  }

 private:
  struct VarInfo {
    VarInfo(bool is_public, const Getter &getter)
        : is_public(is_public), getter(getter) {}

    VarInfo(bool is_public, const Getter &getter, const Setter &setter)
        : is_public(is_public), getter(getter), setter(setter) {}

    const bool is_public;
    const Getter getter;
    const Setter setter;
  };

 public:
  static const GlobalVarGetterSetterRegistry &Instance() { return instance_; }

  static GlobalVarGetterSetterRegistry *MutableInstance() { return &instance_; }

  void Register(const std::string &name, bool is_public, const Getter &getter) {
    PADDLE_ENFORCE_EQ(
        HasGetterMethod(name), false,
        platform::errors::AlreadyExists(
            "Getter of global variable %s has been registered", name));
    PADDLE_ENFORCE_NOT_NULL(getter,
                            platform::errors::InvalidArgument(
                                "Getter of %s should not be null", name));
    var_infos_.insert({name, VarInfo(is_public, getter)});
  }

  void Register(const std::string &name, bool is_public, const Getter &getter,
                const Setter &setter) {
    PADDLE_ENFORCE_EQ(
        HasGetterMethod(name), false,
        platform::errors::AlreadyExists(
            "Getter of global variable %s has been registered", name));

    PADDLE_ENFORCE_EQ(
        HasSetterMethod(name), false,
        platform::errors::AlreadyExists(
            "Setter of global variable %s has been registered", name));

    PADDLE_ENFORCE_NOT_NULL(getter,
                            platform::errors::InvalidArgument(
                                "Getter of %s should not be null", name));

    PADDLE_ENFORCE_NOT_NULL(setter,
                            platform::errors::InvalidArgument(
                                "Setter of %s should not be null", name));

    var_infos_.insert({name, VarInfo(is_public, getter, setter)});
  }

  const Getter &GetterMethod(const std::string &name) const {
    PADDLE_ENFORCE_EQ(
        HasGetterMethod(name), true,
        platform::errors::NotFound("Cannot find global variable %s", name));
    return var_infos_.at(name).getter;
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
    return var_infos_.at(name).setter;
  }

  void Set(const std::string &name, const py::object &value) const {
    SetterMethod(name)(value);
  }

  bool HasGetterMethod(const std::string &name) const {
    return var_infos_.count(name) > 0;
  }

  bool HasSetterMethod(const std::string &name) const {
    return var_infos_.count(name) > 0 && var_infos_.at(name).setter;
  }

  bool IsPublic(const std::string &name) const {
    return var_infos_.count(name) > 0 && var_infos_.at(name).is_public;
  }

  std::unordered_set<std::string> Keys() const {
    std::unordered_set<std::string> keys;
    keys.reserve(var_infos_.size());
    for (auto &pair : var_infos_) {
      keys.insert(pair.first);
    }
    return keys;
  }

 private:
  static GlobalVarGetterSetterRegistry instance_;

  std::unordered_map<std::string, VarInfo> var_infos_;
};

GlobalVarGetterSetterRegistry GlobalVarGetterSetterRegistry::instance_;

class GlobalVarGetterSetterRegistryHelper {
 public:
  GlobalVarGetterSetterRegistryHelper(bool is_public, bool is_writable,
                                      const std::string &var_names)
      : is_public_(is_public),
        is_writable_(is_writable),
        var_names_(SplitVarNames(var_names)) {}

  template <typename... Args>
  void Register(Args &&... args) const {
    Impl<0, sizeof...(args) == 1, Args...>::Register(
        is_public_, is_writable_, var_names_, std::forward<Args>(args)...);
  }

 private:
  static std::vector<std::string> SplitVarNames(const std::string &names) {
    auto valid_char = [](char ch) { return !std::isspace(ch) && ch != ','; };

    std::vector<std::string> ret;
    size_t i = 0, j = 0, n = names.size();
    while (i < n) {
      for (; i < n && !valid_char(names[i]); ++i) {
      }
      for (j = i + 1; j < n && valid_char(names[j]); ++j) {
      }

      if (i < n && j <= n) {
        auto substring = names.substr(i, j - i);
        VLOG(10) << "Get substring: \"" << substring << "\"";
        ret.emplace_back(substring);
      }
      i = j + 1;
    }
    return ret;
  }

 private:
  template <size_t kIdx, bool kIsStop, typename T, typename... Args>
  struct Impl {
    static void Register(bool is_public, bool is_writable,
                         const std::vector<std::string> &var_names, T &&var,
                         Args &&... args) {
      PADDLE_ENFORCE_EQ(kIdx + 1 + sizeof...(args), var_names.size(),
                        platform::errors::InvalidArgument(
                            "Argument number not match name number"));
      Impl<kIdx, true, T>::Register(is_public, is_writable, var_names, var);
      Impl<kIdx + 1, sizeof...(Args) == 1, Args...>::Register(
          is_public, is_writable, var_names, std::forward<Args>(args)...);
    }
  };

  template <size_t kIdx, typename T>
  struct Impl<kIdx, true, T> {
    static void Register(bool is_public, bool is_writable,
                         const std::vector<std::string> &var_names, T &&var) {
      auto *instance = GlobalVarGetterSetterRegistry::MutableInstance();
      if (is_writable) {
        instance->Register(
            var_names[kIdx], is_public,
            GlobalVarGetterSetterRegistry::CreateGetter(std::forward<T>(var)),
            GlobalVarGetterSetterRegistry::CreateSetter(&var));
      } else {
        instance->Register(
            var_names[kIdx], is_public,
            GlobalVarGetterSetterRegistry::CreateGetter(std::forward<T>(var)));
      }
    }
  };

 private:
  const bool is_public_;
  const bool is_writable_;
  const std::vector<std::string> var_names_;
};

static void RegisterGlobalVarGetterSetter();

void BindGlobalValueGetterSetter(pybind11::module *module) {
  RegisterGlobalVarGetterSetter();

  py::class_<GlobalVarGetterSetterRegistry>(*module,
                                            "GlobalVarGetterSetterRegistry")
      .def("__getitem__", &GlobalVarGetterSetterRegistry::Get)
      .def("__setitem__", &GlobalVarGetterSetterRegistry::Set)
      .def("__contains__", &GlobalVarGetterSetterRegistry::HasGetterMethod)
      .def("keys", &GlobalVarGetterSetterRegistry::Keys)
      .def("is_public", &GlobalVarGetterSetterRegistry::IsPublic)
      .def("get", &GlobalVarGetterSetterRegistry::GetOrReturnDefaultValue,
           py::arg("key"), py::arg("default") = py::cast<py::none>(Py_None));

  module->def("globals", &GlobalVarGetterSetterRegistry::Instance,
              py::return_value_policy::reference);
}

/* Public vars are designed to be writable. */
#define REGISTER_PUBLIC_GLOBAL_VAR(...)                                        \
  do {                                                                         \
    GlobalVarGetterSetterRegistryHelper(/*is_public=*/true,                    \
                                        /*is_writable=*/true, "" #__VA_ARGS__) \
        .Register(__VA_ARGS__);                                                \
  } while (0)

#define REGISTER_PRIVATE_GLOBAL_VAR(is_writable, ...)                     \
  do {                                                                    \
    GlobalVarGetterSetterRegistryHelper(/*is_public=*/false, is_writable, \
                                        "" #__VA_ARGS__)                  \
        .Register(__VA_ARGS__);                                           \
  } while (0)

static void RegisterGlobalVarGetterSetter() {
  REGISTER_PRIVATE_GLOBAL_VAR(/*is_writable=*/false, FLAGS_use_mkldnn,
                              FLAGS_free_idle_chunk,
                              FLAGS_free_when_no_cache_hit);

  REGISTER_PUBLIC_GLOBAL_VAR(
      FLAGS_eager_delete_tensor_gb, FLAGS_enable_parallel_graph,
      FLAGS_allocator_strategy, FLAGS_use_system_allocator);

#ifdef PADDLE_WITH_CUDA
  REGISTER_PUBLIC_GLOBAL_VAR(FLAGS_gpu_memory_limit_mb,
                             FLAGS_cudnn_deterministic);
#endif
}

}  // namespace pybind
}  // namespace paddle
