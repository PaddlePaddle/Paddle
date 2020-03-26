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
DECLARE_bool(check_nan_inf);
DECLARE_bool(cpu_deterministic);
DECLARE_bool(enable_rpc_profiler);
DECLARE_int32(multiple_of_cupti_buffer_size);
DECLARE_bool(reader_queue_speed_test_mode);
DECLARE_bool(enable_parallel_graph);
DECLARE_string(pe_profile_fname);
DECLARE_string(print_sub_graph_dir);
DECLARE_double(fraction_of_cpu_memory_to_use);
DECLARE_int32(fuse_parameter_groups_size);
DECLARE_double(fuse_parameter_memory_size);
DECLARE_bool(init_allocated_mem);
DECLARE_uint64(initial_cpu_memory_in_mb);
DECLARE_double(memory_fraction_of_eager_deletion);
DECLARE_bool(use_pinned_memory);
DECLARE_bool(benchmark);
DECLARE_int32(inner_op_parallelism);
DECLARE_string(tracer_profile_fname);
DECLARE_string(allocator_strategy);
#ifdef PADDLE_WITH_CUDA
// DECLARE_int32(rpc_send_thread_num);
// DECLARE_int32(rpc_get_thread_num);
// DECLARE_int32(rpc_prefetch_thread_num);
DECLARE_bool(sync_nccl_allreduce);
DECLARE_string(selected_gpus);
DECLARE_uint64(gpu_memory_limit_mb);
DECLARE_bool(cudnn_deterministic);
DECLARE_uint64(conv_workspace_size_limit);
DECLARE_bool(cudnn_batchnorm_spatial_persistent);
DECLARE_bool(cudnn_exhaustive_search);
DECLARE_bool(eager_delete_scope);
DECLARE_bool(fast_eager_deletion_mode);
DECLARE_double(fraction_of_cuda_pinned_memory_to_use);
DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
// DECLARE_int64(reallocate_gpu_memory_in_mb);
DECLARE_bool(enable_cublas_tensor_op_math);
#endif

namespace paddle {
namespace pybind {

namespace py = pybind11;

class PYBIND11_HIDDEN GlobalVarGetterSetterRegistry {
  DISABLE_COPY_AND_ASSIGN(GlobalVarGetterSetterRegistry);

  GlobalVarGetterSetterRegistry() = default;

 public:
  using Setter = std::function<void(const py::object &)>;
  using Getter = std::function<py::object()>;

  struct Flag_Getter {
    bool flag;
    Getter getter;
  };

  struct Flag_Setter {
    bool flag;
    Setter setter;
  };

  static const GlobalVarGetterSetterRegistry &Instance() { return instance_; }

  static GlobalVarGetterSetterRegistry *MutableInstance() { return &instance_; }

  void RegisterGetter(const std::string &name, const bool flag, Getter func) {
    PADDLE_ENFORCE_EQ(
        getters_.count(name), 0,
        platform::errors::AlreadyExists(
            "Getter of global variable %s has been registered", name));
    PADDLE_ENFORCE_NOT_NULL(func, platform::errors::InvalidArgument(
                                      "Getter of %s should not be null", name));
    Flag_Getter temp;
    temp.flag = flag;
    temp.getter = func;
    getters_[name] = std::move(temp);
  }

  void RegisterSetter(const std::string &name, const bool flag, Setter func) {
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
    Flag_Setter temp;
    temp.flag = flag;
    temp.setter = func;
    setters_[name] = std::move(temp);
  }

  const Flag_Getter &GetterMethod(const std::string &name) const {
    PADDLE_ENFORCE_EQ(
        HasGetterMethod(name), true,
        platform::errors::NotFound("Cannot find global variable %s", name));
    return getters_.at(name);
  }

  py::object GetPublicOrReturnDefaultValue(
      const std::string &name, const py::object &default_value) const {
    if (HasGetterMethod(name)) {
      PADDLE_ENFORCE_EQ(
          GetterMethod(name).flag, true,
          platform::errors::NotFound(
              "Flag %s is public, cann't get it's value through this function.",
              name));
      return GetterMethod(name).getter();
    } else {
      return default_value;
    }
  }

  py::object GetPrivateOrReturnDefaultValue(
      const std::string &name, const py::object &default_value) const {
    if (HasGetterMethod(name)) {
      PADDLE_ENFORCE_EQ(
          GetterMethod(name).flag, false,
          platform::errors::NotFound("Flag %s is private, cann't get it's "
                                     "value through this function.",
                                     name));
      return GetterMethod(name).getter();
    } else {
      return default_value;
    }
  }

  py::object GetPublic(const std::string &name) const {
    PADDLE_ENFORCE_EQ(
        GetterMethod(name).flag, true,
        platform::errors::NotFound(
            "Flag %s is private, cann't get it's value through this function.",
            name));
    return GetterMethod(name).getter();
  }

  py::object GetPrivate(const std::string &name) const {
    PADDLE_ENFORCE_EQ(
        GetterMethod(name).flag, false,
        platform::errors::NotFound(
            "Flag %s is public, cann't get it's value through this function.",
            name));
    return GetterMethod(name).getter();
  }

  const Flag_Setter &SetterMethod(const std::string &name) const {
    PADDLE_ENFORCE_EQ(
        HasSetterMethod(name), true,
        platform::errors::NotFound("Global variable %s is not writable", name));
    return setters_.at(name);
  }

  void SetPublic(const std::string &name, const py::object &value) const {
    PADDLE_ENFORCE_EQ(
        SetterMethod(name).flag, true,
        platform::errors::NotFound(
            "Flag %s is private, cann't set it's value through this function.",
            name));
    SetterMethod(name).setter(value);
  }

  void SetPrivate(const std::string &name, const py::object &value) const {
    PADDLE_ENFORCE_EQ(
        SetterMethod(name).flag, false,
        platform::errors::NotFound(
            "Flag %s is public, cann't set it's value through this function.",
            name));
    SetterMethod(name).setter(value);
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

  std::unordered_map<std::string, Flag_Getter> getters_;
  std::unordered_map<std::string, Flag_Setter> setters_;
};

GlobalVarGetterSetterRegistry GlobalVarGetterSetterRegistry::instance_;

static void RegisterGlobalVarGetterSetter();

void BindGlobalValueGetterSetter(pybind11::module *module) {
  RegisterGlobalVarGetterSetter();

  py::class_<GlobalVarGetterSetterRegistry>(*module,
                                            "GlobalVarGetterSetterRegistry")
      .def("__getpublic__", &GlobalVarGetterSetterRegistry::GetPublic)
      .def("__getprivate__", &GlobalVarGetterSetterRegistry::GetPrivate)
      .def("__setpublic__", &GlobalVarGetterSetterRegistry::SetPublic)
      .def("__setprivate__", &GlobalVarGetterSetterRegistry::SetPrivate)
      .def("__contains__", &GlobalVarGetterSetterRegistry::HasGetterMethod)
      .def("keys", &GlobalVarGetterSetterRegistry::Keys)
      .def("getpublic",
           &GlobalVarGetterSetterRegistry::GetPublicOrReturnDefaultValue,
           py::arg("key"), py::arg("default") = py::cast<py::none>(Py_None))
      .def("getprivate",
           &GlobalVarGetterSetterRegistry::GetPrivateOrReturnDefaultValue,
           py::arg("key"), py::arg("default") = py::cast<py::none>(Py_None));

  module->def("globals", &GlobalVarGetterSetterRegistry::Instance,
              py::return_value_policy::reference);
}

#define REGISTER_PRIVATE_GLOBAL_VAR_GETTER_ONLY(var, flag)          \
  GlobalVarGetterSetterRegistry::MutableInstance()->RegisterGetter( \
      #var, flag, []() -> py::object { return py::cast(var); })

#define REGISTER_PRIVATE_GLOBAL_VAR_SETTER_ONLY(var, flag)            \
  GlobalVarGetterSetterRegistry::MutableInstance()->RegisterSetter(   \
      #var, flag, [](const py::object &obj) {                         \
        using ValueType = std::remove_reference<decltype(var)>::type; \
        var = py::cast<ValueType>(obj);                               \
      })
#define REGISTER_PRIVATE_GLOBAL_VAR_GETTER_SETTER(var, flag) \
  REGISTER_PRIVATE_GLOBAL_VAR_GETTER_ONLY(var, flag);        \
  REGISTER_PRIVATE_GLOBAL_VAR_SETTER_ONLY(var, flag)

#define REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(var, flag) \
  REGISTER_PRIVATE_GLOBAL_VAR_GETTER_ONLY(var, flag);       \
  REGISTER_PRIVATE_GLOBAL_VAR_SETTER_ONLY(var, flag)

static void RegisterGlobalVarGetterSetter() {
  REGISTER_PRIVATE_GLOBAL_VAR_GETTER_ONLY(FLAGS_use_mkldnn, false);
  REGISTER_PRIVATE_GLOBAL_VAR_GETTER_ONLY(FLAGS_use_ngraph, false);
  REGISTER_PRIVATE_GLOBAL_VAR_GETTER_ONLY(FLAGS_free_idle_chunk, false);
  REGISTER_PRIVATE_GLOBAL_VAR_GETTER_ONLY(FLAGS_free_when_no_cache_hit, false);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_eager_delete_tensor_gb, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_use_system_allocator, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_allocator_strategy, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_check_nan_inf, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_cpu_deterministic, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_enable_rpc_profiler, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_multiple_of_cupti_buffer_size,
                                           true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_reader_queue_speed_test_mode,
                                           true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_enable_parallel_graph, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_pe_profile_fname, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_print_sub_graph_dir, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_fraction_of_cpu_memory_to_use,
                                           true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_fuse_parameter_groups_size,
                                           true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_fuse_parameter_memory_size,
                                           true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_init_allocated_mem, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_initial_cpu_memory_in_mb,
                                           true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(
      FLAGS_memory_fraction_of_eager_deletion, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_use_pinned_memory, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_benchmark, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_inner_op_parallelism, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_tracer_profile_fname, true);
#ifdef PADDLE_WITH_CUDA
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_gpu_memory_limit_mb, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_cudnn_deterministic, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_conv_workspace_size_limit,
                                           true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(
      FLAGS_cudnn_batchnorm_spatial_persistent, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_cudnn_exhaustive_search, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_eager_delete_scope, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_fast_eager_deletion_mode,
                                           true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(
      FLAGS_fraction_of_cuda_pinned_memory_to_use, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_fraction_of_gpu_memory_to_use,
                                           true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_initial_gpu_memory_in_mb,
                                           true);
  // REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_reallocate_gpu_memory_in_mb,
  // true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_enable_cublas_tensor_op_math,
                                           true);

  // REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_rpc_send_thread_num, true);
  // REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_rpc_get_thread_num, true);
  // REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_rpc_prefetch_thread_num,
  // true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_selected_gpus, true);
  REGISTER_PUBLIC_GLOBAL_VAR_GETTER_SETTER(FLAGS_sync_nccl_allreduce, true);
#endif
}

}  // namespace pybind
}  // namespace paddle
