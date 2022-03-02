/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <Python.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>  // NOLINT // for call_once
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/ir/coalesce_grad_tensor_pass.h"
#include "paddle/fluid/framework/ir/cost_model.h"
#include "paddle/fluid/framework/ir/generate_pass.h"
#include "paddle/fluid/framework/ir/pass_builder.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/prune.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/save_load_util.h"
#include "paddle/fluid/framework/scope_pool.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/operators/py_func_op.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/monitor.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/pybind/cuda_streams_py.h"
#include "paddle/fluid/pybind/distributed_py.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/io.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/lod_utils.h"
#include "paddle/utils/none.h"
#ifdef PADDLE_WITH_ASCEND
#include "paddle/fluid/pybind/ascend_wrapper_py.h"
#endif
#include "paddle/fluid/pybind/bind_cost_model.h"
#include "paddle/fluid/pybind/bind_fleet_executor.h"
#include "paddle/fluid/pybind/box_helper_py.h"
#include "paddle/fluid/pybind/communication.h"
#include "paddle/fluid/pybind/compatible.h"
#include "paddle/fluid/pybind/const_value.h"
#include "paddle/fluid/pybind/data_set_py.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/fleet_wrapper_py.h"
#include "paddle/fluid/pybind/generator_py.h"
#include "paddle/fluid/pybind/global_value_getter_setter.h"
#include "paddle/fluid/pybind/gloo_context_py.h"
#include "paddle/fluid/pybind/gloo_wrapper_py.h"
#include "paddle/fluid/pybind/heter_wrapper_py.h"
#include "paddle/fluid/pybind/imperative.h"
#include "paddle/fluid/pybind/inference_api.h"
#include "paddle/fluid/pybind/ir.h"
#include "paddle/fluid/pybind/metrics_py.h"
#include "paddle/fluid/pybind/ps_gpu_wrapper_py.h"
#include "paddle/fluid/pybind/pybind_boost_headers.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/pybind/nccl_wrapper_py.h"
#endif
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/pybind/protobuf.h"
#include "paddle/fluid/pybind/pybind.h"  // NOLINT
#include "paddle/fluid/pybind/reader_py.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/fluid/string/to_string.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#endif
#ifndef PADDLE_WITH_HIP
#include "paddle/fluid/platform/device/gpu/cuda/cuda_profiler.h"
#endif
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#endif

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/npu/npu_info.h"
#include "paddle/fluid/platform/device/npu/npu_profiler.h"
#endif

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/xpu_op_list.h"
#endif

#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"
#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#endif

#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/mlu_info.h"
#endif

#ifdef PADDLE_WITH_CRYPTO
#include "paddle/fluid/pybind/crypto.h"
#endif

#if defined PADDLE_WITH_PSCORE
#include "paddle/fluid/pybind/fleet_py.h"
#endif

#include "pybind11/stl.h"

DECLARE_bool(use_mkldnn);

// disable auto conversion to list in Python
PYBIND11_MAKE_OPAQUE(paddle::framework::LoDTensorArray);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchUnmergedList);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchList);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchType);

namespace paddle {
namespace pybind {

PyTypeObject *g_place_pytype = nullptr;
PyTypeObject *g_cudaplace_pytype = nullptr;
PyTypeObject *g_cpuplace_pytype = nullptr;
PyTypeObject *g_xpuplace_pytype = nullptr;
PyTypeObject *g_npuplace_pytype = nullptr;
PyTypeObject *g_cudapinnedplace_pytype = nullptr;
PyTypeObject *g_mluplace_pytype = nullptr;
PyTypeObject *g_framework_tensor_pytype = nullptr;
PyTypeObject *g_framework_lodtensorarray_pytype = nullptr;

bool IsCompiledWithCUDA() {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithNCCL() {
#ifdef PADDLE_WITH_NCCL
  return true;
#else
  return false;
#endif
}

bool IsCompiledWithROCM() {
#ifndef PADDLE_WITH_HIP
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithAscend() {
#ifndef PADDLE_WITH_ASCEND
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithXPU() {
#ifndef PADDLE_WITH_XPU
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithNPU() {
#ifndef PADDLE_WITH_ASCEND_CL
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithIPU() {
#ifndef PADDLE_WITH_IPU
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithMKLDNN() {
#ifndef PADDLE_WITH_MKLDNN
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithCINN() {
#ifndef PADDLE_WITH_CINN
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithMLU() {
#ifndef PADDLE_WITH_MLU
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithHETERPS() {
#ifndef PADDLE_WITH_HETERPS
  return false;
#else
  return true;
#endif
}

bool SupportsBfloat16() {
#ifndef PADDLE_WITH_MKLDNN
  return false;
#else
  if (platform::MayIUse(platform::cpu_isa_t::avx512_core))
    return true;
  else
    return false;
#endif
}

bool SupportsBfloat16FastPerformance() {
#ifndef PADDLE_WITH_MKLDNN
  return false;
#else
  if (platform::MayIUse(platform::cpu_isa_t::avx512_bf16))
    return true;
  else
    return false;
#endif
}

bool SupportsInt8() {
#ifndef PADDLE_WITH_MKLDNN
  return false;
#else
  return (platform::MayIUse(platform::cpu_isa_t::avx2) ||
          platform::MayIUse(platform::cpu_isa_t::avx512f));
#endif
}

bool SupportsVNNI() {
#ifndef PADDLE_WITH_MKLDNN
  return false;
#else
  return platform::MayIUse(platform::cpu_isa_t::avx512_core_vnni);
#endif
}

bool IsCompiledWithBrpc() {
#ifndef PADDLE_WITH_DISTRIBUTE
  return false;
#endif
  return true;
}

bool IsCompiledWithDIST() {
#ifdef PADDLE_WITH_DISTRIBUTE
  return true;
#else
  return false;
#endif
}

template <typename PlaceType1, typename PlaceType2>
static inline bool IsSamePlace(const PlaceType1 &p1, const PlaceType2 &p2) {
  return paddle::platform::Place(p1) == paddle::platform::Place(p2);
}

template <typename PlaceType>
static inline int PlaceIndex(const PlaceType &p) {
  return static_cast<int>(paddle::platform::Place(p).GetType());
}

static PyObject *GetPythonAttribute(PyObject *obj, const char *attr_name) {
  // NOTE(zjl): PyObject_GetAttrString would return nullptr when attr_name
  // is not inside obj, but it would also set the error flag of Python.
  // If the error flag is set in C++, C++ code would not raise Exception,
  // but Python would raise Exception once C++ call ends.
  // To avoid unexpected Exception raised in Python, we check whether
  // attribute exists before calling PyObject_GetAttrString.
  //
  // Caution: PyObject_GetAttrString would increase reference count of PyObject.
  // Developer should call Py_DECREF manually after the attribute is not used.
  if (PyObject_HasAttrString(obj, attr_name)) {
    return PyObject_GetAttrString(obj, attr_name);
  } else {
    return nullptr;
  }
}

template <typename T>
static T PyObjectCast(PyObject *obj) {
  try {
    return py::cast<T>(py::handle(obj));
  } catch (py::cast_error &) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Python object is not type of %s, the real type is %s",
        typeid(T).name(), obj->ob_type->tp_name));
  }
}

using PyNameVarBaseMap = std::unordered_map<std::string, py::handle>;

static std::vector<std::shared_ptr<imperative::VarBase>> GetVarBaseList(
    const PyNameVarBaseMap &state_dict) {
  std::vector<std::shared_ptr<imperative::VarBase>> vec_res;
  vec_res.reserve(state_dict.size());

  for (auto &para : state_dict) {
    PyObject *py_obj = para.second.ptr();
    if (!py_obj || py_obj == Py_None) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The parameter [%s] to save is None", para.first));
    }
    vec_res.emplace_back(
        PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_obj));
  }

  return vec_res;
}

static std::vector<std::string> inline GetNameList(
    const py::handle &py_handle) {
  std::vector<std::string> vec_res;

  PyObject *py_obj = py_handle.ptr();  // get underlying PyObject
  // Python None is not nullptr in C++!
  if (!py_obj || py_obj == Py_None) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The parameter list to save is None"));
  }

  if (PyList_Check(py_obj)) {
    size_t len = PyList_GET_SIZE(py_obj);

    vec_res.reserve(len);

    const char *kNameField = "name";

    for (size_t i = 0; i < len; ++i) {
      PyObject *py_name =
          PyObject_GetAttrString(PyList_GET_ITEM(py_obj, i), kNameField);
      PADDLE_ENFORCE_NOT_NULL(py_name,
                              platform::errors::InvalidArgument(
                                  "The name of parameter to save is None"));
      vec_res.emplace_back(PyObjectCast<std::string>(py_name));
      Py_DECREF(py_name);
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The parameters to save is not a list"));
  }
  return vec_res;
}

static void inline CreateVariableIfNotExit(
    const py::handle &py_handle, const framework::Scope &scope,
    const framework::Executor *exe = nullptr) {
  std::vector<std::string> vec_res;

  PyObject *py_obj = py_handle.ptr();  // get underlying PyObject
  // Python None is not nullptr in C++!
  if (!py_obj || py_obj == Py_None) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("The parameter list to set is None"));
  }

  if (PyList_Check(py_obj)) {
    size_t len = PyList_GET_SIZE(py_obj);

    vec_res.reserve(len);

    const char *kNameField = "name";
    const char *kVarDescField = "desc";

    for (size_t i = 0; i < len; ++i) {
      PyObject *py_name =
          PyObject_GetAttrString(PyList_GET_ITEM(py_obj, i), kNameField);
      PADDLE_ENFORCE_NOT_NULL(py_name,
                              platform::errors::InvalidArgument(
                                  "The name of parameter to set is None"));
      auto para_name = PyObjectCast<std::string>(py_name);
      Py_DECREF(py_name);

      auto var = scope.FindVar(para_name);
      if (var == nullptr) {
        PADDLE_ENFORCE_NOT_NULL(exe,
                                platform::errors::InvalidArgument(
                                    "Parameter not Initialized, "
                                    "Please set argument [executor] not None "
                                    "or run startup program first"));
        PyObject *py_var_desc =
            PyObject_GetAttrString(PyList_GET_ITEM(py_obj, i), kVarDescField);
        PADDLE_ENFORCE_NOT_NULL(
            py_var_desc, platform::errors::InvalidArgument(
                             "The var_desc of parameter to set is None"));
        auto var_desc = PyObjectCast<framework::VarDesc>(py_var_desc);
        Py_DECREF(py_var_desc);
        var = const_cast<framework::Scope *>(&scope)->Var(para_name);
        auto *tensor_temp = var->GetMutable<framework::LoDTensor>();
        tensor_temp->Resize(phi::make_ddim(var_desc.GetShape()));
        tensor_temp->mutable_data(
            exe->GetPlace(),
            framework::TransToPhiDataType(var_desc.GetDataType()));
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The parameters to set is not a list"));
  }

  return;
}

static void AssertStaticGraphAndDygraphGradMakerNoDiff() {
  std::set<std::string> ops;
  for (auto &pair : framework::OpInfoMap::Instance().map()) {
    bool has_static_grad_maker = (pair.second.grad_op_maker_ != nullptr);
    bool has_dygraph_grad_maker =
        (pair.second.dygraph_grad_op_maker_ != nullptr);
    if (has_static_grad_maker ^ has_dygraph_grad_maker) {
      bool has_kernel =
          (framework::OperatorWithKernel::AllOpKernels().count(pair.first) > 0);
      if (has_kernel) {
        ops.insert(pair.first);
      } else {
        VLOG(5) << pair.first << " has no kernels, skip";
      }
    }
  }
  PADDLE_ENFORCE_EQ(ops.empty(), true,
                    platform::errors::Unimplemented(
                        "OperatorWithKernel [%s] have only static graph grad "
                        "maker or have only dygraph grad maker, which is not "
                        "allowed",
                        string::join_strings(ops, ',')));
}

#ifdef PADDLE_WITH_NCCL
static int GetNCCLVersion() {
#if NCCL_VERSION_CODE >= 2304
  int ver;
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGetVersion(&ver));
  return ver;
#else
  PADDLE_THROW(platform::errors::External(
      "Cannot get NCCL version successfully when nccl version < 2.3.4"));
#endif
}
#endif

template <typename PlaceType>
static void TensorCopyFrom(framework::Tensor *dst, const framework::Tensor &src,
                           const PlaceType &place, int64_t batch_size) {
  if (batch_size < 0) {
    framework::TensorCopy(src, place, dst);
  } else {
    auto sliced = src.Slice(0, batch_size);
    framework::TensorCopy(sliced, place, dst);
  }
}

#ifdef PADDLE_WITH_AVX
PYBIND11_MODULE(core_avx, m) {
#else
PYBIND11_MODULE(core_noavx, m) {
#endif

  BindEager(&m);
  BindCudaStream(&m);

  // Not used, just make sure cpu_info.cc is linked.
  paddle::platform::CpuTotalPhysicalMemory();

  paddle::memory::allocation::UseAllocatorStrategyGFlag();

  AssertStaticGraphAndDygraphGradMakerNoDiff();

  m.doc() = "C++ core of PaddlePaddle";

  // using framework in this function. Since it is inside a function, it will
  // not cause namespace pollution.
  using namespace paddle::framework;  // NOLINT

  BindException(&m);

  m.def("set_num_threads", &platform::SetNumThreads);

  m.def("disable_signal_handler", &DisableSignalHandler);

  m.def("clear_gradients",
        [](std::vector<std::shared_ptr<imperative::VarBase>> param_list,
           bool set_to_zero) {
          for (auto param : param_list) {
            param->ClearGradient(set_to_zero);
          }
        });

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  m.def("cudnn_version", &platform::DnnVersion);
  m.def("gpu_memory_available", []() {
    size_t available = 0;
    size_t total = 0;
    paddle::platform::GpuMemoryUsage(&available, &total);
    return available;
  });
#endif

#ifdef PADDLE_WITH_NCCL
  m.def("nccl_version", &GetNCCLVersion);
#endif

  m.def("is_cuda_graph_capturing", &platform::IsCUDAGraphCapturing);
#ifdef PADDLE_WITH_CUDA
  py::class_<platform::CUDAGraph>(m, "CUDAGraph")
      .def_static("begin_capture",
                  [](platform::CUDAPlace place, int mode) {
                    platform::BeginCUDAGraphCapture(
                        place, static_cast<cudaStreamCaptureMode>(mode));
                  })
      .def_static("end_capture", &platform::EndCUDAGraphCapture)
      .def("replay", &platform::CUDAGraph::Replay)
      .def("reset", &platform::CUDAGraph::Reset)
      .def("print_to_dot_files", &platform::CUDAGraph::PrintToDotFiles);
#endif

  m.def("wait_device", [](const platform::Place &place) {
    platform::DeviceContextPool::Instance().Get(place)->Wait();
  });

  m.def("from_dlpack", [](py::capsule *dltensor) {
    DLManagedTensor *dmt = reinterpret_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(dltensor->ptr(), "dltensor"));

    PADDLE_ENFORCE_NOT_NULL(
        dmt, platform::errors::InvalidArgument(
                 "from_dlpack received an invalid capsule. "
                 "Note that a DLPack tensor can be consumed only once."));

    PyCapsule_SetName(dltensor->ptr(), "used_dltensor");
    DLTensor dl = dmt->dl_tensor;
    framework::Tensor tensor;

    if (dl.device.device_type == kDLCPU) {
      paddle::framework::TensorFromDLPack(dl, &tensor);
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (dl.device.device_type == kDLGPU) {
      paddle::framework::TensorFromDLPack(dl, &tensor);
    }
#endif
    return tensor;
  });

  m.def("_create_loaded_parameter",
        [](const py::handle &vec_var_list, const Scope &scope,
           const Executor *executor) {
          CreateVariableIfNotExit(vec_var_list, scope, executor);
        });

  m.def("save_op_version_info", [](framework::ProgramDesc &desc) {
    framework::compatible::pb::OpVersionMap pb_vmap{desc.OpVersionMap()};
    framework::compatible::SaveOpVersions(
        framework::compatible::OpVersionRegistrar::GetInstance()
            .GetVersionMap(),
        &pb_vmap);
  });

  m.def("set_printoptions", [](const py::kwargs &kwargs) {
    auto &print_opt = framework::PrintOptions::Instance();
    if (kwargs.contains("precision")) {
      print_opt.precision = kwargs["precision"].cast<int>();
    }
    if (kwargs.contains("threshold")) {
      print_opt.threshold = kwargs["threshold"].cast<int>();
    }
    if (kwargs.contains("edgeitems")) {
      print_opt.edgeitems = kwargs["edgeitems"].cast<int>();
    }
    if (kwargs.contains("linewidth")) {
      print_opt.linewidth = kwargs["linewidth"].cast<int>();
    }
    if (kwargs.contains("sci_mode")) {
      print_opt.sci_mode = kwargs["sci_mode"].cast<bool>();
    }

    VLOG(4) << "Set printoptions: precision=" << print_opt.precision
            << ", threshold=" << print_opt.threshold
            << ", edgeitems=" << print_opt.edgeitems
            << ", linewidth=" << print_opt.linewidth
            << ", sci_mode=" << print_opt.sci_mode;
  });

  m.def("broadcast_shape", [](const std::vector<int64_t> &x_dim,
                              const std::vector<int64_t> &y_dim) {
    return phi::vectorize(operators::details::BroadcastTwoDims(
        phi::make_ddim(x_dim), phi::make_ddim(y_dim), -1));
  });

  m.def(
      "_append_python_callable_object_and_return_id",
      [](py::object py_obj) -> size_t {
        return paddle::operators::AppendPythonCallableObjectAndReturnId(py_obj);
      });

  m.def("_get_use_default_grad_op_desc_maker_ops",
        [] { return OpInfoMap::Instance().GetUseDefaultGradOpDescMakerOps(); });

  m.def("_get_all_register_op_kernels",
        [](const std::string &lib) {
          std::unordered_map<std::string, std::vector<std::string>>
              all_kernels_info;
          if (lib == "fluid" || lib == "all") {
            auto &all_kernels =
                paddle::framework::OperatorWithKernel::AllOpKernels();

            for (auto &kernel_pair : all_kernels) {
              auto op_type = kernel_pair.first;
              std::vector<std::string> kernel_types;
              for (auto &info_pair : kernel_pair.second) {
                paddle::framework::OpKernelType kernel_type = info_pair.first;
                kernel_types.emplace_back(
                    paddle::framework::KernelTypeToString(kernel_type));
              }
              all_kernels_info.emplace(op_type, kernel_types);
            }
          }
          if (lib == "phi" || lib == "all") {
            auto phi_kernels = phi::KernelFactory::Instance().kernels();
            for (auto &kernel_pair : phi_kernels) {
              auto op_type = phi::TransToFluidOpName(kernel_pair.first);
              std::vector<std::string> kernel_types;
              for (auto &info_pair : kernel_pair.second) {
                framework::OpKernelType kernel_type =
                    framework::TransPhiKernelKeyToOpKernelType(info_pair.first);
                auto kernel_type_str =
                    framework::KernelTypeToString(kernel_type);
                if (all_kernels_info.count(op_type)) {
                  if (std::find(all_kernels_info[op_type].begin(),
                                all_kernels_info[op_type].end(),
                                kernel_type_str) ==
                      all_kernels_info[op_type].end()) {
                    all_kernels_info[op_type].emplace_back(kernel_type_str);
                  }
                } else {
                  kernel_types.emplace_back(kernel_type_str);
                }
              }
              if (!kernel_types.empty()) {
                all_kernels_info.emplace(op_type, kernel_types);
              }
            }
          }

          return all_kernels_info;
        },
        py::arg("lib") = "all",
        R"DOC(
           Return the registered kernels in paddle.

           Args:
               lib[string]: the libarary, could be 'phi', 'fluid' and 'all'.
           )DOC");

  // NOTE(zjl): ctest would load environment variables at the beginning even
  // though we have not `import paddle.fluid as fluid`. So we add this API
  // to enable eager deletion mode in unittest.
  m.def("_set_eager_deletion_mode", &paddle::framework::SetEagerDeletionMode);

  m.def("_set_fuse_parameter_group_size",
        &paddle::framework::ir::SetFuseParameterGroupsSize);
  m.def("_set_fuse_parameter_memory_size",
        &paddle::framework::ir::SetFuseParameterMemorySize);

  m.add_object("_cleanup",
               py::capsule([]() { ScopePool::Instance().Clear(); }));

  m.def("_set_paddle_lib_path", &paddle::platform::dynload::SetPaddleLibPath);

  m.def("_promote_types_if_complex_exists",
        &paddle::framework::PromoteTypesIfComplexExists);

  BindImperative(&m);

  py::class_<framework::Tensor> framework_tensor(m, "Tensor",
                                                 py::buffer_protocol());
  g_framework_tensor_pytype =
      reinterpret_cast<PyTypeObject *>(framework_tensor.ptr());
  framework_tensor
      .def("__array__",
           [](framework::Tensor &self) { return TensorToPyArray(self); })
      .def("_ptr",
           [](const framework::Tensor &self) {
             return reinterpret_cast<uintptr_t>(self.data());
           })
      .def("_is_initialized",
           [](const framework::Tensor &self) { return self.IsInitialized(); })
      .def("_get_dims",
           [](const framework::Tensor &self) { return vectorize(self.dims()); })
      .def("_set_dims",
           [](framework::Tensor &self, const std::vector<int64_t> &dim) {
             self.Resize(phi::make_ddim(dim));
           })
      .def("_set_layout",
           [](framework::Tensor &self, const std::string &layout) {
             self.set_layout(StringToDataLayout(layout));
           })
      .def("_alloc_float",
           [](framework::Tensor &self, paddle::platform::CUDAPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](framework::Tensor &self, paddle::platform::XPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](framework::Tensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](framework::Tensor &self, paddle::platform::NPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](framework::Tensor &self, paddle::platform::MLUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_double",
           [](framework::Tensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<double>(place);
           })
      .def("_alloc_int",
           [](framework::Tensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](framework::Tensor &self, paddle::platform::XPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](framework::Tensor &self, paddle::platform::CUDAPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](framework::Tensor &self, paddle::platform::MLUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](framework::Tensor &self,
              paddle::platform::CUDAPinnedPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_float",
           [](framework::Tensor &self,
              paddle::platform::CUDAPinnedPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_mutable_data",
           [](framework::Tensor &self, paddle::platform::CPUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](framework::Tensor &self, paddle::platform::XPUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](framework::Tensor &self, paddle::platform::CUDAPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](framework::Tensor &self, paddle::platform::CUDAPinnedPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](framework::Tensor &self, paddle::platform::MLUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_clear", &framework::Tensor::clear)
      .def("_mutable_data",
           [](framework::Tensor &self, paddle::platform::NPUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_copy_from", &TensorCopyFrom<paddle::platform::CPUPlace>,
           py::arg("tensor"), py::arg("place"), py::arg("batch_size") = -1)
      .def("_copy_from", &TensorCopyFrom<paddle::platform::XPUPlace>,
           py::arg("tensor"), py::arg("place"), py::arg("batch_size") = -1)
      .def("_copy_from", &TensorCopyFrom<paddle::platform::CUDAPlace>,
           py::arg("tensor"), py::arg("place"), py::arg("batch_size") = -1)
      .def("_copy_from", &TensorCopyFrom<paddle::platform::NPUPlace>,
           py::arg("tensor"), py::arg("place"), py::arg("batch_size") = -1)
      .def("_copy_from", &TensorCopyFrom<paddle::platform::CUDAPinnedPlace>,
           py::arg("tensor"), py::arg("place"), py::arg("batch_size") = -1)
      .def("_copy_from", &TensorCopyFrom<paddle::platform::MLUPlace>,
           py::arg("tensor"), py::arg("place"), py::arg("batch_size") = -1)
      .def("_copy_from", &TensorCopyFrom<paddle::platform::Place>,
           py::arg("tensor"), py::arg("place"), py::arg("batch_size") = -1)
      .def("set", SetTensorFromPyArray<paddle::platform::CPUPlace>,
           py::arg("array"), py::arg("place"), py::arg("zero_copy") = false)
      .def("set", SetTensorFromPyArray<paddle::platform::XPUPlace>,
           py::arg("array"), py::arg("place"), py::arg("zero_copy") = false)
      .def("set", SetTensorFromPyArray<paddle::platform::CUDAPlace>,
           py::arg("array"), py::arg("place"), py::arg("zero_copy") = false)
      .def("set", SetTensorFromPyArray<paddle::platform::NPUPlace>,
           py::arg("array"), py::arg("place"), py::arg("zero_copy") = false)
      .def("set", SetTensorFromPyArray<paddle::platform::IPUPlace>,
           py::arg("array"), py::arg("place"), py::arg("zero_copy") = false)
      .def("set", SetTensorFromPyArray<paddle::platform::MLUPlace>,
           py::arg("array"), py::arg("place"), py::arg("zero_copy") = false)
      .def("set", SetTensorFromPyArray<paddle::platform::CUDAPinnedPlace>,
           py::arg("array"), py::arg("place"), py::arg("zero_copy") = false,
           R"DOC(
        Set the data of Tensor on place with given numpy array.
        
        Args:
          lod (numpy.ndarray): The data to set.
          place (CPUPlace|CUDAPlace|XPUPlace|IPUPlace|CUDAPinnedPlace|NPUPlace|MLUPlace): The place where the
          Tensor is to be set.
          zero_copy (bool, optional): Whether to share memory with the input numpy array.
          This parameter only works with CPUPlace. Default: False.

        Returns:
            None.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                t = fluid.Tensor()
                t.set(np.ndarray([5, 30]), fluid.CPUPlace())
          )DOC")

      .def("shape",
           [](framework::Tensor &self) { return vectorize(self.dims()); },
           R"DOC(
           Return the shape of Tensor.

           Returns:
               list[int]: The shape of Tensor.


           Examples:
               .. code-block:: python

                  import paddle.fluid as fluid
                  import numpy as np

                  t = fluid.Tensor()
                  t.set(np.ndarray([5, 30]), fluid.CPUPlace())
                  print(t.shape())  # [5, 30]
           )DOC")
      .def("_to_dlpack",
           [](framework::Tensor &self) {
             DLPackTensor dlpack_tensor(self, 1);
             DLManagedTensor *dmt = dlpack_tensor.ToDLManagedTensor();
             auto capsule = py::capsule(
                 static_cast<void *>(dmt), "dltensor", [](PyObject *ptr) {
                   if (ptr) {
                     auto dltensor = new DLManagedTensor;
                     try {
                       dltensor = reinterpret_cast<DLManagedTensor *>(
                           PyCapsule_GetPointer(ptr, "used_dltensor"));
                       return;
                     } catch (...) {
                       dltensor = reinterpret_cast<DLManagedTensor *>(
                           PyCapsule_GetPointer(ptr, "dltensor"));
                     }
                     dltensor->deleter(dltensor);
                   }
                 });
             return capsule;
           })
      .def("_set_float_element", TensorSetElement<float>)
      .def("_get_float_element", TensorGetElement<float>)
      .def("_set_double_element", TensorSetElement<double>)
      .def("_get_double_element", TensorGetElement<double>)
      .def("_place", [](framework::Tensor &self) { return self.place(); })
      .def("_dtype",
           [](framework::Tensor &self) {
             return framework::TransToProtoVarType(self.type());
           })
      .def("_layout",
           [](framework::Tensor &self) {
             return DataLayoutToString(self.layout());
           })
      .def("_share_data_with", &framework::Tensor::ShareDataWith)
      .def("__getitem__", PySliceTensor, py::return_value_policy::reference)
      .def("__str__",
           [](const framework::Tensor &self) {
             std::stringstream ostr;
             ostr << self;
             return ostr.str();
           }) /* ------ End of original Tensor ------ */
      .def(
          "__init__",
          [](framework::Tensor &instance, const std::vector<std::vector<size_t>>
                                              &recursive_sequence_lengths) {
            LoD new_lod;
            new_lod.reserve(recursive_sequence_lengths.size());
            std::copy(recursive_sequence_lengths.begin(),
                      recursive_sequence_lengths.end(),
                      std::back_inserter(new_lod));
            LoD new_offset_lod = ConvertToOffsetBasedLoD(new_lod);
            PADDLE_ENFORCE_EQ(
                CheckLoD(new_offset_lod, -1), true,
                platform::errors::InvalidArgument(
                    "The provided recursive_sequence_lengths info is "
                    "invalid, "
                    "the LoD converted by recursive_sequence_lengths is %s",
                    new_lod));
            new (&instance) framework::Tensor(new_offset_lod);
          })
      .def("__init__",
           [](framework::Tensor &instance) {
             new (&instance) framework::Tensor();
           })
      // We implement offset based LOD in C++ while we use length based with
      // Python API. So we changed set_lod to set_recursive_sequence_lengths
      // to
      // avoid misuse.
      // The discussion is here:
      // https://github.com/PaddlePaddle/Paddle/issues/10855
      .def("set_lod",
           [](framework::Tensor &self,
              const std::vector<std::vector<size_t>> &lod) {
             // the input lod is offset-based level-of-detail info
             LoD new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
             PADDLE_ENFORCE_EQ(
                 CheckLoD(new_lod, vectorize(self.dims()).front()), true,
                 platform::errors::InvalidArgument(
                     "The provided LoD is invalid, the LoD is %s", new_lod));
             self.set_lod(new_lod);
           },
           py::arg("lod"), R"DOC(
           Set LoD of the Tensor.

           Args:
               lod (list[list[int]]): The lod to set.

           Returns:
                None.

           Examples:
               .. code-block:: python

                 import paddle.fluid as fluid
                 import numpy as np

                 t = fluid.Tensor()
                 t.set(np.ndarray([5, 30]), fluid.CPUPlace())
                 t.set_lod([[0, 2, 5]])
                 print(t.lod()) # [[0, 2, 5]]
           )DOC")
      .def("set_recursive_sequence_lengths",
           [](framework::Tensor &self, const std::vector<std::vector<size_t>>
                                           &recursive_sequence_lengths) {
             // the input recursive_sequence_lengths is length-based
             // level-of-detail info
             LoD new_lod;
             new_lod.reserve(recursive_sequence_lengths.size());
             std::copy(recursive_sequence_lengths.begin(),
                       recursive_sequence_lengths.end(),
                       std::back_inserter(new_lod));
             LoD new_offset_lod = ConvertToOffsetBasedLoD(new_lod);
             PADDLE_ENFORCE_EQ(
                 CheckLoD(new_offset_lod, vectorize(self.dims()).front()), true,
                 platform::errors::InvalidArgument(
                     "The provided recursive_sequence_lengths info is "
                     "invalid, "
                     "the LoD converted by recursive_sequence_lengths is "
                     "%s",
                     new_lod));
             self.set_lod(new_offset_lod);
           },
           py::arg("recursive_sequence_lengths"), R"DOC(
           Set LoD of the Tensor according to recursive sequence lengths.

           For example, if recursive_sequence_lengths=[[2, 3]], which means
           there are two sequences with length 2 and 3 respectively, the
           corresponding lod would be [[0, 2, 2+3]], i.e., [[0, 2, 5]].

           Args:
                recursive_sequence_lengths (list[list[int]]): The recursive sequence lengths.
           
           Returns:
                None.

           Examples:
               .. code-block:: python

                 import paddle.fluid as fluid
                 import numpy as np

                 t = fluid.Tensor()
                 t.set(np.ndarray([5, 30]), fluid.CPUPlace())
                 t.set_recursive_sequence_lengths([[2, 3]])
                 print(t.recursive_sequence_lengths())  # [[2, 3]]
                 print(t.lod())  # [[0, 2, 5]]
           )DOC")
      .def("lod",
           [](framework::Tensor &self) -> std::vector<std::vector<size_t>> {
             // output the offset-based lod info
             LoD lod = self.lod();
             std::vector<std::vector<size_t>> new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
             return new_lod;
           },
           R"DOC(
           Return the LoD of the Tensor.

           Returns:
               list[list[int]]: The lod of the Tensor.
           
           Examples:
               .. code-block:: python

                 import paddle.fluid as fluid
                 import numpy as np

                 t = fluid.Tensor()
                 t.set(np.ndarray([5, 30]), fluid.CPUPlace())
                 t.set_lod([[0, 2, 5]])
                 print(t.lod()) # [[0, 2, 5]]
           )DOC")
      // Set above comments of set_lod.
      .def("recursive_sequence_lengths",
           [](framework::Tensor &self) -> std::vector<std::vector<size_t>> {
             // output the length-based lod info
             LoD lod = phi::ConvertToLengthBasedLoD(self.lod());
             std::vector<std::vector<size_t>> new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
             return new_lod;
           },
           R"DOC(
           Return the recursive sequence lengths corresponding to of the LodD 
           of the Tensor.

           Returns:
                list[list[int]]: The recursive sequence lengths.

           Examples:
               .. code-block:: python

                 import paddle.fluid as fluid
                 import numpy as np

                 t = fluid.Tensor()
                 t.set(np.ndarray([5, 30]), fluid.CPUPlace())
                 t.set_recursive_sequence_lengths([[2, 3]])
                 print(t.recursive_sequence_lengths()) # [[2, 3]]
           )DOC")
      .def("has_valid_recursive_sequence_lengths",
           [](framework::Tensor &self) -> bool {
             // Check that the lod info is valid and match the outermost
             // dimension of the Tensor data
             return CheckLoD(self.lod(), vectorize(self.dims()).front());
           },
           R"DOC(
           Check whether the LoD of the Tensor is valid.

           Returns:
               bool: Whether the LoD is valid.

           Examples:
               .. code-block:: python

                 import paddle.fluid as fluid
                 import numpy as np

                 t = fluid.Tensor()
                 t.set(np.ndarray([5, 30]), fluid.CPUPlace())
                 t.set_recursive_sequence_lengths([[2, 3]])
                 print(t.has_valid_recursive_sequence_lengths()) # True
           )DOC")
      .def("_as_type",
           [](const framework::Tensor &self,
              paddle::framework::proto::VarType::Type type) {
             framework::Tensor dst;
             if (self.IsInitialized() && self.numel() > 0) {
               TransDataType(self, type, &dst);
             }
             return dst;
           })
      .def("_copy",
           [](const framework::Tensor &self, const platform::Place &place) {
             // follow fetch_op's inplementation
             framework::Tensor dst;
             if (self.IsInitialized() && self.numel() > 0) {
               TensorCopySync(self, place, &dst);
             } else {
               // Not copy, if the src tensor is empty.
               dst.clear();
               dst.Resize({0});
             }
             dst.set_lod(self.lod());
             return dst;
#ifdef _WIN32
           });
#else
           })
      .def(py::pickle(
          [](const framework::Tensor &t) {  // __getstate__
            auto holder = t.Holder();
            PADDLE_ENFORCE_EQ(platform::is_cpu_place(holder->place()), true,
                              platform::errors::PreconditionNotMet(
                                  "Tensor is not on CPU."
                                  "Now only Tensor on CPU can be serialized."));
            auto *mmap_writer_allocation =
                dynamic_cast<memory::allocation::MemoryMapWriterAllocation *>(
                    holder.get());
            PADDLE_ENFORCE_NOT_NULL(
                mmap_writer_allocation,
                platform::errors::PreconditionNotMet(
                    "Tensor is not in shared memory."
                    "Now only Tensor on shared memory can be serialized."));
            int type_idx = static_cast<int>(t.type());

            return py::make_tuple(mmap_writer_allocation->ipc_name(),
                                  mmap_writer_allocation->size(), type_idx,
                                  vectorize(t.dims()), t.lod());
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 5)
              throw std::runtime_error("Invalid Tensor state!");

            // 1. Create a new C++ instance
            framework::Tensor tensor;

            // 2. Rebuild Allocation
            const std::string &ipc_name = t[0].cast<std::string>();
            size_t size = t[1].cast<size_t>();
            auto shared_reader_holder =
                memory::allocation::RebuildMemoryMapReaderAllocation(ipc_name,
                                                                     size);

            // 3. Maintain global fd set
            VLOG(3) << "Tensor ipc name: " << ipc_name;
            memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);

            // 4. Rebuild Tensor
            tensor.ResetHolderWithType(
                shared_reader_holder,
                static_cast<paddle::experimental::DataType>(t[2].cast<int>()));
            tensor.Resize(phi::make_ddim(t[3].cast<std::vector<int>>()));
            tensor.set_lod(t[4].cast<framework::LoD>());

            return tensor;
          }));
#endif

  py::class_<phi::SelectedRows>(m, "SelectedRows")
      .def("__init__",
           [](phi::SelectedRows &instance) {
             new (&instance) phi::SelectedRows();
           })
      .def("__init__",
           [](phi::SelectedRows &instance, const std::vector<int64_t> rows,
              const int64_t &height) {
             new (&instance) phi::SelectedRows(rows, height);
           })
      .def("get_tensor",
           [](phi::SelectedRows &self) { return self.mutable_value(); },
           py::return_value_policy::reference)
      .def("numel",
           [](phi::SelectedRows &self) -> int64_t {
             return self.value().numel();
           })
      .def("set_height", &phi::SelectedRows::set_height)
      .def("height", &phi::SelectedRows::height)
      .def("set_rows",
           [](phi::SelectedRows &self, std::vector<int64_t> rows) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
             self.set_rows(rows);
#else
        Vector<int64_t> new_rows(rows);
        self.set_rows(new_rows);
#endif
           })
      .def("sync_index",
           [](phi::SelectedRows &instance) { instance.SyncIndex(); })
      .def("rows", [](phi::SelectedRows &self) {
        auto rows = self.rows();
        std::vector<int64_t> new_rows;
        new_rows.reserve(rows.size());
        std::copy(rows.begin(), rows.end(), std::back_inserter(new_rows));
        return new_rows;
      });

  py::class_<Variable>(m, "Variable", R"DOC(Variable Class.

All parameter, weight, gradient are variables in Paddle.
)DOC")
      .def(py::init<>())
      .def("is_int", [](const Variable &var) { return var.IsType<int>(); })
      .def("set_int",
           [](Variable &var, int val) -> void { *var.GetMutable<int>() = val; })
      .def("get_int", [](const Variable &var) -> int { return var.Get<int>(); })
      .def("is_float", [](const Variable &var) { return var.IsType<float>(); })
      .def("set_float",
           [](Variable &var, float val) -> void {
             *var.GetMutable<float>() = val;
           })
      .def("get_float",
           [](const Variable &var) -> float { return var.Get<float>(); })
      .def("get_tensor",
           [](Variable &self) -> LoDTensor * {
             return self.GetMutable<LoDTensor>();
           },
           py::return_value_policy::reference)
      .def("get_bytes",
           [](Variable &self) {
             return py::bytes(*self.GetMutable<std::string>());
           })
      .def("set_string_list",
           [](Variable &self, Strings str_list) {
             *self.GetMutable<Strings>() = str_list;
           })
      .def("set_vocab", [](Variable &self,
                           Vocab vocab) { *self.GetMutable<Vocab>() = vocab; })
      .def("get_string_tensor",
           [](Variable &self) { return self.GetMutable<Strings>(); },
           py::return_value_policy::reference)
      .def("get_map_tensor",
           [](Variable &self) { return self.GetMutable<Vocab>(); },
           py::return_value_policy::reference)
      .def("get_lod_rank_table",
           [](Variable &self) { return self.GetMutable<LoDRankTable>(); },
           py::return_value_policy::reference)
      .def("get_selected_rows",
           [](Variable &self) -> phi::SelectedRows * {
             return self.GetMutable<phi::SelectedRows>();
           },
           py::return_value_policy::reference)
      .def("get_lod_tensor_array",
           [](Variable &self) { return self.GetMutable<LoDTensorArray>(); },
           py::return_value_policy::reference)
      .def("get_fetch_list",
           [](Variable &self) { return self.GetMutable<FetchList>(); },
           py::return_value_policy::reference)
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      .def("get_communicator",
           [](Variable &self) -> platform::Communicator * {
             return self.GetMutable<platform::Communicator>();
           },
           py::return_value_policy::reference)
#endif
      .def("get_reader",
           [](Variable &self) -> framework::ReaderHolder * {
             PADDLE_ENFORCE_EQ(
                 self.IsType<framework::ReaderHolder>(), true,
                 platform::errors::InvalidArgument(
                     "The variable is not type of ReaderHolder."));
             return self.GetMutable<framework::ReaderHolder>();
           },
           py::return_value_policy::reference)
      .def("get_scope",
           [](Variable &self) -> Scope * {
             auto scope_vec =
                 self.GetMutable<std::vector<framework::Scope *>>();
             PADDLE_ENFORCE_GT(
                 scope_vec->size(), 0,
                 platform::errors::InvalidArgument(
                     "The size of scope_vec should be greater than 0"));
             return scope_vec->front();
           },
           py::return_value_policy::reference)
      .def("set_scope", [](Variable &self, Scope &scope) {
        auto scope_vec = self.GetMutable<std::vector<framework::Scope *>>();
        scope_vec->emplace_back(&scope);
      });

  BindReader(&m);

  py::class_<Scope>(m, "_Scope", R"DOC(
    Scope is an association of a name to Variable. All variables belong to Scope.

    Variables in a parent scope can be retrieved from local scope.

    You need to specify a scope to run a Net, i.e., `exe.Run(&scope)`.
    One net can run in different scopes and update different variable in the
    scope.

    You can create var in a scope and get it from the scope.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          # create tensor from a scope and set value to it.
          param = scope.var('Param').get_tensor()
          param_array = np.full((height, row_numel), 5.0).astype("float32")
          param.set(param_array, place)

        )DOC")
      .def("_remove_from_pool",
           [](Scope &self) { ScopePool::Instance().Remove(&self); })
      .def("var",
           [](Scope &self, const std::string &name) -> Variable * {
             return self.Var(name);
           },
           py::arg("name"),
           R"DOC(
           Find or create variable named :code:`name` in the current scope.

           If the variable named :code:`name` does not exist in the
           current scope, the variable would be created. Otherwise,
           return the existing variable.

           Args:
               name (str): the variable name.

           Returns:
               out (core.Variable): the found or created variable.
           )DOC",
           py::return_value_policy::reference)
      .def("find_var", &Scope::FindVar, py::arg("name"),
           R"DOC(
           Find variable named :code:`name` in the current scope or
           its parent scope. Return None if not found. 

           Args:
               name (str): the variable name.

           Returns:
               out (core.Variable|None): the found variable or None.
           )DOC",
           py::return_value_policy::reference)
      .def("erase", &Scope::EraseVars, py::arg("names"),
           R"DOC(
           Find variable named :code:`name` in the current scope or
           its parent scope. Return None if not found. 

           Args:
               name (str): the variable names to be erase.

           Returns:
               None
           )DOC",
           py::return_value_policy::reference)
      .def("new_scope", [](Scope &self) -> Scope * { return &self.NewScope(); },
           R"DOC(
           Create a new sub-scope of the current scope.

           Returns:
               out (core._Scope): the created sub-scope.
           )DOC",
           py::return_value_policy::reference)
      .def("drop_kids", &Scope::DropKids,
           R"DOC(
           Delete all sub-scopes of the current scope.
           )DOC")
      .def("_kids", &Scope::kids);

  m.def("Scope",
        []() -> Scope * {
          auto *s = new Scope();
          ScopePool::Instance().Insert(std::unique_ptr<Scope>(s));
          return s;
        },
        R"DOC(
        Create a new scope.

        Returns:
            out (core._Scope): the created scope.
        )DOC",
        py::return_value_policy::reference);

  //! @note: Be careful! PyBind will return std::string as an unicode, not
  //! Python str. If you want a str object, you should cast them in Python.
  m.def("get_all_op_protos", []() -> std::vector<py::bytes> {
    std::vector<py::bytes> ret_values;
    for (auto &iter : OpInfoMap::Instance().map()) {
      auto &info = iter.second;
      if (info.HasOpProtoAndChecker()) {
        std::string str;
        PADDLE_ENFORCE_EQ(
            info.Proto().SerializeToString(&str), true,
            platform::errors::Fatal(
                "Serialize OpProto Error. This could be a bug of Paddle."));
        ret_values.emplace_back(str);
      }
    }
    return ret_values;
  });
  m.def("get_op_attrs_default_value",
        [](py::bytes byte_name) -> paddle::framework::AttributeMap {
          std::string op_type = byte_name;
          paddle::framework::AttributeMap res;
          auto info = OpInfoMap::Instance().GetNullable(op_type);
          if (info != nullptr) {
            if (info->HasOpProtoAndChecker()) {
              auto op_checker = info->Checker();
              res = op_checker->GetDefaultAttrsMap();
            }
          }
          return res;
        });
  m.def(
      "get_grad_op_desc", [](const OpDesc &op_desc,
                             const std::unordered_set<std::string> &no_grad_set,
                             const std::vector<BlockDesc *> &grad_sub_block) {
        std::unordered_map<std::string, std::string> grad_to_var;
        std::vector<std::unique_ptr<OpDesc>> grad_op_descs =
            framework::OpInfoMap::Instance()
                .Get(op_desc.Type())
                .GradOpMaker()(op_desc, no_grad_set, &grad_to_var,
                               grad_sub_block);
        std::vector<OpDesc *> grad_op_desc_ptrs(grad_op_descs.size());
        std::transform(grad_op_descs.begin(), grad_op_descs.end(),
                       grad_op_desc_ptrs.begin(),
                       [](std::unique_ptr<OpDesc> &p) { return p.release(); });
        return std::make_pair(grad_op_desc_ptrs, grad_to_var);
      });
  m.def("has_grad_op_maker", [](const std::string op_type) {
    return framework::OpInfoMap::Instance().Get(op_type).HasGradOpMaker();
  });
  m.def("has_non_empty_grad_op_maker", [](const std::string op_type) {
    return framework::OpInfoMap::Instance()
        .Get(op_type)
        .HasNonEmptyGradOpMaker();
  });
  m.def("has_infer_inplace", [](const std::string op_type) {
    return framework::OpInfoMap::Instance().Get(op_type).HasInferInplace();
  });
  m.def("infer_no_need_buffer_slots",
        [](const std::string op_type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs) {
          auto infer_func = framework::OpInfoMap::Instance()
                                .Get(op_type)
                                .NoNeedBufferVarsInferer();
          if (infer_func) {
            return infer_func(inputs, outputs, attrs);
          } else {
            std::unordered_set<std::string> empty = {};
            return empty;
          }
        });
  m.def("prune", [](const ProgramDesc &origin,
                    const std::set<std::string> &feeded_var_names,
                    const std::vector<std::array<size_t, 2>> &targets) {
    ProgramDesc prog_with_targets(origin);

    for (const auto &t : targets) {
      prog_with_targets.MutableBlock(t[0])->Op(t[1])->SetIsTarget(true);
    }
    proto::ProgramDesc pruned_desc;
    auto pruned_origin_block_id_map =
        Prune(*prog_with_targets.Proto(), feeded_var_names, &pruned_desc);
    return std::make_tuple(ProgramDesc(pruned_desc),
                           pruned_origin_block_id_map);
  });
  m.def("prune_backward",
        [](const framework::ProgramDesc &program) {
          return PruneBackward(program);
        },
        R"DOC(
             Prune the backward part of a program, mostly called in
             program.clone(for_test=True).
              
             Args:
                   program (ProgramDesc): The original program.

             Returns:
                   tuple(ProgramDesc, map<int, int>): The first part is 
                   the pruned program desc, and the second part is a map
                   which contains the id pair of pruned block and corresponding
                   origin block.
           )DOC");
  m.def("empty_var_name",
        []() { return std::string(framework::kEmptyVarName); });
  m.def("grad_var_suffix",
        []() { return std::string(framework::kGradVarSuffix); });
  m.def_submodule(
       "var_names",
       "The module will return special predefined variable name in Paddle")
      .def("empty", []() { return kEmptyVarName; })
      .def("temp", []() { return kTempVarName; });

  // clang-format off
  py::class_<paddle::platform::DeviceContext>(m, "DeviceContext")
      .def_static("create",
                  [](paddle::platform::CPUPlace& place)
                      -> paddle::platform::DeviceContext* {
    auto* context = new paddle::platform::CPUDeviceContext();
    context->SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
        .GetAllocator(place)
        .get());
    context->SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
        .GetAllocator(paddle::platform::CPUPlace())
        .get());
    context->SetZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
        .GetZeroAllocator(place)
        .get());
    return context;
                  })
      .def_static("create",
                  [](paddle::platform::XPUPlace& place)
                      -> paddle::platform::DeviceContext* {
#ifndef PADDLE_WITH_XPU
             PADDLE_THROW(
                 platform::errors::PermissionDenied(
                 "Cannot use XPUPlace in CPU/GPU version, "
                 "Please recompile or reinstall Paddle with XPU support."));
#else
      auto* context = new paddle::platform::XPUDeviceContext(place);
      context->SetAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(place)
          .get());
      context->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
      context->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(place)
          .get());
      return context;
#endif
                  })
        .def_static("create",
                  [](paddle::platform::MLUPlace& place)
                      -> paddle::platform::DeviceContext* {
#ifndef PADDLE_WITH_MLU
             PADDLE_THROW(
                 platform::errors::PermissionDenied(
                 "Cannot use MLUPlace in CPU/GPU version, "
                 "Please recompile or reinstall Paddle with MLU support."));
#else
                    return new paddle::platform::MLUDeviceContext(place);
#endif
                  })
        .def_static("create",
                    [](paddle::platform::NPUPlace& place)
                        -> paddle::platform::DeviceContext* {
#ifndef PADDLE_WITH_ASCEND_CL
             PADDLE_THROW(
                 platform::errors::PermissionDenied(
                 "Cannot use NPUPlace in CPU/GPU/XPU version, "
                 "Please recompile or reinstall Paddle with NPU support."));
#else
                return new paddle::platform::NPUDeviceContext(place);
#endif
        })
      .def_static("create",
                  [](paddle::platform::CUDAPlace& place)
                      -> paddle::platform::DeviceContext* {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
             PADDLE_THROW(
                 platform::errors::PermissionDenied(
                 "Cannot use CUDAPlace in CPU only version, "
                 "Please recompile or reinstall Paddle with CUDA support."));
#else
      auto* context = new paddle::platform::CUDADeviceContext(place);
      context->SetAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(place, context->stream())
          .get());
      context->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
      context->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
        .GetZeroAllocator(place)
        .get());
      context->PartialInitWithAllocator();
      return context;
#endif
                  })
          .def_static("create",
                [](paddle::platform::CUDAPinnedPlace& place)
                        -> paddle::platform::DeviceContext* {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
             PADDLE_THROW(
                 platform::errors::PermissionDenied(
                 "Cannot use CUDAPinnedPlace in CPU only version, "
                 "Please recompile or reinstall Paddle with CUDA support."));
#else
                  return new paddle::platform::CUDAPinnedDeviceContext(place);
#endif
                });;
// clang-format on
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  py::class_<platform::Communicator>(m, "Communicator").def(py::init<>());
#endif
  m.def("get_all_device_type", []() {
    std::vector<std::string> device_types;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    device_types = platform::DeviceManager::GetAllDeviceTypes();
#else
          LOG(WARNING) << string::Sprintf(
              "Cannot use get_all_device_type because you have installed"
              "CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_all_device_type, please try to install"
              "CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle-core\n");
#endif
    return device_types;
  });
  m.def("get_all_custom_device_type", []() {
    std::vector<std::string> device_types;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    device_types = platform::DeviceManager::GetAllCustomDeviceTypes();
#else
          LOG(WARNING) << string::Sprintf(
              "Cannot use get_all_custom_device_type because you have installed"
              "CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_all_custom_device_type, please try to "
              "install CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle-core\n");
#endif
    return device_types;
  });
  m.def("get_available_device", [] {
    std::vector<std::string> devices;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    devices = platform::DeviceManager::GetAllDeviceList();
#else
          LOG(WARNING) << string::Sprintf(
              "Cannot use get_available_device because you have installed"
              "CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_available_device, please try to install"
              "CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle-core\n");
#endif
    return devices;
  });
  m.def("get_available_custom_device", [] {
    std::vector<std::string> devices;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    devices = platform::DeviceManager::GetAllCustomDeviceList();
#else
          LOG(WARNING) << string::Sprintf(
              "Cannot use get_available_custom_device because you have "
              "installed"
              "CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_available_custom_device, please try to "
              "install"
              "CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle-core\n");
#endif
    return devices;
  });
  py::class_<platform::CustomPlace>(m, "CustomPlace",
                                    R"DOC(
    CustomPlace is a descriptor of a device.
    It represents a custom device on which a tensor will be allocated and a model will run.

    Examples:
        .. code-block:: python

          import paddle
          fake_cpu_place = paddle.CustomPlace("FakeCPU", 0)
                                             )DOC")
      .def("__init__",
           [](platform::CustomPlace &self, const std::string &device_type,
              int dev_id) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
             if (UNLIKELY(dev_id < 0)) {
               LOG(ERROR) << string::Sprintf(
                   "Invalid CustomPlace(%s, %d), device id must be 0 "
                   "or "
                   "positive integer",
                   device_type, dev_id);
               std::exit(-1);
             }

             if (LIKELY(platform::DeviceManager::HasDeviceType(device_type) &&
                        platform::DeviceManager::IsCustom(device_type))) {
               int dev_count = static_cast<int>(
                   platform::DeviceManager::GetDeviceCount(device_type));
               if (UNLIKELY(dev_id >= dev_count)) {
                 if (dev_count == 0) {
                   LOG(ERROR) << "Cannot use " << device_type
                              << " because there is no " << device_type
                              << " detected on your "
                                 "machine.";
                   std::exit(-1);
                 } else {
                   LOG(ERROR) << string::Sprintf(
                       "Invalid CustomPlace(%s, %d), dev_id must "
                       "inside "
                       "[0, %d), because %s "
                       "number on your machine is %d",
                       device_type, dev_id, dev_count, device_type, dev_count);
                   std::exit(-1);
                 }
               }
               new (&self) platform::CustomPlace(device_type, dev_id);
             } else {
               LOG(ERROR) << string::Sprintf(
                   "Invalid CustomPlace(%s, %d), the device type is "
                   "not registered "
                   "as a custom device.",
                   device_type, dev_id);
               std::exit(-1);
             }
#else
             LOG(ERROR) << string::Sprintf(
                 "Cannot use CustomDevice because you have installed CPU/GPU"
                 "version PaddlePaddle.\n"
                 "If you want to use CustomDevice, please try to install"
                 "CustomDevice version "
                 "PaddlePaddle by: pip install paddlepaddle-core\n"
                 "If you only have CPU, please change "
                 "CustomPlace(%s, %d) to be CPUPlace().\n",
                 device_type, dev_id);
             std::exit(-1);
#endif
           })
      .def("get_device_id",
           [](const platform::CustomPlace &self) { return self.GetDeviceId(); })
      .def("get_device_type",
           [](const platform::CustomPlace &self) {
             return self.GetDeviceType();
           })
      .def("__repr__", string::to_string<const platform::CustomPlace &>)
      .def("__str__", string::to_string<const platform::CustomPlace &>);
  py::class_<platform::CUDAPlace> cudaplace(m, "CUDAPlace", R"DOC(

    CUDAPlace is a descriptor of a device.
    It represents a GPU device allocated or to be allocated with Tensor or LoDTensor.
    Each CUDAPlace has a dev_id to indicate the graphics card ID represented by the current CUDAPlace,
    staring from 0.
    The memory of CUDAPlace with different dev_id is not accessible.
    Numbering here refers to the logical ID of the visible graphics card, not the actual ID of the graphics card.
    You can set visible GPU devices by setting the `CUDA_VISIBLE_DEVICES` environment variable.
    When the program starts, visible GPU devices will be numbered from 0.
    If `CUDA_VISIBLE_DEVICES` is not set, all devices are visible by default,
    and the logical ID is the same as the actual ID.

    Parameters:
        id (int): GPU device ID.

    Examples:
        .. code-block:: python

          import paddle

          place = paddle.CUDAPlace(0)

        )DOC");
  g_cudaplace_pytype = reinterpret_cast<PyTypeObject *>(cudaplace.ptr());
  cudaplace
      .def("__init__",
           [](platform::CUDAPlace &self, int dev_id) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
             if (UNLIKELY(dev_id < 0)) {
               LOG(ERROR) << string::Sprintf(
                   "Invalid CUDAPlace(%d), device id must be 0 or "
                   "positive integer",
                   dev_id);
               std::exit(-1);
             }

             if (UNLIKELY(dev_id >= platform::GetGPUDeviceCount())) {
               if (platform::GetGPUDeviceCount() == 0) {
                 LOG(ERROR) << "Cannot use GPU because there is no GPU "
                               "detected on your "
                               "machine.";
                 std::exit(-1);
               } else {
                 LOG(ERROR) << string::Sprintf(
                     "Invalid CUDAPlace(%d), must inside [0, %d), because GPU "
                     "number on your machine is %d",
                     dev_id, platform::GetGPUDeviceCount(),
                     platform::GetGPUDeviceCount());
                 std::exit(-1);
               }
             }

             new (&self) platform::CUDAPlace(dev_id);
#else
             LOG(ERROR) << string::Sprintf(
                 "Cannot use GPU because you have installed CPU version "
                 "PaddlePaddle.\n"
                 "If you want to use GPU, please try to install GPU version "
                 "PaddlePaddle by: pip install paddlepaddle-gpu\n"
                 "If you only have CPU, please change CUDAPlace(%d) to be "
                 "CPUPlace().\n",
                 dev_id);
             std::exit(-1);
#endif
           })
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def("get_device_id",
           [](const platform::CUDAPlace &self) { return self.GetDeviceId(); })
      .def("_type", &PlaceIndex<platform::CUDAPlace>)
      .def("_equals", &IsSamePlace<platform::CUDAPlace, platform::Place>)
      .def("_equals", &IsSamePlace<platform::CUDAPlace, platform::CUDAPlace>)
      .def("_equals", &IsSamePlace<platform::CUDAPlace, platform::CPUPlace>)
      .def("_equals", &IsSamePlace<platform::CUDAPlace, platform::XPUPlace>)
      .def("_equals", &IsSamePlace<platform::CUDAPlace, platform::NPUPlace>)
      .def("_equals", &IsSamePlace<platform::CUDAPlace, platform::MLUPlace>)
      .def("_equals",
           &IsSamePlace<platform::CUDAPlace, platform::CUDAPinnedPlace>)
      .def("_get_device_id",
           [](platform::CUDAPlace &self) -> int { return self.GetDeviceId(); })
#endif
      .def("__repr__", string::to_string<const platform::CUDAPlace &>)
      .def("__str__", string::to_string<const platform::CUDAPlace &>);

  py::class_<platform::XPUPlace> xpuplace(m, "XPUPlace", R"DOC(
    **Note**:
    Examples:
        .. code-block:: python
          import paddle.fluid as fluid
          xpu_place = fluid.XPUPlace(0)
        )DOC");
  g_xpuplace_pytype = reinterpret_cast<PyTypeObject *>(xpuplace.ptr());
  xpuplace
      .def("__init__",
           [](platform::XPUPlace &self, int dev_id) {
#ifdef PADDLE_WITH_XPU
             if (UNLIKELY(dev_id < 0)) {
               LOG(ERROR) << string::Sprintf(
                   "Invalid XPUPlace(%d), device id must be 0 or "
                   "positive integer",
                   dev_id);
               std::exit(-1);
             }
             if (UNLIKELY(dev_id >= platform::GetXPUDeviceCount())) {
               if (platform::GetXPUDeviceCount() == 0) {
                 LOG(ERROR) << "Cannot use XPU because there is no XPU "
                               "detected on your "
                               "machine.";
                 std::exit(-1);
               } else {
                 LOG(ERROR) << string::Sprintf(
                     "Invalid XPUPlace(%d), must inside [0, %d), because XPU "
                     "number on your machine is %d",
                     dev_id, platform::GetXPUDeviceCount(),
                     platform::GetXPUDeviceCount());
                 std::exit(-1);
               }
             }
             new (&self) platform::XPUPlace(dev_id);
#else
             LOG(ERROR) << string::Sprintf(
                 "Cannot use XPU because you have installed CPU/GPU version "
                 "PaddlePaddle.\n"
                 "If you want to use XPU, please try to install XPU version "
                 "PaddlePaddle by: pip install paddlepaddle-xpu\n"
                 "If you only have CPU, please change XPUPlace(%d) to be "
                 "CPUPlace().\n",
                 dev_id);
             std::exit(-1);
#endif
           })
#ifdef PADDLE_WITH_XPU
      .def("_type", &PlaceIndex<platform::XPUPlace>)
      .def("_equals", &IsSamePlace<platform::XPUPlace, platform::Place>)
      .def("_equals", &IsSamePlace<platform::XPUPlace, platform::CUDAPlace>)
      .def("_equals", &IsSamePlace<platform::XPUPlace, platform::CPUPlace>)
      .def("_equals", &IsSamePlace<platform::XPUPlace, platform::XPUPlace>)
      .def("_equals",
           &IsSamePlace<platform::XPUPlace, platform::CUDAPinnedPlace>)
      .def("get_device_id",
           [](const platform::XPUPlace &self) { return self.GetDeviceId(); })
#endif
      .def("__repr__", string::to_string<const platform::XPUPlace &>)
      .def("__str__", string::to_string<const platform::XPUPlace &>);
#ifdef PADDLE_WITH_XPU
  py::enum_<phi::backends::xpu::XPUVersion>(m, "XPUVersion", py::arithmetic())
      .value("XPU1", phi::backends::xpu::XPUVersion::XPU1)
      .value("XPU2", phi::backends::xpu::XPUVersion::XPU2)
      .export_values();
  m.def("get_xpu_device_count", platform::GetXPUDeviceCount);
  m.def("get_xpu_device_version",
        [](int device_id) { return platform::get_xpu_version(device_id); });
  m.def("get_xpu_device_op_support_types",
        [](const std::string &op_name, phi::backends::xpu::XPUVersion version) {
          return platform::get_xpu_op_support_type(op_name, version);
        });
  m.def("get_xpu_device_op_list", [](phi::backends::xpu::XPUVersion version) {
    return platform::get_xpu_op_list(version);
  });
  m.def("is_float16_supported", [](const platform::XPUPlace &place) -> bool {
    // XPUs with Compute Capability > xpu2 support float16 and bfloat16
    return platform::get_xpu_version(place.device) >
           phi::backends::xpu::XPUVersion::XPU1;
  });
  m.def("is_bfloat16_supported", [](const platform::XPUPlace &place) -> bool {
    // XPUs with Compute Capability > xpu2 support float16 and bfloat16
    return platform::get_xpu_version(place.device) >
           phi::backends::xpu::XPUVersion::XPU1;
  });
#endif

  py::class_<paddle::platform::CPUPlace> cpuplace(m, "CPUPlace", R"DOC(
    CPUPlace is a descriptor of a device.
    It represents a CPU device on which a tensor will be allocated and a model will run.

    Examples:
        .. code-block:: python

          import paddle
          cpu_place = paddle.CPUPlace()

        )DOC");
  g_cpuplace_pytype = reinterpret_cast<PyTypeObject *>(cpuplace.ptr());
  cpuplace.def(py::init<>())
      .def("_type", &PlaceIndex<platform::CPUPlace>)
      .def("_equals", &IsSamePlace<platform::CPUPlace, platform::Place>)
      .def("_equals", &IsSamePlace<platform::CPUPlace, platform::XPUPlace>)
      .def("_equals", &IsSamePlace<platform::CPUPlace, platform::NPUPlace>)
      .def("_equals", &IsSamePlace<platform::CPUPlace, platform::CUDAPlace>)
      .def("_equals", &IsSamePlace<platform::CPUPlace, platform::CPUPlace>)
      .def("_equals",
           &IsSamePlace<platform::CPUPlace, platform::CUDAPinnedPlace>)
      .def("__repr__", string::to_string<const platform::CPUPlace &>)
      .def("__str__", string::to_string<const platform::CPUPlace &>);

  py::class_<paddle::platform::CUDAPinnedPlace> cudapinnedplace(
      m, "CUDAPinnedPlace", R"DOC(
    CUDAPinnedPlace is a descriptor of a device.
    It refers to the page locked memory allocated by the CUDA function `cudaHostAlloc()` in the host memory.
    The host operating system will not paging and exchanging the memory.
    It can be accessed through direct memory access technology to speed up the copy of data between the host and GPU.
    For more information on CUDA data transfer and `pinned memory`,
    please refer to `official document <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory>`_ .

    Examples:
        .. code-block:: python

          import paddle
          place = paddle.CUDAPinnedPlace()

        )DOC");
  g_cudapinnedplace_pytype =
      reinterpret_cast<PyTypeObject *>(cudapinnedplace.ptr());
  cudapinnedplace
      .def("__init__",
           [](platform::CUDAPinnedPlace &self) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
             PADDLE_THROW(platform::errors::PermissionDenied(
                 "Cannot use CUDAPinnedPlace in CPU only version, "
                 "Please recompile or reinstall Paddle with CUDA support."));
#endif
             new (&self) platform::CUDAPinnedPlace();
           })
      .def("_type", &PlaceIndex<platform::CUDAPinnedPlace>)
      .def("_equals", &IsSamePlace<platform::CUDAPinnedPlace, platform::Place>)
      .def("_equals",
           &IsSamePlace<platform::CUDAPinnedPlace, platform::CUDAPlace>)
      .def("_equals",
           &IsSamePlace<platform::CUDAPinnedPlace, platform::XPUPlace>)
      .def("_equals",
           &IsSamePlace<platform::CUDAPinnedPlace, platform::NPUPlace>)
      .def("_equals",
           &IsSamePlace<platform::CUDAPinnedPlace, platform::CPUPlace>)
      .def("_equals",
           &IsSamePlace<platform::CUDAPinnedPlace, platform::CUDAPinnedPlace>)
      .def("__repr__", string::to_string<const platform::CUDAPinnedPlace &>)
      .def("__str__", string::to_string<const platform::CUDAPinnedPlace &>);

  // NPUPlace
  py::class_<platform::NPUPlace> npuplace(m, "NPUPlace", R"DOC(
    NPUPlace is a descriptor of a device.
    It represents a NPU device on which a tensor will be allocated and a model will run.

    Examples:
        .. code-block:: python
          import paddle
          npu_place = paddle.NPUPlace(0)

        )DOC");
  g_npuplace_pytype = reinterpret_cast<PyTypeObject *>(npuplace.ptr());
  npuplace
      .def("__init__",
           [](platform::NPUPlace &self, int dev_id) {
#ifdef PADDLE_WITH_ASCEND_CL
             if (UNLIKELY(dev_id < 0)) {
               LOG(ERROR) << string::Sprintf(
                   "Invalid NPUPlace(%d), device id must be 0 or "
                   "positive integer",
                   dev_id);
               std::exit(-1);
             }
             if (UNLIKELY(dev_id >= platform::GetNPUDeviceCount())) {
               if (platform::GetNPUDeviceCount() == 0) {
                 LOG(ERROR) << "Cannot use NPU because there is no NPU "
                               "detected on your "
                               "machine.";
                 std::exit(-1);
               } else {
                 LOG(ERROR) << string::Sprintf(
                     "Invalid NPUPlace(%d), must inside [0, %d), because NPU "
                     "number on your machine is %d",
                     dev_id, platform::GetNPUDeviceCount(),
                     platform::GetNPUDeviceCount());
                 std::exit(-1);
               }
             }
             new (&self) platform::NPUPlace(dev_id);
#else
             LOG(ERROR) << string::Sprintf(
                 "Cannot use NPU because you have installed CPU/GPU version "
                 "PaddlePaddle.\n"
                 "If you want to use NPU, please try to install NPU version "
                 "PaddlePaddle by: pip install paddlepaddle-npu\n"
                 "If you only have CPU, please change NPUPlace(%d) to be "
                 "CPUPlace().\n",
                 dev_id);
             std::exit(-1);
#endif
           })
      .def("_type", &PlaceIndex<platform::NPUPlace>)
      .def("_equals", &IsSamePlace<platform::NPUPlace, platform::Place>)
      .def("_equals", &IsSamePlace<platform::NPUPlace, platform::CUDAPlace>)
      .def("_equals", &IsSamePlace<platform::NPUPlace, platform::CPUPlace>)
      .def("_equals", &IsSamePlace<platform::NPUPlace, platform::XPUPlace>)
      .def("_equals", &IsSamePlace<platform::NPUPlace, platform::NPUPlace>)
      .def("_equals",
           &IsSamePlace<platform::NPUPlace, platform::CUDAPinnedPlace>)
      .def("get_device_id",
           [](const platform::NPUPlace &self) { return self.GetDeviceId(); })
      .def("__str__", string::to_string<const platform::NPUPlace &>);

  // IPUPlace
  py::class_<platform::IPUPlace>(m, "IPUPlace", R"DOC(
    IPUPlace is a descriptor of a device.
    It represents a IPU device on which a tensor will be allocated and a model will run.

    Examples:
        .. code-block:: python
          import paddle

          # required: ipu

          ipu_place = paddle.IPUPlace()

        )DOC")
      .def("__init__",
           [](platform::IPUPlace &self) {
#ifdef PADDLE_WITH_IPU
             if (platform::GetIPUDeviceCount() == 0) {
               LOG(ERROR) << "Cannot use IPU because there is no IPU "
                             "detected on your "
                             "machine.";
               std::exit(-1);
             }
             // use ipu(0) to comile, while run with the number user configure
             // in sharding and pipline.
             new (&self) platform::IPUPlace(0);
#else
             LOG(ERROR) << string::Sprintf(
                 "Cannot use IPU because you didn't install IPU version "
                 "PaddlePaddle.\n"
                 "If you want to use IPU, please try to install IPU version "
                 "PaddlePaddle by: pip install paddlepaddle*\n"
                 "If you only have CPU, please change IPUPlace to be "
                 "CPUPlace().\n");
             std::exit(-1);
#endif
           })
      .def("_type", &PlaceIndex<platform::IPUPlace>)
      .def("_equals", &IsSamePlace<platform::IPUPlace, platform::Place>)
      .def("_equals", &IsSamePlace<platform::IPUPlace, platform::CUDAPlace>)
      .def("_equals", &IsSamePlace<platform::IPUPlace, platform::CPUPlace>)
      .def("_equals", &IsSamePlace<platform::IPUPlace, platform::XPUPlace>)
      .def("_equals", &IsSamePlace<platform::IPUPlace, platform::NPUPlace>)
      .def("_equals", &IsSamePlace<platform::IPUPlace, platform::IPUPlace>)
      .def("_equals",
           &IsSamePlace<platform::IPUPlace, platform::CUDAPinnedPlace>)
#ifdef PADDLE_WITH_IPU
      .def("get_device_id",
           [](const platform::IPUPlace &self) { return self.GetDeviceId(); })
#endif
      .def("__str__", string::to_string<const platform::IPUPlace &>);

  // MLUPlace
  py::class_<platform::MLUPlace> mluplace(m, "MLUPlace", R"DOC(
    MLUPlace is a descriptor of a device.
    It represents a MLU device on which a tensor will be allocated and a model will run.

    Examples:
        .. code-block:: python
          import paddle
          # required: mlu
          mlu_place = paddle.MLUPlace(0)

        )DOC");
  g_mluplace_pytype = reinterpret_cast<PyTypeObject *>(mluplace.ptr());
  mluplace
      .def("__init__",
           [](platform::MLUPlace &self, int dev_id) {
#ifdef PADDLE_WITH_MLU
             if (UNLIKELY(dev_id < 0)) {
               LOG(ERROR) << string::Sprintf(
                   "Invalid MLUPlace(%d), device id must be 0 or "
                   "positive integer",
                   dev_id);
               std::exit(-1);
             }
             if (UNLIKELY(dev_id >= platform::GetMLUDeviceCount())) {
               if (platform::GetMLUDeviceCount() == 0) {
                 LOG(ERROR) << "Cannot use MLU because there is no MLU "
                               "detected on your "
                               "machine.";
                 std::exit(-1);
               } else {
                 LOG(ERROR) << string::Sprintf(
                     "Invalid MLUPlace(%d), must inside [0, %d), because MLU "
                     "number on your machine is %d",
                     dev_id, platform::GetMLUDeviceCount(),
                     platform::GetMLUDeviceCount());
                 std::exit(-1);
               }
             }
             new (&self) platform::MLUPlace(dev_id);
#else
             LOG(ERROR) << string::Sprintf(
                 "Cannot use MLU because you have installed CPU/GPU/... "
                 "version "
                 "PaddlePaddle.\n"
                 "If you want to use MLU, please try to install MLU version "
                 "PaddlePaddle by: pip install paddlepaddle-mlu\n"
                 "If you only have CPU, please change MLUPlace(%d) to be "
                 "CPUPlace().\n",
                 dev_id);
             std::exit(-1);
#endif
           })
      .def("_type", &PlaceIndex<platform::MLUPlace>)
#ifdef PADDLE_WITH_MLU
      .def("_equals", &IsSamePlace<platform::MLUPlace, platform::Place>)
      .def("_equals", &IsSamePlace<platform::MLUPlace, platform::CUDAPlace>)
      .def("_equals", &IsSamePlace<platform::MLUPlace, platform::CPUPlace>)
      .def("_equals", &IsSamePlace<platform::MLUPlace, platform::XPUPlace>)
      .def("_equals", &IsSamePlace<platform::MLUPlace, platform::NPUPlace>)
      .def("_equals", &IsSamePlace<platform::MLUPlace, platform::IPUPlace>)
      .def("_equals", &IsSamePlace<platform::MLUPlace, platform::MLUPlace>)
      .def("_equals",
           &IsSamePlace<platform::MLUPlace, platform::CUDAPinnedPlace>)
      .def("get_device_id",
           [](const platform::MLUPlace &self) { return self.GetDeviceId(); })
#endif
      .def("__str__", string::to_string<const platform::MLUPlace &>);

  py::class_<platform::Place> platformplace(m, "Place");
  g_place_pytype = reinterpret_cast<PyTypeObject *>(platformplace.ptr());
  platformplace.def(py::init<>())
      .def("_type", &PlaceIndex<platform::Place>)
      .def("_equals", &IsSamePlace<platform::Place, platform::Place>)
      .def("_equals", &IsSamePlace<platform::Place, platform::CUDAPlace>)
      .def("_equals", &IsSamePlace<platform::Place, platform::CPUPlace>)
      .def("_equals", &IsSamePlace<platform::Place, platform::XPUPlace>)
      .def("_equals", &IsSamePlace<platform::Place, platform::NPUPlace>)
      .def("_equals", &IsSamePlace<platform::Place, platform::IPUPlace>)
      .def("_equals", &IsSamePlace<platform::Place, platform::CUDAPinnedPlace>)
      .def("_equals", &IsSamePlace<platform::Place, platform::MLUPlace>)
      .def("is_gpu_place",
           [](platform::Place &self) { return platform::is_gpu_place(self); })
      .def("is_cpu_place",
           [](platform::Place &self) { return platform::is_cpu_place(self); })
      .def("is_xpu_place",
           [](platform::Place &self) { return platform::is_xpu_place(self); })
      .def("is_npu_place",
           [](platform::Place &self) { return platform::is_npu_place(self); })
      .def("is_ipu_place",
           [](platform::Place &self) { return platform::is_ipu_place(self); })
      .def("is_cuda_pinned_place",
           [](platform::Place &self) {
             return platform::is_cuda_pinned_place(self);
           })
      .def("is_mlu_place",
           [](platform::Place &self) { return platform::is_mlu_place(self); })
      .def(
          "is_custom_place",
          [](platform::Place &self) { return platform::is_custom_place(self); })
      .def("gpu_device_id", [](platform::Place &self) { return self.device; })
      .def("xpu_device_id", [](platform::Place &self) { return self.device; })
      .def("npu_device_id", [](platform::Place &self) { return self.device; })
      .def("ipu_device_id", [](platform::Place &self) { return self.device; })
      .def("mlu_device_id", [](platform::Place &self) { return self.device; })
      .def("custom_device_id",
           [](platform::Place &self) { return self.device; })
      .def("set_place", [](platform::Place &self,
                           const platform::Place &other) { self = other; })
      .def("set_place",
           [](platform::Place &self, const platform::CPUPlace &cpu_place) {
             self = cpu_place;
           })
      .def("set_place",
           [](platform::Place &self, const platform::XPUPlace &xpu_place) {
             self = xpu_place;
           })
      .def("set_place",
           [](platform::Place &self, const platform::CUDAPlace &gpu_place) {
             self = gpu_place;
           })
      .def("set_place",
           [](platform::Place &self,
              const platform::CUDAPinnedPlace &cuda_pinned_place) {
             self = cuda_pinned_place;
           })
      .def("set_place",
           [](platform::Place &self, const platform::NPUPlace &npu_place) {
             self = npu_place;
           })
      .def("set_place",
           [](platform::Place &self, const platform::IPUPlace &ipu_place) {
             self = ipu_place;
           })
      .def("set_place",
           [](platform::Place &self, const platform::MLUPlace &mlu_place) {
             self = mlu_place;
           })
      .def("set_place",
           [](platform::Place &self, const platform::CustomPlace &plug_place) {
             self = plug_place;
           })
      .def("__repr__", string::to_string<const platform::Place &>)
      .def("__str__", string::to_string<const platform::Place &>);

  py::class_<OperatorBase>(m, "Operator")
      .def_static("create",
                  [](py::bytes protobin) {
                    proto::OpDesc desc;
                    PADDLE_ENFORCE_EQ(desc.ParsePartialFromString(protobin),
                                      true,
                                      platform::errors::InvalidArgument(
                                          "Cannot parse user input to OpDesc"));
                    PADDLE_ENFORCE_EQ(desc.IsInitialized(), true,
                                      platform::errors::InvalidArgument(
                                          "The provided OpDesc is not "
                                          "initialized, the reason is: %s",
                                          desc.InitializationErrorString()));
                    return OpRegistry::CreateOp(desc);
                  })
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::CPUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::XPUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::NPUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::CUDAPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::CUDAPinnedPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::MLUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("type",
           [](const OperatorBase &op) -> std::string { return op.Type(); })
      .def("outputs",
           [](const OperatorBase &op)
               -> std::map<std::string, std::vector<std::string>> {
                 return op.Outputs();
               })
      .def("output_vars",
           [](const OperatorBase &op) { return op.OutputVars(true); })
      .def("inputs", [](const OperatorBase &op) { return op.Inputs(); })
      .def("input_vars", [](const OperatorBase &op) { return op.InputVars(); })
      .def("__str__", &OperatorBase::DebugString)
      .def("no_intermediate_outputs",
           [](const OperatorBase &op) { return op.OutputVars(false); })
      .def("support_gpu", &OperatorBase::SupportGPU);

  py::class_<framework::ExecutorPrepareContext>(m, "ExecutorPrepareContext")
      .def(py::init<const ProgramDesc &, size_t>());

  py::class_<framework::TrainerBase, std::shared_ptr<framework::TrainerBase>>(
      m, "TrainerBase")
      .def("get_worker_scope",
           [](TrainerBase &self, int thread_id) -> Scope * {
             return self.GetWorkerScope(thread_id);
           },
           py::return_value_policy::reference)
      .def("finalize", &TrainerBase::Finalize)
      .def("ResetDataset", &TrainerBase::ResetDataset);

  m.def("_get_eager_deletion_vars", &framework::GetEagerDeletionCleanVars);

  py::class_<framework::Executor>(m, "Executor")
      .def(py::init<const platform::Place &>())
      .def("close", &Executor::Close)
      .def("run_from_dataset", &Executor::RunFromDataset,
           py::call_guard<py::gil_scoped_release>())
      .def("release_trainer", &Executor::ReleaseTrainer,
           py::call_guard<py::gil_scoped_release>())
      .def("init_for_dataset",
           [](Executor &self, const ProgramDesc &prog,
              const std::string &trainer_desc, Scope *scope,
              Dataset *dataset) -> std::shared_ptr<TrainerBase> {
             pybind11::gil_scoped_release release;
             return self.InitForDataset(prog, trainer_desc, scope, dataset);
           })
      .def("run_from_dataset",
           [](Executor &self, std::shared_ptr<TrainerBase> trainer) {
             pybind11::gil_scoped_release release;
             self.RunFromDataset(trainer);
           })
      .def("run_prepared_ctx",
           [](Executor &self, ExecutorPrepareContext *ctx, Scope *scope,
              std::map<std::string, const LoDTensor *> *feed_targets,
              std::map<std::string, FetchType *> *fetch_targets,
              bool create_local_scope = true, bool create_vars = true,
              const std::string &feed_holder_name = "feed",
              const std::string &fetch_holder_name = "fetch") {
             pybind11::gil_scoped_release release;
             self.RunPreparedContext(ctx, scope, feed_targets, fetch_targets,
                                     create_local_scope, create_vars,
                                     feed_holder_name, fetch_holder_name);
           })
      .def("run_prepared_ctx",
           [](Executor &self, ExecutorPrepareContext *ctx, Scope *scope,
              bool create_local_scope = true, bool create_vars = true,
              bool keep_kids = false) {
             pybind11::gil_scoped_release release;
             self.RunPreparedContext(ctx, scope, create_local_scope,
                                     create_vars, keep_kids);
           })
      .def("prepare",
           [](Executor &self, const ProgramDesc &program, int block_id,
              const std::vector<std::string> &skip_ref_cnt_vars =
                  std::vector<std::string>(),
              bool force_disable_gc = false) {
             pybind11::gil_scoped_release release;
             return self.Prepare(program, block_id, skip_ref_cnt_vars,
                                 force_disable_gc);
           })
      .def("create_variables", &Executor::CreateVariables)
      .def("run", [](Executor &self, const ProgramDesc &prog, Scope *scope,
                     int block_id, bool create_local_scope, bool create_vars,
                     const std::vector<std::string> &fetch_vars) {
        pybind11::gil_scoped_release release;
        self.Run(prog, scope, block_id, create_local_scope, create_vars,
                 fetch_vars);
      });

  py::class_<framework::interpreter::CostInfo>(m, "CostInfo")
      .def(py::init<>())
      .def("total_time",
           [](interpreter::CostInfo &self) { return self.total_time; })
      .def("device_memory_bytes", [](interpreter::CostInfo &self) {
        return self.device_memory_bytes;
      });

  py::class_<framework::StandaloneExecutor>(m, "StandaloneExecutor")
      .def(py::init<const platform::Place &, const ProgramDesc &,
                    const ProgramDesc &, Scope *>())
      .def("run",
           [](StandaloneExecutor &self,
              const std::unordered_map<std::string, py::array> &input_dict,
              std::vector<std::string> fetch_names) {
             std::vector<framework::LoDTensor> feed_tensors;
             std::vector<std::string> feed_names;

             for (auto &item : input_dict) {
               framework::LoDTensor t;
               SetTensorFromPyArray<platform::CPUPlace>(
                   &t, item.second, platform::CPUPlace(), false);
               feed_names.push_back(item.first);
               feed_tensors.push_back(t);
             }

             paddle::framework::FetchList ret;
             {
               pybind11::gil_scoped_release release;
               ret = self.Run(feed_names, feed_tensors, fetch_names);
             }
             return py::cast(std::move(ret));
           })
      .def("run",
           [](StandaloneExecutor &self,
              const std::unordered_map<std::string, framework::LoDTensor>
                  &input_dict,
              std::vector<std::string> fetch_names) {
             std::vector<framework::LoDTensor> feed_tensors;
             std::vector<std::string> feed_names;

             for (auto &item : input_dict) {
               feed_names.push_back(item.first);
               feed_tensors.push_back(item.second);
             }

             paddle::framework::FetchList ret;
             {
               pybind11::gil_scoped_release release;
               ret = self.Run(feed_names, feed_tensors, fetch_names);
             }
             return py::cast(std::move(ret));
           })
      .def("run",
           [](StandaloneExecutor &self, std::vector<std::string> feed_names,
              std::vector<std::string> fetch_names) {
             paddle::framework::FetchList ret;
             {
               pybind11::gil_scoped_release release;
               ret = self.Run(feed_names, fetch_names);
             }
             return py::cast(std::move(ret));
           })
      .def("dry_run",
           [](StandaloneExecutor &self,
              const std::unordered_map<std::string, py::array> &input_dict) {
             std::vector<framework::LoDTensor> feed_tensors;
             std::vector<std::string> feed_names;

             for (auto &item : input_dict) {
               framework::LoDTensor t;
               SetTensorFromPyArray<platform::CPUPlace>(
                   &t, item.second, platform::CPUPlace(), false);
               feed_names.push_back(item.first);
               feed_tensors.push_back(t);
             }

             framework::interpreter::CostInfo cost_info;
             {
               pybind11::gil_scoped_release release;
               cost_info = self.DryRun(feed_names, feed_tensors);
             }
             return cost_info;
           });

  m.def("init_gflags", framework::InitGflags);
  m.def("init_glog", framework::InitGLOG);
  m.def("load_op_meta_info_and_register_op",
        framework::LoadOpMetaInfoAndRegisterOp);
  m.def("init_devices", []() { framework::InitDevices(); });

  m.def("is_compiled_with_cuda", IsCompiledWithCUDA);
  m.def("is_compiled_with_ascend", IsCompiledWithAscend);
  m.def("is_compiled_with_rocm", IsCompiledWithROCM);
  m.def("is_compiled_with_npu", IsCompiledWithNPU);
  m.def("is_compiled_with_ipu", IsCompiledWithIPU);
  m.def("is_compiled_with_xpu", IsCompiledWithXPU);
  m.def("is_compiled_with_mkldnn", IsCompiledWithMKLDNN);
  m.def("is_compiled_with_nccl", IsCompiledWithNCCL);
  m.def("is_compiled_with_cinn", IsCompiledWithCINN);
  m.def("is_compiled_with_mlu", IsCompiledWithMLU);
  m.def("_is_compiled_with_heterps", IsCompiledWithHETERPS);
  m.def("supports_bfloat16", SupportsBfloat16);
  m.def("supports_bfloat16_fast_performance", SupportsBfloat16FastPerformance);
  m.def("supports_int8", SupportsInt8);
  m.def("supports_vnni", SupportsVNNI);
  m.def("op_supported_infos", imperative::OpSupportedInfos);
  m.def("is_compiled_with_brpc", IsCompiledWithBrpc);
  m.def("is_compiled_with_dist", IsCompiledWithDIST);
  m.def("_cuda_synchronize", [](const platform::CUDAPlace &place) {
    platform::DeviceContextPool::Instance().Get(place)->Wait();
  });

  m.def("get_float_stats", []() {
    std::vector<paddle::platform::ExportedStatValue<float>> float_stats;
    paddle::platform::StatRegistry<float>::Instance().publish(float_stats);
    std::unordered_map<std::string, float> stats_map;
    for (const auto &stat : float_stats) {
      stats_map[stat.key] = stat.value;
    }
    return stats_map;
  });
  m.def("get_int_stats", []() {
    std::vector<paddle::platform::ExportedStatValue<int64_t>> int_stats;
    paddle::platform::StatRegistry<int64_t>::Instance().publish(int_stats);
    std::unordered_map<std::string, int64_t> stats_map;
    for (const auto &stat : int_stats) {
      stats_map[stat.key] = stat.value;
    }
    return stats_map;
  });
  m.def("run_cmd",
        [](const std::string &cmd, int time_out = -1,
           int sleep_inter = -1) -> const std::string {
          return paddle::framework::shell_get_command_output(cmd, time_out,
                                                             sleep_inter);
        },
        py::arg("cmd"), py::arg("time_out") = -1, py::arg("sleep_inter") = -1);
  m.def("shell_execute_cmd",
        [](const std::string &cmd, int time_out = 0, int sleep_inter = 0,
           bool redirect_stderr = false) -> std::vector<std::string> {
          return paddle::framework::shell_execute_cmd(
              cmd, time_out, sleep_inter, redirect_stderr);
        },
        py::arg("cmd"), py::arg("time_out") = 0, py::arg("sleep_inter") = 0,
        py::arg("redirect_stderr") = false);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  m.def("is_float16_supported", [](const platform::CUDAPlace &place) -> bool {
    // Only GPUs with Compute Capability >= 53 support float16
    return platform::GetGPUComputeCapability(place.device) >= 53;
  });
#endif

  m.def("set_feed_variable",
        static_cast<void (*)(Scope *, const LoDTensor &, const std::string &,
                             size_t)>(&framework::SetFeedVariable));
  m.def("set_feed_variable",
        static_cast<void (*)(Scope *, const Strings &, const std::string &,
                             size_t)>(&framework::SetFeedVariable));
  m.def("get_fetch_variable",
        [](const Scope &scope, const std::string &var_name,
           size_t index) -> py::object {
          auto &var = framework::GetFetchVariable(scope, var_name, index);
          if (data_is_lod_tensor(var)) {
            return py::cast(BOOST_GET(LoDTensor, var));
          } else {
            return py::cast(BOOST_GET(LoDTensorArray, var));
          }
        });
  m.def("get_variable_tensor", framework::GetVariableTensor);

  m.def("_is_program_version_supported", IsProgramVersionSupported);

  BindProgramDesc(&m);
  BindBlockDesc(&m);
  BindVarDsec(&m);
  BindOpDesc(&m);
  BindCostModel(&m);
  BindConstValue(&m);
  BindGlobalValueGetterSetter(&m);
  BindProcessMeshDesc(&m);
  BindFleetExecutor(&m);
  BindTCPStore(&m);

  py::class_<framework::LoDRankTable>(m, "LodRankTable")
      .def("items", [](framework::LoDRankTable &table) {
        std::vector<std::pair<size_t, size_t>> res;
        for (auto &item : table.items()) {
          res.push_back({item.index, item.length});
        }
        return res;
      });

  py::class_<LoDTensorArray> pylodtensorarray(m, "LoDTensorArray", R"DOC(
    LoDTensorArray is array of LoDTensor, it supports operator[], len() and for-loop iteration.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          arr = fluid.LoDTensorArray()
)DOC");
  g_framework_lodtensorarray_pytype =
      reinterpret_cast<PyTypeObject *>(pylodtensorarray.ptr());
  pylodtensorarray
      .def("__init__",
           [](LoDTensorArray &instance) { new (&instance) LoDTensorArray(); })
      .def("__getitem__",
           [](LoDTensorArray &self, size_t i) { return &self.at(i); },
           py::return_value_policy::reference)
      .def("__len__", [](LoDTensorArray &self) { return self.size(); })
      .def("__setitem__",
           [](LoDTensorArray &self, size_t i, const LoDTensor &t) {
             PADDLE_ENFORCE_LT(i, self.size(),
                               platform::errors::InvalidArgument(
                                   "The index to set is larger than the size "
                                   "of LoDTensorArray."));
             self[i].ShareDataWith(t);
             self[i].set_lod(t.lod());
           })
      .def("append",
           [](LoDTensorArray &self, const LoDTensor &t) {
             self.emplace_back();
             self.back().ShareDataWith(t);
             self.back().set_lod(t.lod());
           },
           py::arg("tensor"), R"DOC(
             Append a LoDensor to LoDTensorArray.
              
             Args:
                   tensor (LoDTensor): The LoDTensor to be appended.

             Returns:
                   None.

             Examples:
                 .. code-block:: python

                   import paddle.fluid as fluid
                   import numpy as np

                   arr = fluid.LoDTensorArray()
                   t = fluid.LoDTensor()
                   t.set(np.ndarray([5, 30]), fluid.CPUPlace())
                   arr.append(t)
           )DOC")
      .def("_move_to_list",
           [](LoDTensorArray &self) -> py::list {
             py::list res(self.size());
             for (size_t i = 0; i < self.size(); ++i) {
               res[i] = py::cast(std::move(self[i]));
             }
             self.clear();
             return res;
           },
           py::return_value_policy::take_ownership);

  py::class_<FetchList>(m, "FetchList", R"DOC( FetchList is a
        vector of boost::variant<LoDTensor, LoDTensorArray>.
        )DOC")
      .def("_move_to_list",
           [](FetchList &self) -> py::list {
             py::list res(self.size());
             for (size_t i = 0; i < self.size(); ++i) {
               if (data_is_lod_tensor(self[i])) {
                 auto &data = BOOST_GET(LoDTensor, self[i]);
                 res[i] = py::cast(std::move(data));
               } else {
                 auto &data = BOOST_GET(LoDTensorArray, self[i]);
                 py::list tmp(data.size());
                 for (size_t j = 0; j < data.size(); ++j) {
                   tmp[j] = py::cast(std::move(data[j]));
                 }
                 res[i] = std::move(tmp);
               }
             }
             self.clear();
             return res;
           },
           py::return_value_policy::take_ownership)

      .def("append",
           [](FetchList &self, const LoDTensor &t) {
             self.emplace_back();
             auto &lod_tensor = BOOST_GET(LoDTensor, self.back());
             lod_tensor.ShareDataWith(t);
             lod_tensor.set_lod(t.lod());
           },
           py::arg("var"))

      .def("append",
           [](FetchList &self, const LoDTensorArray &t) {
             self.emplace_back();
             auto &lod_tensor_array = BOOST_GET(LoDTensorArray, self.back());
             for (size_t i = 0; i < t.size(); ++i) {
               lod_tensor_array[i].ShareDataWith(t[i]);
               lod_tensor_array[i].set_lod(t[i].lod());
             }
           },
           py::arg("var"));

  py::class_<FetchUnmergedList>(m, "FetchUnmergedList", R"DOC(
        FetchUnmergedList is 2-D array of FetchType(boost::variant(LoDTensor, LoDTensorArray)).
        )DOC")
      .def("_move_to_list",
           [](FetchUnmergedList &self) -> py::list {
             py::list res(self.size());
             for (size_t i = 0; i < self.size(); ++i) {
               py::list tmp(self[i].size());
               for (size_t j = 0; j < self[i].size(); ++j) {
                 if (data_is_lod_tensor(self[i][j])) {
                   auto &var = BOOST_GET(LoDTensor, self[i][j]);
                   tmp[j] = py::cast(std::move(var));
                 } else {
                   auto &var = BOOST_GET(LoDTensorArray, self[i][j]);
                   py::list tmp_array(var.size());
                   for (size_t k = 0; k < var.size(); ++k) {
                     tmp_array[k] = std::move(var[k]);
                   }
                   tmp[j] = std::move(tmp_array);
                 }
               }
               res[i] = std::move(tmp);
               self[i].clear();
             }
             self.clear();
             return res;
           },
           py::return_value_policy::take_ownership);

  m.def("op_support_gpu", OpSupportGPU);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  m.def("get_cuda_device_count", platform::GetGPUDeviceCount);
  m.def("cuda_empty_cache", [] {
    for (int dev_id : platform::GetSelectedDevices()) {
      auto *dev_ctx = platform::DeviceContextPool::Instance().GetByPlace(
          platform::CUDAPlace(dev_id));
      dev_ctx->cudnn_workspace_handle().ResetWorkspace();
    }
    platform::EmptyCache();
  });
  m.def("get_device_properties",
        [](int id) -> const gpuDeviceProp & {
          return platform::GetDeviceProperties(id);
        },
        py::return_value_policy::copy);

  py::class_<gpuDeviceProp>(m, "_gpuDeviceProperties")
      .def_property_readonly(
          "name", [](const gpuDeviceProp &prop) { return prop.name; })
      .def_property_readonly(
          "major", [](const gpuDeviceProp &prop) { return prop.major; })
      .def_property_readonly(
          "minor", [](const gpuDeviceProp &prop) { return prop.minor; })
      .def_property_readonly(
          "total_memory",
          [](const gpuDeviceProp &prop) { return prop.totalGlobalMem; })
      .def_property_readonly(
          "multi_processor_count",
          [](const gpuDeviceProp &prop) { return prop.multiProcessorCount; })
      .def_property_readonly(
          "is_multi_gpu_board",
          [](const gpuDeviceProp &prop) { return prop.isMultiGpuBoard; })
      .def_property_readonly(
          "is_integrated",
          [](const gpuDeviceProp &prop) { return prop.integrated; })
      .def("__repr__", [](const gpuDeviceProp &prop) {
        std::stringstream ostr;
        ostr << "_gpuDeviceProperties(name='" << prop.name
             << "', major=" << prop.major << ", minor=" << prop.minor
             << ", total_memory=" << prop.totalGlobalMem / (1024 * 1024)
             << "MB, multi_processor_count=" << prop.multiProcessorCount << ")";
        return ostr.str();
      });

#if !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
  m.def("nvprof_init", platform::CudaProfilerInit);
  m.def("nvprof_start", platform::CudaProfilerStart);
  m.def("nvprof_stop", platform::CudaProfilerStop);
  m.def("nvprof_nvtx_push", platform::CudaNvtxRangePush);
  m.def("nvprof_nvtx_pop", platform::CudaNvtxRangePop);
  m.def("nvprof_enable_record_event", platform::NvprofEnableRecordEvent);
  m.def("nvprof_disable_record_event", platform::NvprofDisableRecordEvent);
#endif
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  m.def("get_npu_device_count", platform::GetNPUDeviceCount);
  m.def("npu_finalize", []() {
    platform::HCCLCommContext::Instance().ReleaseHCCLComms();

    auto &pool = platform::DeviceContextPool::Instance();
    auto devices = platform::GetSelectedNPUDevices();
    for (size_t i = 0; i < devices.size(); ++i) {
      platform::NPUDeviceGuard guard(devices[i]);
      pool.Get(platform::NPUPlace(devices[i]))->Wait();
    }
    platform::AclInstance::Instance().Finalize();
  });

  py::class_<platform::NPUProfConfigWrapper>(m, "NPUProfConfigWrapper");

  m.def("npu_prof_init", platform::NPUProfilerInit);
  m.def("npu_prof_start", [](platform::NPUProfConfigWrapper c) {
    platform::NPUProfilerStart(c.ptr());
  });
  m.def("npu_prof_stop", [](platform::NPUProfConfigWrapper c) {
    platform::NPUProfilerStop(c.ptr());
  });
  m.def("npu_prof_finalize", platform::NPUProfilerFinalize);
  m.def("npu_prof_create_config", []() {
    return platform::NPUProfConfigWrapper(platform::NPUProfilerCreateConfig());
  });

  m.def("npu_prof_destropy_config", [](platform::NPUProfConfigWrapper c) {
    platform::NPUProfilerDestroyConfig(c.ptr());
  });
#endif

#ifdef PADDLE_WITH_IPU
  m.def("get_ipu_device_count", platform::GetIPUDeviceCount);
#endif

#ifdef PADDLE_WITH_MLU
  m.def("get_mlu_device_count", platform::GetMLUDeviceCount);
#endif

  py::enum_<platform::TracerOption>(m, "TracerOption", py::arithmetic())
      .value("kDefault", platform::TracerOption::kDefault)
      .value("kOpDetail", platform::TracerOption::kOpDetail)
      .value("kAllOpDetail", platform::TracerOption::kAllOpDetail)
      .export_values();

  py::enum_<platform::ProfilerState>(m, "ProfilerState", py::arithmetic())
      .value("kDisabled", platform::ProfilerState::kDisabled)
      .value("kCPU", platform::ProfilerState::kCPU)
      .value("kCUDA", platform::ProfilerState::kCUDA)
      .value("kAll", platform::ProfilerState::kAll)
      .export_values();

  py::enum_<platform::EventSortingKey>(m, "EventSortingKey", py::arithmetic())
      .value("kDefault", platform::EventSortingKey::kDefault)
      .value("kCalls", platform::EventSortingKey::kCalls)
      .value("kTotal", platform::EventSortingKey::kTotal)
      .value("kMin", platform::EventSortingKey::kMin)
      .value("kMax", platform::EventSortingKey::kMax)
      .value("kAve", platform::EventSortingKey::kAve)
      .export_values();

  m.def("set_tracer_option", platform::SetTracerOption);
  m.def("enable_profiler", platform::EnableProfiler);
  m.def("disable_profiler", platform::DisableProfiler);
  m.def("is_profiler_enabled", platform::IsProfileEnabled);
  m.def("reset_profiler", platform::ResetProfiler);
  m.def("register_pass", [](const std::string &pass_type, py::object callable) {
    PADDLE_ENFORCE_EQ(
        framework::ir::PassRegistry::Instance().Has(pass_type), false,
        platform::errors::AlreadyExists("Pass '%s' is registered more than "
                                        "once. Please use another name.",
                                        pass_type));
    callable.inc_ref();
    framework::ir::PassRegistry::Instance().Insert(pass_type, [pass_type,
                                                               callable]() {
      py::gil_scoped_acquire guard;
      std::unique_ptr<framework::ir::Pass> pass(
          new framework::ir::GeneratePass(py::cast<std::string>(callable())));
      return pass;
    });
  });
  m.def("get_pass", [](const std::string &pass_type) {
    auto pass = framework::ir::PassRegistry::Instance().Get(pass_type);
    return std::shared_ptr<framework::ir::Pass>(std::move(pass));
  });

  m.def("size_of_dtype", framework::SizeOfType);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  m.def("set_cublas_switch", platform::SetAllowTF32Cublas);
  m.def("get_cublas_switch", platform::AllowTF32Cublas);
  m.def("set_cudnn_switch", platform::SetAllowTF32Cudnn);
  m.def("get_cudnn_switch", platform::AllowTF32Cudnn);
#endif  // PADDLE_WITH_CUDA
  m.def("clear_executor_cache",
        []() { framework::ExecutorInfoCache::Instance().Finalize(); });

  using VarQuantScale =
      std::unordered_map<std::string, std::pair<bool, LoDTensor>>;

  py::class_<ir::Pass, std::shared_ptr<ir::Pass>> pass(m, "Pass");
  pass.def(py::init())
      .def("has", &ir::Pass::Has)
      .def("set_not_owned",
           [](ir::Pass &self, const std::string &attr_name, ProgramDesc &attr) {
             self.SetNotOwned<ProgramDesc>(attr_name, &attr);
           })
      .def(
          "set",
          [](ir::Pass &self, const std::string &name, const std::string &attr) {
            self.Set<std::string>(name, new std::string(attr));
          })
      .def("set", [](ir::Pass &self, const std::string &name,
                     bool val) { self.Set<bool>(name, new bool(val)); })
      .def("set", [](ir::Pass &self, const std::string &name,
                     int val) { self.Set<const int>(name, new int(val)); })
      .def("set",
           [](ir::Pass &self, const std::string &name,
              std::vector<std::string> set) {
             self.Set(name, new std::vector<std::string>(set));
           })
      .def("set",
           [](ir::Pass &self, const std::string &name,
              std::unordered_set<std::string> set) {
             self.Set(name, new std::unordered_set<std::string>(set));
           })
      .def("set",
           [](ir::Pass &self, const std::string &name,
              std::unordered_set<int> set) {
             self.Set(name, new std::unordered_set<int>(set));
           })
      .def("set",
           [](ir::Pass &self, const std::string &name, VarQuantScale scales) {
             self.Set(name, new VarQuantScale(scales));
           })
      .def("type", &ir::Pass::Type)
      .def("apply", [](ir::Pass &self, std::shared_ptr<ir::Graph> graph) {
        self.Apply(graph.get());
      });

  py::class_<ir::PassBuilder, std::shared_ptr<ir::PassBuilder>> pb(
      m, "PassBuilder");
  pb.def(py::init())
      .def("append_pass",
           [](ir::PassBuilder &self,
              const std::string &pass_type) -> std::shared_ptr<ir::Pass> {
             return self.AppendPass(pass_type);
           })
      .def("all_passes", [](ir::PassBuilder &self) { return self.AllPasses(); })
      .def("insert_pass",
           [](ir::PassBuilder &self, size_t idx, const std::string &pass_type) {
             return self.InsertPass(idx, pass_type);
           })
      .def("remove_pass",
           [](ir::PassBuilder &self, size_t idx) { self.RemovePass(idx); });

  // -- python binds for parallel executor.
  py::class_<ParallelExecutor> pe(m, "ParallelExecutor");
  py::class_<ExecutionStrategy> exec_strategy(pe, "ExecutionStrategy", R"DOC(
    ExecutionStrategy allows the user to more preciously control how to run
    the program in ParallelExecutor by setting the property.

    Returns:
        ExecutionStrategy: An ExecutionStrategy object.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.static as static
          import paddle.nn.functional as F

          paddle.enable_static()

          x = static.data(name='x', shape=[None, 13], dtype='float32')
          y = static.data(name='y', shape=[None, 1], dtype='float32')
          y_predict = static.nn.fc(input=x, size=1, act=None)

          cost = F.square_error_cost(input=y_predict, label=y)
          avg_loss = paddle.mean(cost)

          sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
          sgd_optimizer.minimize(avg_loss)

          exec_strategy = static.ExecutionStrategy()
          exec_strategy.num_threads = 4

          train_exe = static.ParallelExecutor(use_cuda=False,
                                              loss_name=avg_loss.name,
                                              exec_strategy=exec_strategy)
        )DOC");

  py::enum_<paddle::platform::DeviceType>(m, "DeviceType", py::arithmetic())
      .value("CPU", paddle::platform::DeviceType::CPU)
      .value("CUDA", paddle::platform::DeviceType::CUDA)
      .value("XPU", paddle::platform::DeviceType::XPU);

  exec_strategy.def(py::init())
      .def_property(
          "num_threads",
          [](const ExecutionStrategy &self) { return self.num_threads_; },
          [](ExecutionStrategy &self, size_t num_threads) {
            self.num_threads_ = num_threads;
          },
          R"DOC(
            The type is INT, num_threads represents the size of thread pool that
            used to run the operators of the current program in ParallelExecutor.
            If :math:`num\_threads=1`, all the operators will execute one by one,
            but the order maybe difference between iterations.
            If it is not set, it will be set in ParallelExecutor according to the
            device type and device count, for GPU, :math:`num\_threads=device\_count*4`, for CPU,
            :math:`num\_threads=CPU\_NUM*4`, the explanation of:math:`CPU\_NUM` is in ParallelExecutor.
            if it is not set, ParallelExecutor will get the cpu count by calling
            `multiprocessing.cpu_count()`. Default 0.

            Examples:
                .. code-block:: python

                    import paddle
                    import paddle.static as static

                    paddle.enable_static()

                    exec_strategy = static.ExecutionStrategy()
                    exec_strategy.num_threads = 4
            )DOC")
      .def_property(
          "_use_device",
          [](const ExecutionStrategy &self) { return self.use_device_; },
          [](ExecutionStrategy &self, paddle::platform::DeviceType use_device) {
            self.use_device_ = use_device;
          })  // NOTE(liuyuhui): Doesn't add doc for 'use_device', because
              // use_device isn‘t exposed to users.
      .def_property(
          "allow_op_delay",
          [](const ExecutionStrategy &self) { return self.allow_op_delay_; },
          [](ExecutionStrategy &self, bool allow_op_delay) {
            self.allow_op_delay_ = allow_op_delay;
          },
          R"DOC(The type is BOOL, allow_op_delay represents whether to delay the
                communication operators to run, it may make the execution faster.
                Note that this option is invalid now, and it will be removed in
                next version. Default False.)DOC")
      .def_property(
          "num_iteration_per_drop_scope",
          [](const ExecutionStrategy &self) {
            return self.num_iteration_per_drop_scope_;
          },
          [](ExecutionStrategy &self, size_t num_iteration_per_drop_scope) {
            self.num_iteration_per_drop_scope_ = num_iteration_per_drop_scope;
          },
          R"DOC(The type is INT, num_iteration_per_drop_scope indicates how
                many iterations to clean up the temp variables which
                is generated during execution. It may make the execution faster,
                because the temp variable's shape maybe the same between two iterations.
                Default 100.

                .. note::
                    1. If you fetch data when calling the 'run', the ParallelExecutor 
                    will clean up the temp variables at the end of the current iteration. 
                    2. In some NLP model, it may cause the GPU memory is insufficient, 
                    in this case, you should reduce `num_iteration_per_drop_scope`.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        exec_strategy = static.ExecutionStrategy()
                        exec_strategy.num_iteration_per_drop_scope = 10
              )DOC")
      .def_property(
          "num_iteration_per_run",
          [](const ExecutionStrategy &self) {
            return self.num_iteration_per_run_;
          },
          [](ExecutionStrategy &self, size_t num_iteration_per_run) {
            self.num_iteration_per_run_ = num_iteration_per_run;
          },
          R"DOC(This config that how many iteration the executor will run when
                user call exe.run() in python。Default: 1.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        exec_strategy = static.ExecutionStrategy()
                        exec_strategy.num_iteration_per_run = 10
              )DOC")
      .def_property(
          "use_thread_barrier",
          [](const ExecutionStrategy &self) { return self.thread_barrier_; },
          [](ExecutionStrategy &self, bool use_thread_barrier) {
            self.thread_barrier_ = use_thread_barrier;
          },
          R"DOC(This config that the this is distributed training with parameter server
              )DOC")
      .def_property("_dry_run",
                    [](const ExecutionStrategy &self) { return self.dry_run_; },
                    [](ExecutionStrategy &self, bool dry_run) {
                      self.dry_run_ = dry_run;
                    });

  exec_strategy.def_property(
      "use_experimental_executor",
      [](const ExecutionStrategy &self) {
        return self.type_ == ExecutionStrategy::kExperimental;
      },
      [](ExecutionStrategy &self, bool experimental) {
        self.type_ = experimental ? ExecutionStrategy::kExperimental
                                  : ExecutionStrategy::kDefault;
      });

  py::class_<BuildStrategy> build_strategy(pe, "BuildStrategy", R"DOC(
    BuildStrategy allows the user to more preciously control how to
    build the SSA Graph in ParallelExecutor by setting the property.

    Returns:
        BuildStrategy: An BuildStrategy object.

    Examples:
        .. code-block:: python

            import os
            import paddle
            import paddle.static as static

            paddle.enable_static()

            os.environ['CPU_NUM'] = str(2)
            places = static.cpu_places()

            data = static.data(name="x", shape=[None, 1], dtype="float32")
            hidden = static.nn.fc(input=data, size=10)
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

            build_strategy = static.BuildStrategy()
            build_strategy.enable_inplace = True
            build_strategy.memory_optimize = True
            build_strategy.reduce_strategy = static.BuildStrategy.ReduceStrategy.Reduce
            program = static.CompiledProgram(static.default_main_program())
            program = program.with_data_parallel(loss_name=loss.name,
                                                  build_strategy=build_strategy,
                                                  places=places)
)DOC");

  py::enum_<BuildStrategy::ReduceStrategy>(build_strategy, "ReduceStrategy")
      .value("Reduce", BuildStrategy::ReduceStrategy::kReduce)
      .value("AllReduce", BuildStrategy::ReduceStrategy::kAllReduce)
      .value("_NoReduce", BuildStrategy::ReduceStrategy::kNoReduce);
  py::enum_<BuildStrategy::GradientScaleStrategy>(build_strategy,
                                                  "GradientScaleStrategy")
      .value("CoeffNumDevice",
             BuildStrategy::GradientScaleStrategy::kCoeffNumDevice)
      .value("One", BuildStrategy::GradientScaleStrategy::kOne)
      .value("Customized", BuildStrategy::GradientScaleStrategy::kCustomized);

  build_strategy.def(py::init())
      .def("_clear_finalized", &BuildStrategy::ClearFinalized)
      .def_property(
          "reduce_strategy",
          [](const BuildStrategy &self) { return self.reduce_; },
          [](BuildStrategy &self, BuildStrategy::ReduceStrategy strategy) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.reduce_ = strategy;
          },
          R"DOC((fluid.BuildStrategy.ReduceStrategy, optional): there are two reduce
                strategies in ParallelExecutor, AllReduce and Reduce. If you want
                that all the parameters' optimization are done on all devices independently,
                you should choose AllReduce; otherwise, if you choose Reduce, all the parameters'
                optimization will be evenly distributed to different devices, and then
                broadcast the optimized parameter to other devices.
                Default is 'AllReduce'.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.reduce_strategy = static.BuildStrategy.ReduceStrategy.Reduce
                  )DOC")
      .def_property(
          "gradient_scale_strategy",
          [](const BuildStrategy &self) { return self.gradient_scale_; },
          [](BuildStrategy &self,
             BuildStrategy::GradientScaleStrategy strategy) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.gradient_scale_ = strategy;
          },
          R"DOC((paddle.static.BuildStrategy.GradientScaleStrategy, optional): there are three
                ways of defining :math:`loss@grad` in ParallelExecutor, that is, CoeffNumDevice,
                One and Customized. By default, ParallelExecutor sets the :math:`loss@grad`
                according to the number of devices. If you want to customize :math:`loss@grad`,
                you can choose Customized. Default is 'CoeffNumDevice'.

                Examples:
                    .. code-block:: python

                        import numpy
                        import os
                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        use_cuda = True
                        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
                        exe = static.Executor(place)

                        # NOTE: If you use CPU to run the program, you need
                        # to specify the CPU_NUM, otherwise, paddle will use
                        # all the number of the logic core as the CPU_NUM,
                        # in that case, the batch size of the input should be
                        # greater than CPU_NUM, if not, the process will be
                        # failed by an exception.
                        if not use_cuda:
                            os.environ['CPU_NUM'] = str(2)
                            places = static.cpu_places()
                        else:
                            places = static.cuda_places()

                        data = static.data(name='X', shape=[None, 1], dtype='float32')
                        hidden = static.nn.fc(input=data, size=10)
                        loss = paddle.mean(hidden)
                        paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

                        exe.run(static.default_startup_program())

                        build_strategy = static.BuildStrategy()
                        build_strategy.gradient_scale_strategy = \
                                  static.BuildStrategy.GradientScaleStrategy.Customized
                        compiled_prog = static.CompiledProgram(
                                  static.default_main_program()).with_data_parallel(
                                          loss_name=loss.name, build_strategy=build_strategy,
                                          places=places)

                        dev_count =  len(places)
                        x = numpy.random.random(size=(10, 1)).astype('float32')
                        loss_grad = numpy.ones((dev_count)).astype("float32") * 0.01
                        loss_grad_name = loss.name+"@GRAD"
                        loss_data = exe.run(compiled_prog,
                                              feed={"X": x, loss_grad_name : loss_grad},
                                              fetch_list=[loss.name, loss_grad_name])
                   )DOC")
      .def_property(
          "debug_graphviz_path",
          [](const BuildStrategy &self) { return self.debug_graphviz_path_; },
          [](BuildStrategy &self, const std::string &path) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.debug_graphviz_path_ = path;
          },
          R"DOC((str, optional): debug_graphviz_path indicates the path that
                writing the SSA Graph to file in the form of graphviz.
                It is useful for debugging. Default is empty string, that is, ""

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.debug_graphviz_path = "./graph"
                    )DOC")
      .def_property(
          "enable_sequential_execution",
          [](const BuildStrategy &self) {
            return self.enable_sequential_execution_;
          },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.enable_sequential_execution_ = b;
          },
          R"DOC((bool, optional): If set True, the execution order of ops would
                be the same as what is in the program. Default is False.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.enable_sequential_execution = True
          )DOC")
      .def_property(
          "remove_unnecessary_lock",
          [](const BuildStrategy &self) {
            return self.remove_unnecessary_lock_;
          },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.remove_unnecessary_lock_ = b;
          },
          R"DOC((bool, optional): If set True, some locks in GPU ops would be
                released and ParallelExecutor would run faster. Default is True.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.remove_unnecessary_lock = True
          )DOC")
      .def_property(
          "num_trainers",
          [](const BuildStrategy &self) { return self.num_trainers_; },
          [](BuildStrategy &self, int num_trainers) {
#ifdef WIN32
            PADDLE_THROW(platform::errors::Unavailable(
                "Distribution mode is not supported on Windows platform."));
#endif
            self.num_trainers_ = num_trainers;
          })
      .def_property(
          "trainers_endpoints",
          [](const BuildStrategy &self) { return self.trainers_endpoints_; },
          [](BuildStrategy &self,
             const std::vector<std::string> &trainers_endpoints) {
            self.trainers_endpoints_ = trainers_endpoints;
          })
      .def_property("trainer_id",
                    [](const BuildStrategy &self) { return self.trainer_id_; },
                    [](BuildStrategy &self, int trainer_id) {
                      self.trainer_id_ = trainer_id;
                    })
      .def_property(
          "nccl_comm_num",
          [](const BuildStrategy &self) { return self.nccl_comm_num_; },
          [](BuildStrategy &self, int nccl_comm_num) {
            self.nccl_comm_num_ = nccl_comm_num;
          })
      .def_property(
          "bkcl_comm_num",
          [](const BuildStrategy &self) { return self.bkcl_comm_num_; },
          [](BuildStrategy &self, int bkcl_comm_num) {
            self.bkcl_comm_num_ = bkcl_comm_num;
          })
      .def_property("use_hierarchical_allreduce",
                    [](const BuildStrategy &self) {
                      return self.use_hierarchical_allreduce_;
                    },
                    [](BuildStrategy &self, bool use) {
                      self.use_hierarchical_allreduce_ = use;
                    })
      .def_property("hierarchical_allreduce_inter_nranks",
                    [](const BuildStrategy &self) {
                      return self.hierarchical_allreduce_inter_nranks_;
                    },
                    [](BuildStrategy &self, int nranks) {
                      self.hierarchical_allreduce_inter_nranks_ = nranks;
                    })

      .def_property(
          "fuse_elewise_add_act_ops",
          [](const BuildStrategy &self) {
            return self.fuse_elewise_add_act_ops_;
          },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.fuse_elewise_add_act_ops_ = b;
          },
          R"DOC((bool, optional): fuse_elewise_add_act_ops indicate whether
                to fuse elementwise_add_op and activation_op,
                it may make the execution faster. Default is False.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.fuse_elewise_add_act_ops = True
                     )DOC")
      .def_property(
          "fuse_bn_act_ops",
          [](const BuildStrategy &self) { return self.fuse_bn_act_ops_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.fuse_bn_act_ops_ = b;
          },
          R"DOC((bool, optional): fuse_bn_act_ops indicate whether
                to fuse batch_norm and activation_op,
                it may make the execution faster. Default is False.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.fuse_bn_act_ops = True
                     )DOC")
      .def_property(
          "fuse_bn_add_act_ops",
          [](const BuildStrategy &self) { return self.fuse_bn_add_act_ops_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.fuse_bn_add_act_ops_ = b;
          },
          R"DOC((bool, optional): fuse_bn_add_act_ops indicate whether
                to fuse batch_norm, elementwise_add and activation_op,
                it may make the execution faster. Default is True

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.fuse_bn_add_act_ops = True
                     )DOC")
      .def_property(
          "enable_auto_fusion",
          [](const BuildStrategy &self) { return self.enable_auto_fusion_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.enable_auto_fusion_ = b;
          },
          R"DOC((bool, optional): Whether to enable fusing subgraph to a
                fusion_group. Now we only support fusing subgraph that composed
                of elementwise-like operators, such as elementwise_add/mul
                without broadcast and activations.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.enable_auto_fusion = True
                    )DOC")
      .def_property(
          "fuse_relu_depthwise_conv",
          [](const BuildStrategy &self) {
            return self.fuse_relu_depthwise_conv_;
          },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.fuse_relu_depthwise_conv_ = b;
          },
          R"DOC((bool, optional): fuse_relu_depthwise_conv indicate whether
                to fuse relu and depthwise_conv2d,
                it will save GPU memory and may make the execution faster.
                This options is only available in GPU devices.
                Default is False.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.fuse_relu_depthwise_conv = True
          )DOC")
      .def_property("fuse_broadcast_ops",
                    [](const BuildStrategy &self) {
                      return self.fuse_broadcast_ops_ == true ||
                             self.fuse_broadcast_ops_ == paddle::none;
                    },
                    [](BuildStrategy &self, bool b) {
                      PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                                        platform::errors::PreconditionNotMet(
                                            "BuildStrategy has been finlaized, "
                                            "cannot be configured again."));
                      self.fuse_broadcast_ops_ = b;
                    },
                    R"DOC((bool, optional): fuse_broadcast_op indicates whether
                      to fuse the broadcast ops. Note that, in Reduce mode,
                      fusing broadcast ops may make the program faster. Because
                      fusing broadcast OP equals delaying the execution of all
                      broadcast Ops, in this case, all nccl streams are used only
                      for NCCLReduce operations for a period of time. Default False.

                      Examples:
                          .. code-block:: python

                              import paddle
                              import paddle.static as static

                              paddle.enable_static()

                              build_strategy = static.BuildStrategy()
                              build_strategy.fuse_broadcast_ops = True
                    )DOC")
      .def_property("fuse_all_optimizer_ops",
                    [](const BuildStrategy &self) {
                      return self.fuse_all_optimizer_ops_ == true ||
                             self.fuse_all_optimizer_ops_ == paddle::none;
                    },
                    [](BuildStrategy &self, bool b) {
                      PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                                        platform::errors::PreconditionNotMet(
                                            "BuildStrategy has been finlaized, "
                                            "cannot be configured again."));
                      self.fuse_all_optimizer_ops_ = b;
                    })
      .def_property(
          "sync_batch_norm",
          [](const BuildStrategy &self) { return self.sync_batch_norm_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(), true,
                              platform::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.sync_batch_norm_ = b;
          },
          R"DOC((bool, optional): sync_batch_norm indicates whether to use
                synchronous batch normalization which synchronizes the mean
                and variance through multi-devices in training phase.
                Current implementation doesn't support FP16 training and CPU.
                And only synchronous on one machine, not all machines. 
                Default is False.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.sync_batch_norm = True
                )DOC")
      .def_property(
          "memory_optimize",
          [](const BuildStrategy &self) -> py::object {
            if (self.memory_optimize_) {
              return py::cast(self.memory_optimize_.get());
            } else {
              return py::cast(nullptr);
            }
          },
          [](BuildStrategy &self, const py::handle &value) {
            auto *py_obj = value.ptr();
            if (py_obj == nullptr || py_obj == Py_None) {
              self.memory_optimize_ = paddle::none;
            } else if (PyBool_Check(py_obj)) {
              self.memory_optimize_ = (py_obj == Py_True);
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "BuildStrategy.memory_optimize must be set to None, False "
                  "or True"));
            }
          },
          R"DOC((bool, optional): memory opitimize aims to save total memory
                consumption, set to True to enable it.

                Default None. None means framework would choose to use or not use 
                this strategy automatically. Currently, None means that it is 
                enabled when GC is disabled, and disabled when GC is enabled. 
                True means enabling and False means disabling. Default is None.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.memory_optimize = True
                
                )DOC")
      .def_property(
          "is_distribution",
          [](const BuildStrategy &self) { return self.is_distribution_; },
          [](BuildStrategy &self, bool b) {
#ifdef WIN32
            if (b) {
              PADDLE_THROW(platform::errors::Unavailable(
                  "Distribution mode is not supported on Windows platform."));
            }
#else
            self.is_distribution_ = b;
#endif
          })
      .def_property("async_mode",
                    [](const BuildStrategy &self) { return self.async_mode_; },
                    [](BuildStrategy &self, bool b) { self.async_mode_ = b; })
      .def_property(
          "enable_inplace",
          [](const BuildStrategy &self) { return self.enable_inplace_; },
          [](BuildStrategy &self, bool b) { self.enable_inplace_ = b; })
      .def_property(
          "enable_addto",
          [](const BuildStrategy &self) { return self.enable_addto_; },
          [](BuildStrategy &self, bool b) { self.enable_addto_ = b; })
      .def_property(
          "fuse_all_reduce_ops",
          [](const BuildStrategy &self) {
            return self.fuse_all_reduce_ops_ == true ||
                   self.fuse_all_reduce_ops_ == paddle::none;
          },
          [](BuildStrategy &self, bool b) { self.fuse_all_reduce_ops_ = b; })
      .def_property("enable_backward_optimizer_op_deps",
                    [](const BuildStrategy &self) {
                      return self.enable_backward_optimizer_op_deps_;
                    },
                    [](BuildStrategy &self, bool b) {
                      self.enable_backward_optimizer_op_deps_ = b;
                    })
      .def_property(
          "cache_runtime_context",
          [](const BuildStrategy &self) { return self.cache_runtime_context_; },
          [](BuildStrategy &self, bool b) { self.cache_runtime_context_ = b; })
      .def_property(
          "mkldnn_enabled_op_types",
          [](const BuildStrategy &self) {
            return self.mkldnn_enabled_op_types_;
          },
          [](BuildStrategy &self,
             const std::unordered_set<std::string> &mkldnn_enabled_op_types) {
            self.mkldnn_enabled_op_types_ = mkldnn_enabled_op_types;
          })
      .def_property(
          "fix_op_run_order",
          [](const BuildStrategy &self) { return self.fix_op_run_order_; },
          [](BuildStrategy &self, bool fix_op_run_order) {
            self.fix_op_run_order_ = fix_op_run_order;
          })
      .def_property("allow_cuda_graph_capture",
                    [](const BuildStrategy &self) {
                      return self.allow_cuda_graph_capture_;
                    },
                    [](BuildStrategy &self, bool allow_cuda_graph_capture) {
                      self.allow_cuda_graph_capture_ = allow_cuda_graph_capture;
                    })
      .def("_copy",
           [](const BuildStrategy &self) {
             auto new_bs = self;
             new_bs.ClearFinalized();
             return new_bs;
           })
      .def("_finalize_strategy_and_create_passes",
           [](BuildStrategy &self) -> std::shared_ptr<ir::PassBuilder> {
             return self.CreatePassesFromStrategy(true);
           },
           R"DOC(Allow user to customized passes. Normally model-specific
                optimization passes should be defined in this way. BuildStrategy
                cannot be updated after being finalized.)DOC");

  m.def("_set_cached_executor_build_strategy",
        [](int64_t program_id, const BuildStrategy &build_strategy) {
          auto &cached_exe_info = framework::ExecutorInfoCache::Instance();
          cached_exe_info.SetBuildStrategy(program_id, build_strategy);
        });

  pe.def(py::init<const std::vector<platform::Place> &,
                  const std::vector<std::string> &, const std::string &,
                  Scope *, std::vector<Scope *> &, const ExecutionStrategy &,
                  const BuildStrategy &, ir::Graph *>())
      // NOTE: even we return a vec<Scope*>* to Python use reference policy.
      // We still cannot get local_scope from this vector, since the element
      // of vec<Scope*> will be freed by Python GC. We can only return Scope*
      // one by one and mark them as reference.
      .def("local_scopes",
           [](ParallelExecutor &self) -> std::vector<Scope *> * {
             return &self.GetLocalScopes();
           },
           py::return_value_policy::reference)
      .def("drop_local_exe_scopes", &ParallelExecutor::DropLocalExeScopes)
      .def("_need_create_local_exe_scopes",
           &ParallelExecutor::NeedCreateLocalExeScope)
      .def("feed_tensors_into_local_scopes",
           &ParallelExecutor::FeedTensorsIntoLocalScopes)
      .def("feed_and_split_tensor_into_local_scopes",
           &ParallelExecutor::FeedAndSplitTensorIntoLocalScopes)
      .def("run",
           [](ParallelExecutor &self,
              const std::vector<std::string> &fetch_tensors,
              bool return_merged) -> py::object {
             paddle::framework::FetchResultType ret;
             {
               pybind11::gil_scoped_release release;
               ret = self.Run(fetch_tensors, return_merged);
             }
             if (return_merged) {
               return py::cast(
                   std::move(BOOST_GET(paddle::framework::FetchList, ret)));
             } else {
               return py::cast(std::move(
                   BOOST_GET(paddle::framework::FetchUnmergedList, ret)));
             }
           })
      .def("device_count", &ParallelExecutor::DeviceCount);

#ifdef PADDLE_WITH_IPU
  py::class_<platform::ipu::IpuBackend,
             std::unique_ptr<platform::ipu::IpuBackend, py::nodelete>>(
      m, "IpuBackend")
      // manage IpuBackend in C++
      .def("get_instance",
           []() {
             return std::unique_ptr<platform::ipu::IpuBackend, py::nodelete>(
                 platform::ipu::IpuBackend::GetInstance());
           },
           py::return_value_policy::reference)
      .def("detach", &platform::ipu::IpuBackend::Detach)
      .def("reset", &platform::ipu::IpuBackend::Reset)
      .def("set_scope", &platform::ipu::IpuBackend::SetScope)
      .def("set_ipu_strategy", &platform::ipu::IpuBackend::SetIpuStrategy)
      .def("save_model_proto", &platform::ipu::IpuBackend::SaveModelProto);

  py::class_<platform::ipu::IpuStrategy>(m, "IpuStrategy")
      .def(py::init())
      .def("set_options",
           [](platform::ipu::IpuStrategy &self, const py::dict &opt) {
             for (auto element : opt) {
               auto option_name = element.first.cast<std::string>();
               VLOG(10) << "Set option: " << option_name;
               if (py::isinstance<py::bool_>(element.second)) {
                 self.AddBoolOption(option_name, element.second.cast<bool>());
               } else if (py::isinstance<py::float_>(element.second)) {
                 self.AddDoubleOption(option_name,
                                      element.second.cast<double>());
               } else if (py::isinstance<py::int_>(element.second)) {
                 self.AddUint64Option(option_name,
                                      element.second.cast<std::uint64_t>());
               } else if (py::isinstance<py::str>(element.second)) {
                 self.AddStringOption(option_name,
                                      element.second.cast<std::string>());
               } else if (py::isinstance<py::set>(element.second) ||
                          py::isinstance<py::list>(element.second)) {
                 for (auto option : element.second.cast<py::list>()) {
                   std::string option_val;
                   if (py::isinstance<py::str>(option)) {
                     option_val = option.cast<std::string>();
                   } else if (py::isinstance<py::int_>(option)) {
                     option_val = std::to_string(option.cast<std::uint64_t>());
                   } else {
                     PADDLE_THROW(platform::errors::Unimplemented(
                         "Failed to convert type: %s when set IpuStrategy "
                         "option: %s",
                         option.get_type(), option_name));
                   }
                   self.InsertStringOption(option_name, option_val);
                 }
               } else if (py::isinstance<py::dict>(element.second)) {
                 if (option_name.rfind("location_", 0) == 0) {
                   for (auto option : element.second.cast<py::dict>()) {
                     self.SetTensorLocation(
                         option_name, option.first.cast<std::string>(),
                         option.second.cast<std::uint64_t>());
                   }
                 } else if (option_name == "custom_op") {
                   std::string paddle_op;
                   std::string popart_op;
                   std::string domain;
                   int version = -1;
                   for (auto option : element.second.cast<py::dict>()) {
                     std::string option_key = option.first.cast<std::string>();
                     if (option_key == "paddle_op") {
                       paddle_op = option.second.cast<std::string>();
                     } else if (option_key == "popart_op") {
                       popart_op = option.second.cast<std::string>();
                     } else if (option_key == "domain") {
                       domain = option.second.cast<std::string>();
                     } else if (option_key == "version") {
                       version = option.second.cast<int>();
                     } else {
                       PADDLE_THROW(platform::errors::InvalidArgument(
                           "Invalid argument, key must be one of paddle_op, "
                           "popart_op, domain or version, but revecived %s",
                           option_key));
                     }
                   }
                   self.AddCustomOp(paddle_op, popart_op, domain, version);
                 } else {
                   for (auto option : element.second.cast<py::dict>()) {
                     std::string option_key = option.first.cast<std::string>();
                     std::string option_val;
                     if (py::isinstance<py::str>(option.second)) {
                       option_val = option.second.cast<std::string>();
                     } else if (py::isinstance<py::int_>(option.second)) {
                       option_val =
                           std::to_string(option.second.cast<std::uint64_t>());
                     } else {
                       PADDLE_THROW(platform::errors::Unimplemented(
                           "Failed to convert value type: %s when set "
                           "IpuStrategy option: %s",
                           option.second.get_type(), option_key));
                     }
                     self.InsertStringPairOption(option_name, option_key,
                                                 option_val);
                   }
                 }
               } else {
                 PADDLE_THROW(platform::errors::InvalidArgument(
                     "Invalid IpuStrategy option value type: %s, please check "
                     "input value for option: %s",
                     element.second.get_type(), option_name));
               }
             }
           })
      .def("get_option",
           [](platform::ipu::IpuStrategy &self, const std::string &name) {
             py::dict res;
             auto option_type = self.GetOptionType(name);
             res["name"] = name;
             res["type"] = option_type;
             if (option_type == "vector") {
               auto value = self.GetVectorOption(name);
               res["value"] = value;
             } else if (option_type == "map") {
               auto value = self.GetMapOption(name);
               res["value"] = value;
             } else {
               auto value_s = self.GetOption(name);
               res["value_s"] = value_s;
               if (option_type == "bool") {
                 res["value"] = static_cast<bool>(std::stoi(value_s));
               } else if (option_type == "uint64") {
                 res["value"] = std::stoul(value_s);
               } else if (option_type == "double") {
                 res["value"] = std::stod(value_s);
               } else if (option_type == "string") {
                 res["value"] = value_s;
               }
             }
             return res;
           })
      .def("get_all_option_names",
           &platform::ipu::IpuStrategy::GetAllOptionNames)
      .def("enable_pattern", &platform::ipu::IpuStrategy::EnablePattern)
      .def("disable_pattern", &platform::ipu::IpuStrategy::DisablePattern)
      .def("is_pattern_enabled", &platform::ipu::IpuStrategy::IsPatternEnabled);
#endif

  BindFleetWrapper(&m);
  BindIO(&m);

#if defined(PADDLE_WITH_PSLIB) && !defined(PADDLE_WITH_HETERPS)
  BindHeterWrapper(&m);
  BindMetrics(&m);
#endif
#ifdef PADDLE_WITH_HETERPS
  BindPSGPUWrapper(&m);
#endif
  BindGlooWrapper(&m);
  BindBoxHelper(&m);
#ifdef PADDLE_WITH_BOX_PS
  BindBoxWrapper(&m);
#endif
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  BindNCCLWrapper(&m);
#endif
#ifdef PADDLE_WITH_GLOO
  BindGlooContext(&m);
#endif
  BindGraph(&m);
  BindNode(&m);
  BindPass(&m);
  BindInferenceApi(&m);
  BindCompatible(&m);
  BindDataset(&m);
  BindGenerator(&m);
#ifndef PADDLE_ON_INFERENCE
  BindDistributed(&m);
#endif
#ifdef PADDLE_WITH_ASCEND
  BindAscendWrapper(&m);
  BindAscendGraph(&m);
  BindAscendDevice(&m);
#endif
#ifdef PADDLE_WITH_CRYPTO
  BindCrypto(&m);
#endif

#if defined PADDLE_WITH_PSCORE
  BindDistFleetWrapper(&m);
  BindPSHost(&m);
  BindCommunicatorContext(&m);
  BindDistCommunicator(&m);
  BindHeterClient(&m);
  BindGraphPyFeatureNode(&m);
  BindGraphNode(&m);
  BindGraphPyService(&m);
  BindGraphPyServer(&m);
  BindGraphPyClient(&m);
  BindIndexNode(&m);
  BindTreeIndex(&m);
  BindIndexWrapper(&m);
  BindIndexSampler(&m);
  BindSparseShardingTools(&m);
#endif
}
}  // namespace pybind
}  // namespace paddle
