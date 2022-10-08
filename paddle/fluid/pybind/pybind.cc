/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

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
#include <sstream>
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
#include "paddle/fluid/framework/new_executor/executor_statistics.h"
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
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/memory/allocation/cuda_ipc_allocator.h"
#endif
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/operators/ops_extra_info.h"
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
#include "paddle/fluid/platform/profiler/event_python.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/platform/profiler/profiler.h"
#include "paddle/fluid/pybind/cuda_streams_py.h"
#include "paddle/fluid/pybind/distributed_py.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/imperative.h"
#include "paddle/fluid/pybind/io.h"
#include "paddle/fluid/pybind/jit.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/lod_utils.h"
#include "paddle/utils/none.h"
#ifdef PADDLE_WITH_ASCEND
#include "paddle/fluid/pybind/ascend_wrapper_py.h"
#endif
#include "paddle/fluid/pybind/auto_parallel_py.h"
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
#include "paddle/fluid/pybind/inference_api.h"
#include "paddle/fluid/pybind/ir.h"
#include "paddle/fluid/pybind/metrics_py.h"
#include "paddle/fluid/pybind/ps_gpu_wrapper_py.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/phi/backends/device_manager.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/pybind/nccl_wrapper_py.h"
#endif
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/pybind/parallel_executor.h"
#include "paddle/fluid/pybind/place.h"
#include "paddle/fluid/pybind/protobuf.h"
#include "paddle/fluid/pybind/pybind.h"  // NOLINT
#include "paddle/fluid/pybind/reader_py.h"
#include "paddle/fluid/pybind/tensor.h"
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

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/fluid/operators/custom_device_common_op_registry.h"
#include "paddle/phi/capi/capi.h"
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

#ifdef PADDLE_WITH_CINN
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#endif

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"
#include "pybind11/stl.h"

DECLARE_bool(use_mkldnn);

// disable auto conversion to list in Python
PYBIND11_MAKE_OPAQUE(paddle::framework::LoDTensorArray);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchUnmergedList);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchList);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchType);

namespace paddle {
namespace pybind {

PyTypeObject *g_framework_scope_pytype = nullptr;
PyTypeObject *g_framework_lodtensorarray_pytype = nullptr;
PyTypeObject *g_custom_op_kernel_ctx_pytype = nullptr;

bool IsCompiledWithAVX() {
#ifndef PADDLE_WITH_AVX
  return false;
#else
  return true;
#endif
}

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

bool IsCompiledWithMPI() {
#ifdef PADDLE_WITH_MPI
  return true;
#else
  return false;
#endif
}

// NOTE some mpi lib can support cuda aware, support it in the future.
bool IsCompiledWithMPIAWARE() {
#ifdef PADDLE_WITH_MPI_AWARE
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
#else
  return true;
#endif
}

bool IsCompiledWithDIST() {
#ifdef PADDLE_WITH_DISTRIBUTE
  return true;
#else
  return false;
#endif
}

struct iinfo {
  int64_t min, max;
  int bits;
  std::string dtype;

  explicit iinfo(const framework::proto::VarType::Type &type) {
    switch (type) {
      case framework::proto::VarType::INT16:
        min = std::numeric_limits<int16_t>::min();
        max = std::numeric_limits<int16_t>::max();
        bits = 16;
        dtype = "int16";
        break;
      case framework::proto::VarType::INT32:
        min = std::numeric_limits<int32_t>::min();
        max = std::numeric_limits<int32_t>::max();
        bits = 32;
        dtype = "int32";
        break;
      case framework::proto::VarType::INT64:
        min = std::numeric_limits<int64_t>::min();
        max = std::numeric_limits<int64_t>::max();
        bits = 64;
        dtype = "int64";
        break;
      case framework::proto::VarType::INT8:
        min = std::numeric_limits<int8_t>::min();
        max = std::numeric_limits<int8_t>::max();
        bits = 8;
        dtype = "int8";
        break;
      case framework::proto::VarType::UINT8:
        min = std::numeric_limits<uint8_t>::min();
        max = std::numeric_limits<uint8_t>::max();
        bits = 8;
        dtype = "uint8";
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "the argument of paddle.iinfo can only be paddle.int8, "
            "paddle.int16, paddle.int32, paddle.int64, or paddle.uint8"));
        break;
    }
  }
};

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
        typeid(T).name(),
        obj->ob_type->tp_name));
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

static void inline CreateVariableIfNotExist(
    const py::handle &py_handle,
    const framework::Scope &scope,
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
            py_var_desc,
            platform::errors::InvalidArgument(
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
  PADDLE_ENFORCE_EQ(ops.empty(),
                    true,
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

PYBIND11_MODULE(libpaddle, m) {
  BindImperative(&m);
  BindEager(&m);
  BindEagerStringTensor(&m);
  BindCudaStream(&m);
  BindJit(&m);

  // Not used, just make sure cpu_info.cc is linked.
  paddle::platform::CpuTotalPhysicalMemory();

  paddle::memory::allocation::UseAllocatorStrategyGFlag();

  AssertStaticGraphAndDygraphGradMakerNoDiff();

  m.doc() = "C++ core of PaddlePaddle";

  // using framework in this function. Since it is inside a function, it will
  // not cause namespace pollution.
  using namespace paddle::framework;  // NOLINT

  BindException(&m);

  py::class_<iinfo>(m, "iinfo")
      .def(py::init<const framework::proto::VarType::Type &>())
      .def_readonly("min", &iinfo::min)
      .def_readonly("max", &iinfo::max)
      .def_readonly("bits", &iinfo::bits)
      .def_readonly("dtype", &iinfo::dtype)
      .def("__repr__", [](const iinfo &a) {
        std::ostringstream oss;
        oss << "paddle.iinfo(min=" << a.min;
        oss << ", max=" << a.max;
        oss << ", bits=" << a.bits;
        oss << ", dtype=" << a.dtype << ")";
        return oss.str();
      });

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
      .def_static("gen_new_memory_pool_id",
                  &platform::CUDAGraph::UniqueMemoryPoolID)
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
        dmt,
        platform::errors::InvalidArgument(
            "from_dlpack received an invalid capsule. "
            "Note that a DLPack tensor can be consumed only once."));

    PyCapsule_SetName(dltensor->ptr(), "used_dltensor");
    DLTensor dl = dmt->dl_tensor;
    phi::DenseTensor tensor;

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
        [](const py::handle &vec_var_list,
           const Scope &scope,
           const Executor *executor) {
          CreateVariableIfNotExist(vec_var_list, scope, executor);
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

  m.def(
      "broadcast_shape",
      [](const std::vector<int64_t> &x_dim, const std::vector<int64_t> &y_dim) {
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

  m.def(
      "_get_all_register_op_kernels",
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
              auto kernel_type_str = framework::KernelTypeToString(kernel_type);
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

  // NOTE(Aganlengzi): KernelFactory static instance is initialized BEFORE
  // plugins are loaded for custom kernels, but de-initialized AFTER they are
  // unloaded. We need manually clear symbols(may contain plugins' symbols)
  // stored in this static instance to avoid illegal memory access.
  m.def("clear_kernel_factory",
        []() { phi::KernelFactory::Instance().kernels().clear(); });
  m.def("clear_device_manager", []() {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    phi::DeviceManager::Clear();
#endif
  });

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

  py::class_<paddle::CustomOpKernelContext> custom_op_kernel_ctx(
      m, "CustomOpKernelContext", R"DOC()DOC");
  g_custom_op_kernel_ctx_pytype =
      reinterpret_cast<PyTypeObject *>(custom_op_kernel_ctx.ptr());
  custom_op_kernel_ctx.def(py::init<>())
      .def("add_inputs",
           [](paddle::CustomOpKernelContext &self, const py::handle &input) {
             PyObject *obj = input.ptr();
             if (PyList_Check(obj) || PyTuple_Check(obj)) {
               self.EmplaceBackInputs(
                   std::move(CastPyArg2VectorOfTensor(obj, 1)));
             } else {
               self.EmplaceBackInput(std::move(CastPyArg2Tensor(obj, 1)));
             }
           })
      .def("add_outputs",
           [](paddle::CustomOpKernelContext &self, py::handle &outputs) {
             PyObject *obj = outputs.ptr();
             if (PyList_Check(obj) || PyTuple_Check(obj)) {
               self.EmplaceBackOutputs(
                   std::move(CastPyArg2VectorOfTensor(obj, 1)));
             } else {
               self.EmplaceBackOutput(std::move(CastPyArg2Tensor(obj, 1)));
             }
           })
      .def("add_attr",
           [](paddle::CustomOpKernelContext &self, bool attr) {
             self.EmplaceBackAttr(attr);
           })
      .def("add_attr",
           [](paddle::CustomOpKernelContext &self, int attr) {
             self.EmplaceBackAttr(attr);
           })
      .def("add_attr",
           [](paddle::CustomOpKernelContext &self, float attr) {
             self.EmplaceBackAttr(attr);
           })
      .def("add_attr",
           [](paddle::CustomOpKernelContext &self, int64_t attr) {
             self.EmplaceBackAttr(attr);
           })
      .def("add_attr",
           [](paddle::CustomOpKernelContext &self, const std::string &attr) {
             self.EmplaceBackAttr(attr);
           })
      .def("add_attr",
           [](paddle::CustomOpKernelContext &self,
              const std::vector<int> &attr) { self.EmplaceBackAttr(attr); })
      .def("add_attr",
           [](paddle::CustomOpKernelContext &self,
              const std::vector<float> &attr) { self.EmplaceBackAttr(attr); })
      .def("add_attr",
           [](paddle::CustomOpKernelContext &self,
              const std::vector<int64_t> &attr) { self.EmplaceBackAttr(attr); })
      .def("add_attr",
           [](paddle::CustomOpKernelContext &self,
              const std::vector<std::string> &attr) {
             self.EmplaceBackAttr(attr);
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
      .def(
          "get_tensor",
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
      .def("set_vocab",
           [](Variable &self, Vocab vocab) {
             *self.GetMutable<Vocab>() = vocab;
           })
      .def(
          "get_string_tensor",
          [](Variable &self) { return self.GetMutable<Strings>(); },
          py::return_value_policy::reference)
      .def(
          "get_map_tensor",
          [](Variable &self) { return self.GetMutable<Vocab>(); },
          py::return_value_policy::reference)
      .def(
          "get_lod_rank_table",
          [](Variable &self) { return self.GetMutable<LoDRankTable>(); },
          py::return_value_policy::reference)
      .def(
          "get_selected_rows",
          [](Variable &self) -> phi::SelectedRows * {
            return self.GetMutable<phi::SelectedRows>();
          },
          py::return_value_policy::reference)
      .def(
          "get_lod_tensor_array",
          [](Variable &self) { return self.GetMutable<LoDTensorArray>(); },
          py::return_value_policy::reference)
      .def(
          "get_fetch_list",
          [](Variable &self) { return self.GetMutable<FetchList>(); },
          py::return_value_policy::reference)
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      .def(
          "get_communicator",
          [](Variable &self) -> platform::Communicator * {
            return self.GetMutable<platform::Communicator>();
          },
          py::return_value_policy::reference)
#endif
      .def(
          "get_reader",
          [](Variable &self) -> framework::ReaderHolder * {
            PADDLE_ENFORCE_EQ(self.IsType<framework::ReaderHolder>(),
                              true,
                              platform::errors::InvalidArgument(
                                  "The variable is not type of ReaderHolder."));
            return self.GetMutable<framework::ReaderHolder>();
          },
          py::return_value_policy::reference)
      .def(
          "get_scope",
          [](Variable &self) -> Scope * {
            auto scope_vec = self.GetMutable<std::vector<framework::Scope *>>();
            PADDLE_ENFORCE_GT(
                scope_vec->size(),
                0,
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

  py::class_<Scope> _Scope(m, "_Scope", R"DOC(
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

        )DOC");
  g_framework_scope_pytype = reinterpret_cast<PyTypeObject *>(_Scope.ptr());
  _Scope
      .def("_remove_from_pool",
           [](Scope &self) { ScopePool::Instance().Remove(&self); })
      .def(
          "var",
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
      .def("find_var",
           &Scope::FindVar,
           py::arg("name"),
           R"DOC(
           Find variable named :code:`name` in the current scope or
           its parent scope. Return None if not found.

           Args:
               name (str): the variable name.

           Returns:
               out (core.Variable|None): the found variable or None.
           )DOC",
           py::return_value_policy::reference)
      .def("size", &Scope::Size)
      .def("erase",
           &Scope::EraseVars,
           py::arg("names"),
           R"DOC(
           Find variable named :code:`name` in the current scope or
           its parent scope. Return None if not found.

           Args:
               name (str): the variable names to be erase.

           Returns:
               None
           )DOC",
           py::return_value_policy::reference)
      .def(
          "new_scope",
          [](Scope &self) -> Scope * { return &self.NewScope(); },
          R"DOC(
           Create a new sub-scope of the current scope.

           Returns:
               out (core._Scope): the created sub-scope.
           )DOC",
          py::return_value_policy::reference)
      .def("drop_kids",
           &Scope::DropKids,
           R"DOC(
           Delete all sub-scopes of the current scope.
           )DOC")
      .def("_kids", &Scope::kids)
      .def_property("_can_reuesd", &Scope::CanReuesd, &Scope::SetCanReuesd);

  m.def(
      "Scope",
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
            info.Proto().SerializeToString(&str),
            true,
            platform::errors::Fatal(
                "Serialize OpProto Error. This could be a bug of Paddle."));
        ret_values.emplace_back(str);
      }
    }
    return ret_values;
  });
  m.def(
      "get_all_op_names",
      [](const std::string &lib) {
        std::vector<std::string> op_names;
        for (auto &iter : OpInfoMap::Instance().map()) {
          op_names.emplace_back(iter.first);
        }
        if (lib == "phi") {
          std::vector<std::string> ops_with_phi_kernel;
          for (const auto &op_name : op_names) {
            if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(
                    op_name)) {
              ops_with_phi_kernel.emplace_back(op_name);
            }
          }
          return ops_with_phi_kernel;
        } else if (lib == "fluid") {
          std::vector<std::string> ops_with_fluid_kernel;
          auto all_fluid_op_kernels =
              paddle::framework::OperatorWithKernel::AllOpKernels();
          for (const auto &op_name : op_names) {
            if (all_fluid_op_kernels.find(op_name) !=
                all_fluid_op_kernels.end()) {
              ops_with_fluid_kernel.emplace_back(op_name);
            }
          }
          return ops_with_fluid_kernel;
        } else {
          return op_names;
        }
      },
      py::arg("lib") = "all",
      R"DOC(
      Return the operator names in paddle.

      Args:
          lib[string]: the library contains corresponding OpKernel, could be 'phi', 'fluid' and 'all'. Default value is 'all'.
  )DOC");
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
      "get_op_extra_attrs",
      [](const std::string &op_type)
          -> const paddle::framework::AttributeMap & {
        return operators::ExtraInfoUtils::Instance().GetExtraAttrsMap(op_type);
      });

  m.def(
      "get_attrtibute_type",
      [](const std::string &op_type,
         const std::string &attr_name) -> paddle::framework::proto::AttrType {
        const auto &defalut_val =
            operators::ExtraInfoUtils::Instance().GetExtraAttrsMap(op_type).at(
                attr_name);
        return static_cast<paddle::framework::proto::AttrType>(
            defalut_val.index() - 1);
      });
  m.def("get_grad_op_desc",
        [](const OpDesc &op_desc,
           const std::unordered_set<std::string> &no_grad_set,
           const std::vector<BlockDesc *> &grad_sub_block) {
          std::unordered_map<std::string, std::string> grad_to_var;
          std::vector<std::unique_ptr<OpDesc>> grad_op_descs =
              framework::OpInfoMap::Instance()
                  .Get(op_desc.Type())
                  .GradOpMaker()(
                      op_desc, no_grad_set, &grad_to_var, grad_sub_block);
          std::vector<OpDesc *> grad_op_desc_ptrs(grad_op_descs.size());
          std::transform(
              grad_op_descs.begin(),
              grad_op_descs.end(),
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
        [](const std::string op_type,
           const framework::VariableNameMap &inputs,
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
  m.def("prune",
        [](const ProgramDesc &origin,
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
  m.def(
      "prune_backward",
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
  m.def("get_serialize_comile_key", [](int64_t compilation_key) {
#ifdef PADDLE_WITH_CINN
    auto compiler = framework::paddle2cinn::CinnCompiler::GetInstance();
    auto s = compiler->SerializeKey(compilation_key);
    VLOG(4) << s;
    return s;
#else
    PADDLE_THROW(
                 platform::errors::PermissionDenied(
                 "Cannot get compilation key in non-CINN version, "
                 "Please recompile or reinstall Paddle with CINN support."));
#endif
  });
  m.def("empty_var_name",
        []() { return std::string(framework::kEmptyVarName); });
  m.def("grad_var_suffix",
        []() { return std::string(framework::kGradVarSuffix); });
  m.def_submodule(
       "var_names",
       "The module will return special predefined variable name in Paddle")
      .def("empty", []() { return kEmptyVarName; })
      .def("temp", []() { return kTempVarName; });

  py::class_<paddle::platform::DeviceContext>(m, "DeviceContext")
      .def_static("create",
                  [](paddle::platform::CPUPlace &place)
                      -> paddle::platform::DeviceContext * {
                    auto *context = new phi::CPUContext();
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
      .def_static(
          "create",
          [](paddle::platform::XPUPlace &place)
              -> paddle::platform::DeviceContext * {
#ifndef PADDLE_WITH_XPU
            PADDLE_THROW(platform::errors::PermissionDenied(
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
      .def_static(
          "create",
          [](paddle::platform::MLUPlace &place)
              -> paddle::platform::DeviceContext * {
#ifndef PADDLE_WITH_MLU
            PADDLE_THROW(platform::errors::PermissionDenied(
                "Cannot use MLUPlace in CPU/GPU version, "
                "Please recompile or reinstall Paddle with MLU support."));
#else
                    return new paddle::platform::MLUDeviceContext(place);
#endif
          })
      .def_static(
          "create",
          [](paddle::platform::NPUPlace &place)
              -> paddle::platform::DeviceContext * {
#ifndef PADDLE_WITH_ASCEND_CL
            PADDLE_THROW(platform::errors::PermissionDenied(
                "Cannot use NPUPlace in CPU/GPU/XPU version, "
                "Please recompile or reinstall Paddle with NPU support."));
#else
                return new paddle::platform::NPUDeviceContext(place);
#endif
          })
      .def_static("create",
                  [](paddle::platform::CustomPlace &place)
                      -> paddle::platform::DeviceContext * {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
                    PADDLE_THROW(platform::errors::PermissionDenied(
                        "Cannot use CustomPlace in CPU/GPU/XPU version, "
                        "Please recompile or reinstall Paddle with "
                        "CustomDevice support."));
#else
                return new paddle::platform::CustomDeviceContext(place);
#endif
                  })
      .def_static(
          "create",
          [](paddle::platform::CUDAPlace &place)
              -> paddle::platform::DeviceContext * {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
            PADDLE_THROW(platform::errors::PermissionDenied(
                "Cannot use CUDAPlace in CPU only version, "
                "Please recompile or reinstall Paddle with CUDA support."));
#else
      auto* context = new phi::GPUContext(place);
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
      context->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CUDAPinnedPlace())
          .get());
      context->PartialInitWithAllocator();
      return context;
#endif
          })
      .def_static(
          "create",
          [](paddle::platform::CUDAPinnedPlace &place)
              -> paddle::platform::DeviceContext * {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
            PADDLE_THROW(platform::errors::PermissionDenied(
                "Cannot use CUDAPinnedPlace in CPU only version, "
                "Please recompile or reinstall Paddle with CUDA support."));
#else
                  return new paddle::platform::CUDAPinnedDeviceContext(place);
#endif
          });
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  py::class_<platform::Communicator>(m, "Communicator").def(py::init<>());
#endif
  m.def("get_all_device_type", []() {
    std::vector<std::string> device_types;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    device_types = phi::DeviceManager::GetAllDeviceTypes();
#else
          VLOG(1) << string::Sprintf(
              "Cannot use get_all_device_type because you have installed"
              "CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_all_device_type, please try to install"
              "CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle\n");
#endif
    return device_types;
  });
  m.def("get_all_custom_device_type", []() {
    std::vector<std::string> device_types;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
#else
          VLOG(1) << string::Sprintf(
              "Cannot use get_all_custom_device_type because you have installed"
              "CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_all_custom_device_type, please try to "
              "install CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle\n");
#endif
    return device_types;
  });
  m.def("get_available_device", [] {
    std::vector<std::string> devices;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    devices = phi::DeviceManager::GetAllDeviceList();
#else
          VLOG(1) << string::Sprintf(
              "Cannot use get_available_device because you have installed"
              "CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_available_device, please try to install"
              "CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle\n");
#endif
    return devices;
  });
  m.def("get_available_custom_device", [] {
    std::vector<std::string> devices;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    devices = phi::DeviceManager::GetAllCustomDeviceList();
#else
          VLOG(1) << string::Sprintf(
              "Cannot use get_available_custom_device because you have "
              "installed"
              "CPU/GPU version PaddlePaddle.\n"
              "If you want to use get_available_custom_device, please try to "
              "install"
              "CustomDevice version "
              "PaddlePaddle by: pip install paddlepaddle\n");
#endif
    return devices;
  });

  py::class_<OperatorBase>(m, "Operator")
      .def_static("create",
                  [](py::bytes protobin) {
                    proto::OpDesc desc;
                    PADDLE_ENFORCE_EQ(desc.ParsePartialFromString(protobin),
                                      true,
                                      platform::errors::InvalidArgument(
                                          "Cannot parse user input to OpDesc"));
                    PADDLE_ENFORCE_EQ(desc.IsInitialized(),
                                      true,
                                      platform::errors::InvalidArgument(
                                          "The provided OpDesc is not "
                                          "initialized, the reason is: %s",
                                          desc.InitializationErrorString()));
                    return OpRegistry::CreateOp(desc);
                  })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const platform::CPUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const platform::XPUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const platform::NPUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const platform::CUDAPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const platform::CUDAPinnedPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const platform::MLUPlace &place) {
             pybind11::gil_scoped_release release;
             self.Run(scope, place);
           })
      .def("run",
           [](OperatorBase &self,
              const Scope &scope,
              const platform::CustomPlace &place) {
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
      .def(
          "get_worker_scope",
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
      .def("run_from_dataset",
           &Executor::RunFromDataset,
           py::call_guard<py::gil_scoped_release>())
      .def("release_trainer",
           &Executor::ReleaseTrainer,
           py::call_guard<py::gil_scoped_release>())
      .def("init_for_dataset",
           [](Executor &self,
              const ProgramDesc &prog,
              const std::string &trainer_desc,
              Scope *scope,
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
           [](Executor &self,
              ExecutorPrepareContext *ctx,
              Scope *scope,
              std::map<std::string, const LoDTensor *> *feed_targets,
              std::map<std::string, FetchType *> *fetch_targets,
              bool create_local_scope = true,
              bool create_vars = true,
              const std::string &feed_holder_name = "feed",
              const std::string &fetch_holder_name = "fetch") {
             pybind11::gil_scoped_release release;
             self.RunPreparedContext(ctx,
                                     scope,
                                     feed_targets,
                                     fetch_targets,
                                     create_local_scope,
                                     create_vars,
                                     feed_holder_name,
                                     fetch_holder_name);
           })
      .def("run_prepared_ctx",
           [](Executor &self,
              ExecutorPrepareContext *ctx,
              Scope *scope,
              bool create_local_scope = true,
              bool create_vars = true,
              bool keep_kids = false) {
             pybind11::gil_scoped_release release;
             self.RunPreparedContext(
                 ctx, scope, create_local_scope, create_vars, keep_kids);
           })
      .def("prepare",
           [](Executor &self,
              const ProgramDesc &program,
              int block_id,
              const std::vector<std::string> &skip_ref_cnt_vars =
                  std::vector<std::string>(),
              bool force_disable_gc = false) {
             pybind11::gil_scoped_release release;
             return self.Prepare(
                 program, block_id, skip_ref_cnt_vars, force_disable_gc);
           })
      .def("create_variables", &Executor::CreateVariables)
      .def("run",
           [](Executor &self,
              const ProgramDesc &prog,
              Scope *scope,
              int block_id,
              bool create_local_scope,
              bool create_vars,
              const std::vector<std::string> &fetch_vars) {
             pybind11::gil_scoped_release release;
             self.Run(prog,
                      scope,
                      block_id,
                      create_local_scope,
                      create_vars,
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
      .def(py::init<const platform::Place &, const ProgramDesc &>())
      .def("run",
           [](StandaloneExecutor &self,
              Scope *scope,
              std::vector<std::string> feed_names,
              std::vector<std::string> fetch_names) {
             paddle::framework::FetchList ret;
             {
               pybind11::gil_scoped_release release;
               ret = self.Run(scope, feed_names, fetch_names);
             }
             return py::cast(std::move(ret));
           })
      .def("dry_run",
           [](StandaloneExecutor &self,
              Scope *scope,
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
               cost_info = self.DryRun(scope, feed_names, feed_tensors);
             }
             return cost_info;
           });

  m.def("init_gflags", framework::InitGflags);
  m.def("init_glog", framework::InitGLOG);
  m.def("load_op_meta_info_and_register_op", [](const std::string dso_name) {
    egr::Controller::Instance().MergeOpMetaInfoMap(
        framework::LoadOpMetaInfoAndRegisterOp(dso_name));
  });
  m.def("init_devices", []() {
    framework::InitDevices();
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    for (auto &dev_type : phi::DeviceManager::GetAllCustomDeviceTypes()) {
      paddle::operators::RegisterCustomDeviceCommonKernel(dev_type);
    }
#endif
  });
  m.def("init_default_kernel_signatures",
        []() { framework::InitDefaultKernelSignatureMap(); });
  m.def("is_compiled_with_avx", IsCompiledWithAVX);
  m.def("is_compiled_with_cuda", IsCompiledWithCUDA);
  m.def("is_compiled_with_ascend", IsCompiledWithAscend);
  m.def("is_compiled_with_rocm", IsCompiledWithROCM);
  m.def("is_compiled_with_npu", IsCompiledWithNPU);
  m.def("is_compiled_with_ipu", IsCompiledWithIPU);
  m.def("is_compiled_with_xpu", IsCompiledWithXPU);
  m.def("is_compiled_with_mkldnn", IsCompiledWithMKLDNN);
  m.def("is_compiled_with_nccl", IsCompiledWithNCCL);
  m.def("is_compiled_with_mpi", IsCompiledWithMPI);
  m.def("is_compiled_with_mpi_aware", IsCompiledWithMPIAWARE);
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
  m.def("device_memory_stat_current_value",
        memory::DeviceMemoryStatCurrentValue);
  m.def("device_memory_stat_peak_value", memory::DeviceMemoryStatPeakValue);
  m.def(
      "run_cmd",
      [](const std::string &cmd,
         int time_out = -1,
         int sleep_inter = -1) -> const std::string {
        return paddle::framework::shell_get_command_output(
            cmd, time_out, sleep_inter);
      },
      py::arg("cmd"),
      py::arg("time_out") = -1,
      py::arg("sleep_inter") = -1);
  m.def(
      "shell_execute_cmd",
      [](const std::string &cmd,
         int time_out = 0,
         int sleep_inter = 0,
         bool redirect_stderr = false) -> std::vector<std::string> {
        return paddle::framework::shell_execute_cmd(
            cmd, time_out, sleep_inter, redirect_stderr);
      },
      py::arg("cmd"),
      py::arg("time_out") = 0,
      py::arg("sleep_inter") = 0,
      py::arg("redirect_stderr") = false);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  m.def("is_float16_supported", [](const platform::CUDAPlace &place) -> bool {
    // Only GPUs with Compute Capability >= 53 support float16
    return platform::GetGPUComputeCapability(place.device) >= 53;
  });
  m.def("is_bfloat16_supported", [](const platform::CUDAPlace &place) -> bool {
    // Only GPUs with Compute Capability >= 80 support bfloat16
    return platform::GetGPUComputeCapability(place.device) >= 80;
  });
#endif

  m.def("set_feed_variable",
        static_cast<void (*)(  // NOLINT
            Scope *,
            const LoDTensor &,
            const std::string &,
            size_t)>(&framework::SetFeedVariable));
  m.def("set_feed_variable",
        static_cast<void (*)(  // NOLINT
            Scope *,
            const Strings &,
            const std::string &,
            size_t)>(&framework::SetFeedVariable));
  m.def("get_fetch_variable",
        [](const Scope &scope,
           const std::string &var_name,
           size_t index) -> py::object {
          auto &var = framework::GetFetchVariable(scope, var_name, index);
          if (data_is_lod_tensor(var)) {
            return py::cast(PADDLE_GET(LoDTensor, var));
          } else {
            return py::cast(PADDLE_GET(LoDTensorArray, var));
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
  BindFleetExecutor(&m);
  BindTCPStore(&m);
  BindAutoParallel(&m);
  BindJitProperty(&m);

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
      .def(
          "__getitem__",
          [](LoDTensorArray &self, size_t i) { return &self.at(i); },
          py::return_value_policy::reference)
      .def("__len__", [](LoDTensorArray &self) { return self.size(); })
      .def("__setitem__",
           [](LoDTensorArray &self, size_t i, const LoDTensor &t) {
             PADDLE_ENFORCE_LT(i,
                               self.size(),
                               platform::errors::InvalidArgument(
                                   "The index to set is larger than the size "
                                   "of LoDTensorArray."));
             self[i].ShareDataWith(t);
             self[i].set_lod(t.lod());
           })
      .def(
          "append",
          [](LoDTensorArray &self, const LoDTensor &t) {
            self.emplace_back();
            self.back().ShareDataWith(t);
            self.back().set_lod(t.lod());
          },
          py::arg("tensor"),
          R"DOC(
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
      .def(
          "_move_to_list",
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
        vector of paddle::variant<LoDTensor, LoDTensorArray>.
        )DOC")
      .def(
          "_move_to_list",
          [](FetchList &self) -> py::list {
            py::list res(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
              if (data_is_lod_tensor(self[i])) {
                auto &data = PADDLE_GET(LoDTensor, self[i]);
                res[i] = py::cast(std::move(data));
              } else if (data_is_sparse_coo_tensor(self[i])) {
                auto &data = PADDLE_GET(phi::SparseCooTensor, self[i]);
                res[i] = py::cast(std::move(data));
              } else {
                auto &data = PADDLE_GET(LoDTensorArray, self[i]);
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

      .def(
          "append",
          [](FetchList &self, const LoDTensor &t) {
            self.emplace_back();
            auto &lod_tensor = PADDLE_GET(LoDTensor, self.back());
            lod_tensor.ShareDataWith(t);
            lod_tensor.set_lod(t.lod());
          },
          py::arg("var"))

      .def(
          "append",
          [](FetchList &self, const LoDTensorArray &t) {
            self.emplace_back();
            auto &lod_tensor_array = PADDLE_GET(LoDTensorArray, self.back());
            for (size_t i = 0; i < t.size(); ++i) {
              lod_tensor_array[i].ShareDataWith(t[i]);
              lod_tensor_array[i].set_lod(t[i].lod());
            }
          },
          py::arg("var"));

  py::class_<FetchUnmergedList>(m, "FetchUnmergedList", R"DOC(
        FetchUnmergedList is 2-D array of FetchType(paddle::variant(LoDTensor, LoDTensorArray)).
        )DOC")
      .def(
          "_move_to_list",
          [](FetchUnmergedList &self) -> py::list {
            py::list res(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
              py::list tmp(self[i].size());
              for (size_t j = 0; j < self[i].size(); ++j) {
                if (data_is_lod_tensor(self[i][j])) {
                  auto &var = PADDLE_GET(LoDTensor, self[i][j]);
                  tmp[j] = py::cast(std::move(var));
                } else {
                  auto &var = PADDLE_GET(LoDTensorArray, self[i][j]);
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
  m.def("get_cuda_current_device_id", &platform::GetCurrentDeviceId);
  m.def("cuda_empty_cache", [] {
    for (int dev_id : platform::GetSelectedDevices()) {
      auto *dev_ctx = platform::DeviceContextPool::Instance().GetByPlace(
          platform::CUDAPlace(dev_id));
      dev_ctx->cudnn_workspace_handle().ResetWorkspace();
    }
    platform::EmptyCache();
  });
  m.def(
      "get_device_properties",
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
        framework::ir::PassRegistry::Instance().Has(pass_type),
        false,
        platform::errors::AlreadyExists("Pass '%s' is registered more than "
                                        "once. Please use another name.",
                                        pass_type));
    callable.inc_ref();
    framework::ir::PassRegistry::Instance().Insert(
        pass_type, [pass_type, callable]() {
          py::gil_scoped_acquire guard;
          std::unique_ptr<framework::ir::Pass> pass(
              new framework::ir::GeneratePass(
                  py::cast<std::string>(callable())));
          return pass;
        });
  });
  m.def("get_pass", [](const std::string &pass_type) {
    auto pass = framework::ir::PassRegistry::Instance().Get(pass_type);
    return std::shared_ptr<framework::ir::Pass>(std::move(pass));
  });

  m.def("size_of_dtype", framework::SizeOfType);
  py::class_<paddle::platform::ProfilerResult>(m, "_ProfilerResult")
      .def(py::init<>())
      .def("get_data",
           &paddle::platform::ProfilerResult::GetData,
           py::return_value_policy::automatic_reference)
      .def("save", &paddle::platform::ProfilerResult::Save)
      .def("get_extra_info", &paddle::platform::ProfilerResult::GetExtraInfo)
      .def("get_version", &paddle::platform::ProfilerResult::GetVersion)
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def("get_span_indx", &paddle::platform::ProfilerResult::GetSpanIndx)
      .def("get_device_property",
           &paddle::platform::ProfilerResult::GetDeviceProperty);
#else
      .def("get_span_indx", &paddle::platform::ProfilerResult::GetSpanIndx);
#endif

  py::class_<paddle::platform::MemPythonNode>(m, "MemPythonNode")
      .def(py::init<>())
      .def_readwrite("timestamp_ns",
                     &paddle::platform::MemPythonNode::timestamp_ns)
      .def_readwrite("addr", &paddle::platform::MemPythonNode::addr)
      .def_readwrite("type", &paddle::platform::MemPythonNode::type)
      .def_readwrite("process_id", &paddle::platform::MemPythonNode::process_id)
      .def_readwrite("thread_id", &paddle::platform::MemPythonNode::thread_id)
      .def_readwrite("increase_bytes",
                     &paddle::platform::MemPythonNode::increase_bytes)
      .def_readwrite("place", &paddle::platform::MemPythonNode::place)
      .def_readwrite("current_allocated",
                     &paddle::platform::MemPythonNode::current_allocated)
      .def_readwrite("current_reserved",
                     &paddle::platform::MemPythonNode::current_reserved)
      .def_readwrite("peak_allocated",
                     &paddle::platform::MemPythonNode::peak_allocated)
      .def_readwrite("peak_reserved",
                     &paddle::platform::MemPythonNode::peak_reserved);

  py::class_<paddle::platform::DevicePythonNode>(m, "DevicePythonNode")
      .def(py::init<>())
      .def_readwrite("name", &paddle::platform::DevicePythonNode::name)
      .def_readwrite("type", &paddle::platform::DevicePythonNode::type)
      .def_readwrite("start_ns", &paddle::platform::DevicePythonNode::start_ns)
      .def_readwrite("end_ns", &paddle::platform::DevicePythonNode::end_ns)
      .def_readwrite("device_id",
                     &paddle::platform::DevicePythonNode::device_id)
      .def_readwrite("context_id",
                     &paddle::platform::DevicePythonNode::context_id)
      .def_readwrite("stream_id",
                     &paddle::platform::DevicePythonNode::stream_id)
      .def_readwrite("correlation_id",
                     &paddle::platform::DevicePythonNode::correlation_id)
      .def_readwrite("block_x", &paddle::platform::DevicePythonNode::block_x)
      .def_readwrite("block_y", &paddle::platform::DevicePythonNode::block_y)
      .def_readwrite("block_z", &paddle::platform::DevicePythonNode::block_z)
      .def_readwrite("grid_x", &paddle::platform::DevicePythonNode::grid_x)
      .def_readwrite("grid_y", &paddle::platform::DevicePythonNode::grid_y)
      .def_readwrite("grid_z", &paddle::platform::DevicePythonNode::grid_z)
      .def_readwrite("shared_memory",
                     &paddle::platform::DevicePythonNode::shared_memory)
      .def_readwrite("registers_per_thread",
                     &paddle::platform::DevicePythonNode::registers_per_thread)
      .def_readwrite("blocks_per_sm",
                     &paddle::platform::DevicePythonNode::blocks_per_sm)
      .def_readwrite("warps_per_sm",
                     &paddle::platform::DevicePythonNode::warps_per_sm)
      .def_readwrite("occupancy",
                     &paddle::platform::DevicePythonNode::occupancy)
      .def_readwrite("num_bytes",
                     &paddle::platform::DevicePythonNode::num_bytes)
      .def_readwrite("value", &paddle::platform::DevicePythonNode::value);

  py::class_<paddle::platform::HostPythonNode>(m, "HostPythonNode")
      .def(py::init<>())
      .def_readwrite("name", &paddle::platform::HostPythonNode::name)
      .def_readwrite("type", &paddle::platform::HostPythonNode::type)
      .def_readwrite("start_ns", &paddle::platform::HostPythonNode::start_ns)
      .def_readwrite("end_ns", &paddle::platform::HostPythonNode::end_ns)
      .def_readwrite("process_id",
                     &paddle::platform::HostPythonNode::process_id)
      .def_readwrite("thread_id", &paddle::platform::HostPythonNode::thread_id)
      .def_readwrite("correlation_id",
                     &paddle::platform::HostPythonNode::correlation_id)
      .def_readwrite("input_shapes",
                     &paddle::platform::HostPythonNode::input_shapes)
      .def_readwrite("dtypes", &paddle::platform::HostPythonNode::dtypes)
      .def_readwrite("callstack", &paddle::platform::HostPythonNode::callstack)
      .def_readwrite("children_node",
                     &paddle::platform::HostPythonNode::children_node_ptrs)
      .def_readwrite("runtime_node",
                     &paddle::platform::HostPythonNode::runtime_node_ptrs)
      .def_readwrite("device_node",
                     &paddle::platform::HostPythonNode::device_node_ptrs)
      .def_readwrite("mem_node",
                     &paddle::platform::HostPythonNode::mem_node_ptrs);

  py::class_<paddle::platform::Profiler>(m, "_Profiler")
      .def("create",
           &paddle::platform::Profiler::Create,
           py::return_value_policy::take_ownership)
      .def("is_cupti_supported", &paddle::platform::Profiler::IsCuptiSupported)
      .def("is_cnpapi_supported",
           &paddle::platform::Profiler::IsCnpapiSupported)
      .def("prepare",
           [](paddle::platform::Profiler *profiler) {
             platform::EnableHostEventRecorder();
             profiler->Prepare();
           })
      .def("start", &paddle::platform::Profiler::Start)
      .def(
          "stop",
          [](paddle::platform::Profiler *profiler) {
            platform::DisableHostEventRecorder();
            auto result = profiler->Stop();
            framework::StaticGraphExecutorPerfStatistics(
                result->GetNodeTrees());
            return result;
          },
          py::return_value_policy::automatic_reference);

  py::class_<paddle::platform::ProfilerOptions>(m, "ProfilerOptions")
      .def(py::init<>())
      .def_readwrite("trace_switch",
                     &paddle::platform::ProfilerOptions::trace_switch);

  py::class_<platform::RecordEvent>(m, "_RecordEvent")
      .def(py::init([](std::string name, platform::TracerEventType type) {
        return std::make_unique<platform::RecordEvent>(
            name, type, 1, paddle::platform::EventRole::kOrdinary);
      }))
      .def("end", [](platform::RecordEvent *event) { event->End(); });

  py::enum_<paddle::platform::TracerMemEventType>(m, "TracerMemEventType")
      .value("Allocate", paddle::platform::TracerMemEventType::Allocate)
      .value("Free", paddle::platform::TracerMemEventType::Free)
      .value("ReservedAllocate",
             paddle::platform::TracerMemEventType::ReservedAllocate)
      .value("ReservedFree",
             paddle::platform::TracerMemEventType::ReservedFree);

  py::enum_<paddle::platform::TracerEventType>(m, "TracerEventType")
      .value("Operator", paddle::platform::TracerEventType::Operator)
      .value("Dataloader", paddle::platform::TracerEventType::Dataloader)
      .value("ProfileStep", paddle::platform::TracerEventType::ProfileStep)
      .value("CudaRuntime", paddle::platform::TracerEventType::CudaRuntime)
      .value("Kernel", paddle::platform::TracerEventType::Kernel)
      .value("Memcpy", paddle::platform::TracerEventType::Memcpy)
      .value("Memset", paddle::platform::TracerEventType::Memset)
      .value("UserDefined", paddle::platform::TracerEventType::UserDefined)
      .value("OperatorInner", paddle::platform::TracerEventType::OperatorInner)
      .value("Forward", paddle::platform::TracerEventType::Forward)
      .value("Backward", paddle::platform::TracerEventType::Backward)
      .value("Optimization", paddle::platform::TracerEventType::Optimization)
      .value("Communication", paddle::platform::TracerEventType::Communication)
      .value("PythonOp", paddle::platform::TracerEventType::PythonOp)
      .value("PythonUserDefined",
             paddle::platform::TracerEventType::PythonUserDefined);
  m.def("load_profiler_result", &paddle::platform::LoadProfilerResult);
  m.def("enable_memory_recorder", &paddle::platform::EnableMemoryRecorder);
  m.def("disable_memory_recorder", &paddle::platform::DisableMemoryRecorder);
  m.def("enable_input_shape_recorder",
        &paddle::platform::EnableInputShapeRecorder);
  m.def("disable_input_shape_recorder",
        &paddle::platform::DisableInputShapeRecorder);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  m.def("set_cublas_switch", platform::SetAllowTF32Cublas);
  m.def("get_cublas_switch", platform::AllowTF32Cublas);
  m.def("set_cudnn_switch", platform::SetAllowTF32Cudnn);
  m.def("get_cudnn_switch", platform::AllowTF32Cudnn);
#endif  // PADDLE_WITH_CUDA
  m.def("clear_executor_cache", []() {
    pybind11::gil_scoped_release release;
    framework::ExecutorInfoCache::Instance().Finalize();
    framework::InterpreterCoreInfoCache::Instance().Finalize();
  });

  m.def("parse_safe_eager_deletion_skip_vars",
        paddle::framework::details::ParseSafeEagerDeletionSkipVarsSet);

#ifdef PADDLE_WITH_IPU
  py::class_<platform::ipu::IpuBackend,
             std::unique_ptr<platform::ipu::IpuBackend, py::nodelete>>(
      m, "IpuBackend")
      // manage IpuBackend in C++
      .def(
          "get_instance",
          []() {
            return std::unique_ptr<platform::ipu::IpuBackend, py::nodelete>(
                platform::ipu::IpuBackend::GetInstance());
          },
          py::return_value_policy::reference)
      .def("weights_to_host", &platform::ipu::IpuBackend::WeightsToHost)
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
               if (option_name == "compilation_progress_logger") {
                 self.SetCompilationProgressLogger(
                     element.second.cast<py::function>());
               } else if (py::isinstance<py::bool_>(element.second)) {
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
                         option.get_type(),
                         option_name));
                   }
                   self.InsertStringOption(option_name, option_val);
                 }
               } else if (py::isinstance<py::dict>(element.second)) {
                 if (option_name.rfind("location_", 0) == 0) {
                   for (auto option : element.second.cast<py::dict>()) {
                     self.SetTensorLocation(
                         option_name,
                         option.first.cast<std::string>(),
                         option.second.cast<std::uint64_t>());
                   }
                 } else if (option_name == "replicated_collectives_settings") {
                   for (auto option : element.second.cast<py::dict>()) {
                     self.SetReplicatedCollectivesSettings(
                         option.first.cast<std::string>(),
                         option.second.cast<bool>());
                   }
                 } else if (option_name == "accumulate_outer_fragment") {
                   for (auto option : element.second.cast<py::dict>()) {
                     std::vector<int> values;
                     for (auto value : option.second.cast<py::list>()) {
                       values.push_back(value.cast<int>());
                     }
                     self.SetAccumulateOuterFragmentSettings(
                         option.first.cast<std::uint64_t>(), values);
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
                           option.second.get_type(),
                           option_key));
                     }
                     self.InsertStringPairOption(
                         option_name, option_key, option_val);
                   }
                 }
               } else {
                 PADDLE_THROW(platform::errors::InvalidArgument(
                     "Invalid IpuStrategy option value type: %s, please check "
                     "input value for option: %s",
                     element.second.get_type(),
                     option_name));
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

  m.def("enable_autotune", [] {
    return phi::autotune::AutoTuneStatus::Instance().EnableAutoTune();
  });

  m.def("disable_autotune", [] {
    return phi::autotune::AutoTuneStatus::Instance().DisableAutoTune();
  });

  m.def("set_autotune_range", [](int64_t start, int64_t stop) {
    return phi::autotune::AutoTuneStatus::Instance().SetAutoTuneRange(start,
                                                                      stop);
  });

  m.def("update_autotune_status",
        [] { return phi::autotune::AutoTuneStatus::Instance().Update(); });

  m.def("autotune_status", [] {
    py::dict res;
    phi::autotune::AutoTuneCache::Instance().UpdateStatus();
    res["step_id"] = phi::autotune::AutoTuneStatus::Instance().StepID();
    res["cache_size"] = phi::autotune::AutoTuneCache::Instance().Size();
    res["cache_hit_rate"] =
        phi::autotune::AutoTuneCache::Instance().CacheHitRate();
    return res;
  });

  m.def("enable_layout_autotune",
        [] { return egr::Controller::Instance().EnableLayoutAutoTune(); });

  m.def("disable_layout_autotune",
        [] { return egr::Controller::Instance().DisableLayoutAutoTune(); });

  m.def("use_layout_autotune",
        [] { return egr::Controller::Instance().UseLayoutAutoTune(); });

  BindFleetWrapper(&m);
  BindIO(&m);
  BindParallelExecutor(m);
  BindPlace(m);
  BindTensor(m);

#if defined(PADDLE_WITH_PSLIB) && !defined(PADDLE_WITH_HETERPS)
  BindHeterWrapper(&m);
  BindMetrics(&m);
#endif
#ifdef PADDLE_WITH_HETERPS
  BindPSGPUWrapper(&m);
#ifdef PADDLE_WITH_PSLIB
  BindAfsWrapper(&m);
#endif
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
#ifndef PADDLE_NO_PYTHON
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
#ifdef PADDLE_WITH_HETERPS
  BindNodeQueryResult(&m);
  BindNeighborSampleQuery(&m);
  BindNeighborSampleResult(&m);
  BindGraphGpuWrapper(&m);
#endif
#endif
}
}  // namespace pybind
}  // namespace paddle
