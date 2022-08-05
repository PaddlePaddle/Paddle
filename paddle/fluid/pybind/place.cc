/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#endif

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/xpu_op_list.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
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
#include "paddle/fluid/pybind/place.h"
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
PyTypeObject *g_place_pytype = nullptr;
PyTypeObject *g_customplace_pytype = nullptr;
PyTypeObject *g_cudaplace_pytype = nullptr;
PyTypeObject *g_cpuplace_pytype = nullptr;
PyTypeObject *g_xpuplace_pytype = nullptr;
PyTypeObject *g_npuplace_pytype = nullptr;
PyTypeObject *g_cudapinnedplace_pytype = nullptr;
PyTypeObject *g_mluplace_pytype = nullptr;
PyTypeObject *g_ipuplace_pytype = nullptr;

template <typename PlaceType>
static inline int PlaceIndex(const PlaceType &p) {  // NOLINT
  return static_cast<int>(paddle::platform::Place(p).GetType());
}

template <typename PlaceType1, typename PlaceType2>
static inline bool IsSamePlace(const PlaceType1 &p1, const PlaceType2 &p2) {
  return paddle::platform::Place(p1) == paddle::platform::Place(p2);
}

void BindPlace(pybind11::module &m) {  // NOLINT
  using namespace paddle::framework;   // NOLINT
  py::class_<platform::CustomPlace> customplace(m,
                                                "CustomPlace",
                                                R"DOC(
    CustomPlace is a descriptor of a device.
    It represents a custom device on which a tensor will be allocated and a model will run.

    Examples:
        .. code-block:: python

          import paddle
          fake_cpu_place = paddle.CustomPlace("FakeCPU", 0)
                                             )DOC");
  g_customplace_pytype = reinterpret_cast<PyTypeObject *>(customplace.ptr());
  customplace
      .def("__init__",
           [](platform::CustomPlace &self,
              const std::string &device_type,
              int dev_id) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
             if (UNLIKELY(dev_id < 0)) {
               LOG(ERROR) << string::Sprintf(
                   "Invalid CustomPlace(%s, %d), device id must be 0 "
                   "or "
                   "positive integer",
                   device_type,
                   dev_id);
               std::exit(-1);
             }

             if (LIKELY(phi::DeviceManager::HasDeviceType(device_type) &&
                        phi::DeviceManager::IsCustom(device_type))) {
               int dev_count = static_cast<int>(
                   phi::DeviceManager::GetDeviceCount(device_type));
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
                       device_type,
                       dev_id,
                       dev_count,
                       device_type,
                       dev_count);
                   std::exit(-1);
                 }
               }
               new (&self) platform::CustomPlace(device_type, dev_id);
             } else {
               LOG(ERROR) << string::Sprintf(
                   "Invalid CustomPlace(%s, %d), the device type is "
                   "not registered "
                   "as a custom device.",
                   device_type,
                   dev_id);
               std::exit(-1);
             }
#else
             LOG(ERROR) << string::Sprintf(
                 "Cannot use CustomDevice because you have installed CPU/GPU"
                 "version PaddlePaddle.\n"
                 "If you want to use CustomDevice, please try to install"
                 "CustomDevice version "
                 "PaddlePaddle by: pip install paddlepaddle\n"
                 "If you only have CPU, please change "
                 "CustomPlace(%s, %d) to be CPUPlace().\n",
                 device_type, dev_id);
             std::exit(-1);
#endif
           })
      .def("_type", &PlaceIndex<platform::CustomPlace>)
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
                     dev_id,
                     platform::GetGPUDeviceCount(),
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
                     dev_id,
                     platform::GetXPUDeviceCount(),
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
#ifdef PADDLE_WITH_XPU_KP
  m.def("get_xpu_device_op_support_types",
        [](const std::string &op_name, phi::backends::xpu::XPUVersion version) {
          return platform::get_xpu_kp_op_support_type(op_name, version);
        });
#else
  m.def("get_xpu_device_op_support_types",
        [](const std::string &op_name, phi::backends::xpu::XPUVersion version) {
          return platform::get_xpu_op_support_type(op_name, version);
        });
#endif
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
                     dev_id,
                     platform::GetNPUDeviceCount(),
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
  py::class_<platform::IPUPlace> ipuplace(m, "IPUPlace", R"DOC(
    IPUPlace is a descriptor of a device.
    It represents a IPU device on which a tensor will be allocated and a model will run.

    Examples:
        .. code-block:: python
          import paddle

          # required: ipu

          ipu_place = paddle.IPUPlace()

        )DOC");
  g_ipuplace_pytype = reinterpret_cast<PyTypeObject *>(ipuplace.ptr());
  ipuplace
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
                     dev_id,
                     platform::GetMLUDeviceCount(),
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
      .def("_equals", &IsSamePlace<platform::Place, platform::CustomPlace>)
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
      .def("set_place",
           [](platform::Place &self, const platform::Place &other) {
             self = other;
           })
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
}

}  // namespace pybind
}  // namespace paddle
