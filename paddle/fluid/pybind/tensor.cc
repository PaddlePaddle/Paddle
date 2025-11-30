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
#if defined(__linux__)
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
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
#include "paddle/fluid/framework/dense_tensor_array.h"
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
#include "paddle/fluid/framework/new_executor/executor_statistics.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/prune.h"
#include "paddle/fluid/framework/scope_pool.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/phi/core/framework/reader.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/memory/allocation/allocator_strategy.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cuda_driver.h"
#include "paddle/phi/core/memory/allocation/cuda_ipc_allocator.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator.h"
#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"
#endif
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/profiler/event_python.h"
#include "paddle/fluid/platform/profiler/profiler.h"
#include "paddle/fluid/pybind/bind_cost_model.h"
#include "paddle/fluid/pybind/box_helper_py.h"
#include "paddle/fluid/pybind/communication.h"
#include "paddle/fluid/pybind/compatible.h"
#include "paddle/fluid/pybind/const_value.h"
#include "paddle/fluid/pybind/cuda_streams_py.h"
#include "paddle/fluid/pybind/data_set_py.h"
#include "paddle/fluid/pybind/distributed_py.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/fleet_wrapper_py.h"
#include "paddle/fluid/pybind/generator_py.h"
#include "paddle/fluid/pybind/global_value_getter_setter.h"
#include "paddle/fluid/pybind/gloo_context_py.h"
#include "paddle/fluid/pybind/gloo_wrapper_py.h"
#include "paddle/fluid/pybind/graph.h"
#include "paddle/fluid/pybind/heter_wrapper_py.h"
#include "paddle/fluid/pybind/imperative.h"
#include "paddle/fluid/pybind/inference_api.h"
#include "paddle/fluid/pybind/io.h"
#include "paddle/fluid/pybind/metrics_py.h"
#include "paddle/fluid/pybind/ps_gpu_wrapper_py.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/lod_utils.h"
#include "paddle/phi/core/memory/allocation/mmap_allocator.h"
#include "paddle/phi/core/platform/cpu_helper.h"
#include "paddle/phi/core/platform/device/device_wrapper.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/platform/monitor.h"
#include "paddle/phi/core/platform/profiler.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "paddle/utils/none.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/pybind/nccl_wrapper_py.h"
#endif
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/pybind/protobuf.h"
#include "paddle/fluid/pybind/pybind.h"  // NOLINT
#include "paddle/fluid/pybind/reader_py.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/utils/string/to_string.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#endif
#ifndef PADDLE_WITH_HIP
#include "paddle/phi/core/platform/device/gpu/cuda/cuda_profiler.h"
#endif
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif

#ifdef PADDLE_WITH_XPU
#include <cuda.h>
#include <cuda_runtime.h>
#include "paddle/fluid/pybind/xpu_streams_py.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/memory/allocation/xpu_ipc_allocator.h"
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#include "paddle/phi/core/platform/device/xpu/xpu_op_list.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/capi/capi.h"
#endif

#include "paddle/phi/core/platform/cuda_graph_with_memory_pool.h"

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"
#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#endif

#ifdef PADDLE_WITH_CRYPTO
#include "paddle/fluid/pybind/crypto.h"
#endif

#if defined PADDLE_WITH_PSCORE
#include "paddle/fluid/pybind/fleet_py.h"
#endif

#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/fluid/pybind/complex.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/tensor.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"
#include "pybind11/stl.h"

COMMON_DECLARE_bool(use_mkldnn);
COMMON_DECLARE_bool(use_onednn);
COMMON_DECLARE_bool(use_shm_cache);

// disable auto conversion to list in Python
PYBIND11_MAKE_OPAQUE(phi::TensorArray);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchUnmergedList);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchList);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchType);

namespace paddle::pybind {

PyTypeObject *g_framework_tensor_pytype = nullptr;

template <typename PlaceType>
static void TensorCopyFrom(phi::DenseTensor *dst,
                           const phi::DenseTensor &src,
                           const PlaceType &place,
                           int64_t batch_size) {
  if (batch_size < 0) {
    framework::TensorCopy(src, place, dst);
  } else {
    auto sliced = src.Slice(0, batch_size);
    framework::TensorCopy(sliced, place, dst);
  }
}

std::tuple<phi::DenseTensor, bool> HandleTensorCopy(
    const phi::DenseTensor &src,
    const std::optional<std::tuple<int, int>> dl_device,
    std::optional<bool> copy) {
  bool force_copy = copy.has_value() && copy.value();
  bool disallow_copy = copy.has_value() && !copy.value();

  phi::Place dst_place = src.place();
  if (dl_device.has_value()) {
    ::DLDeviceType dl_type =
        static_cast<::DLDeviceType>(std::get<0>(dl_device.value()));
    int dl_id = std::get<1>(dl_device.value());
    dst_place = framework::DLDeviceToPlace({dl_type, dl_id});
  }

  if (src.place() != dst_place && disallow_copy) {
    throw pybind11::buffer_error(
        "The src tensor is on a different device from the target "
        "device, so a copy will be performed. However, the user "
        "has set copy=False, which means that the user does not "
        "want to perform a copy operation. If you want to "
        "perform a copy operation, please set copy=True or "
        "copy=None.");
  }

  if (force_copy || src.place() != dst_place) {
    phi::Place ctx_place =
        src.place() != phi::CPUPlace() ? src.place() : dst_place;
    phi::DenseTensor dst(
        std::make_shared<phi::Allocation>(nullptr, 0, dst_place), src.meta());
    const auto *dev_ctx = phi::DeviceContextPool::Instance().Get(ctx_place);
    phi::Copy(*dev_ctx, src, dst_place, false, &dst);
    return std::make_tuple(dst, true);
  }

  return std::make_tuple(src, false);
}

template <typename T>
pybind11::capsule TensorToDLPack(
    const phi::DenseTensor &tensor,
    const std::optional<std::tuple<int, int>> dl_device = std::nullopt,
    std::optional<bool> copy = std::nullopt) {
  const auto [maybe_copied_tensor, is_copied] =
      HandleTensorCopy(tensor, dl_device, copy);
  uint64_t flags =
      static_cast<uint64_t>(is_copied) * DLPACK_FLAG_BITMASK_IS_COPIED;
  T *dlMTensor =
      framework::DLPackTraits<T>::ToDLPack(maybe_copied_tensor, flags);
  auto capsule = pybind11::capsule(
      static_cast<void *>(dlMTensor),
      framework::DLPackTraits<T>::capsule,
      [](PyObject *data) {
        if (!PyCapsule_IsValid(data, framework::DLPackTraits<T>::capsule)) {
          return;
        }
        T *dlMTensor = reinterpret_cast<T *>(
            PyCapsule_GetPointer(data, framework::DLPackTraits<T>::capsule));
        dlMTensor->deleter(dlMTensor);
      });
  return capsule;
}

void BindTensor(pybind11::module &m) {  // NOLINT
  using namespace paddle::framework;    // NOLINT
  py::class_<phi::DenseTensor> framework_tensor(
      m, "DenseTensor", py::buffer_protocol());
  g_framework_tensor_pytype =
      reinterpret_cast<PyTypeObject *>(framework_tensor.ptr());
  framework_tensor
      .def(
          // TODO(risemeup): Modify the logic of
          // TensorToPyArray() according to the dtype and copy
          // parameters.
          "__array__",
          [](phi::DenseTensor &self, py::object dtype, py::object copy) {
            return TensorToPyArray(self, copy);
          },
          py::arg("dtype") = py::none(),
          py::arg("copy") = py::none())
      .def("_ptr",
           [](const phi::DenseTensor &self) {
             return reinterpret_cast<uintptr_t>(self.data());
           })
      .def("_slice",
           [](phi::DenseTensor &self, int64_t begin_idx, int64_t end_idx) {
             if (!self.meta().is_contiguous()) {
               PADDLE_THROW(common::errors::InvalidArgument(
                   "Tensor is not contiguous, cannot call "
                   "_slice on it."));
             }
             return self.Slice(begin_idx, end_idx);
           })
      .def("_numel", &phi::DenseTensor::numel)
      .def("_is_initialized",
           [](const phi::DenseTensor &self) { return self.IsInitialized(); })
      .def("_get_dims",
           [](const phi::DenseTensor &self) {
             return common::vectorize(self.dims());
           })
      .def("_set_dims",
           [](phi::DenseTensor &self, const std::vector<int64_t> &dim) {
             self.Resize(common::make_ddim(dim));
           })
      .def("_set_layout",
           [](phi::DenseTensor &self, const std::string &layout) {
             self.set_layout(common::StringToDataLayout(layout));
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, phi::CustomPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, phi::GPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, phi::XPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, phi::CPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_double",
           [](phi::DenseTensor &self, phi::CPUPlace &place) {
             self.mutable_data<double>(place);
           })
      .def("_alloc_int",
           [](phi::DenseTensor &self, phi::CPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](phi::DenseTensor &self, phi::CustomPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](phi::DenseTensor &self, phi::XPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](phi::DenseTensor &self, phi::GPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](phi::DenseTensor &self, phi::GPUPinnedPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, phi::GPUPinnedPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              phi::CPUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, phi::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              phi::CustomPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, phi::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              phi::XPUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, phi::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              phi::GPUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, phi::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              phi::GPUPinnedPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, phi::TransToPhiDataType(type)));
           })
      .def("_clear", &phi::DenseTensor::clear)
      .def("_copy_from",
           &TensorCopyFrom<phi::CPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<phi::CustomPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<phi::XPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<phi::GPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<phi::GPUPinnedPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<phi::IPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<phi::Place>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("set",
           SetTensorFromPyArray<phi::CPUPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<phi::CustomPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<phi::XPUPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<phi::GPUPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<phi::IPUPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<phi::XPUPinnedPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<phi::GPUPinnedPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false,
           R"DOC(
        Set the data of Tensor on place with given numpy array.

        Args:
          array (numpy.ndarray): The shape where the DenseTensor is to be set.
          place (CPUPlace|CUDAPlace|XPUPlace|IPUPlace|CUDAPinnedPlace|XPUPinnedPlace): The place where the
          Tensor is to be set.
          zero_copy (bool, optional): Whether to share memory with the input numpy array.
          This parameter only works with CPUPlace. Default: False.

        Returns:
            None.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import numpy as np

                >>> t = paddle.framework.core.Tensor()
                >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
          )DOC")

      .def(
          "shape",
          [](phi::DenseTensor &self) { return common::vectorize(self.dims()); },
          R"DOC(
           Return the shape of Tensor.

           Returns:
               list[int]: The shape of Tensor.


           Examples:
                .. code-block:: python

                    >>> import paddle
                    >>> import numpy as np

                    >>> t = paddle.framework.core.Tensor()
                    >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
                    >>> print(t.shape())
                    [5, 30]
           )DOC")
      .def("_to_dlpack",
           TensorToDLPack<::DLManagedTensor>,
           py::arg("dl_device") = py::none(),
           py::arg("copy") = py::none())
      .def("_to_dlpack_versioned",
           TensorToDLPack<::DLManagedTensorVersioned>,
           py::arg("dl_device") = py::none(),
           py::arg("copy") = py::none())
      .def("_set_float_element", TensorSetElement<float>)
      .def("_get_float_element", TensorGetElement<float>)
      .def("_set_double_element", TensorSetElement<double>)
      .def("_get_double_element", TensorGetElement<double>)
      .def("_set_complex64_element", TensorSetElement<paddle::complex64>)
      .def("_get_complex64_element", TensorGetElement<paddle::complex64>)
      .def("_set_complex128_element", TensorSetElement<paddle::complex128>)
      .def("_get_complex128_element", TensorGetElement<paddle::complex128>)
      .def("_place", [](phi::DenseTensor &self) { return self.place(); })
#ifdef PADDLE_WITH_XPU
      .def("get_xpu_scale_value",
           [](phi::DenseTensor &self) {
             if (self.storage_properties_initialized()) {
               const phi::XPUStorageProperties &sp =
                   self.storage_properties<phi::XPUStorageProperties>();
               return sp.xpu_scale_value;
             } else {
               return phi::XPUStorageProperties::default_xpu_scale_value;
             }
           })
      .def("set_xpu_scale_value",
           [](phi::DenseTensor &self, float new_value) {
             std::unique_ptr<phi::StorageProperties> sp =
                 std::make_unique<phi::XPUStorageProperties>(new_value);
             self.set_storage_properties(std::move(sp));
           })
#endif
      .def("_dtype",
           [](phi::DenseTensor &self) {
             return framework::TransToProtoVarType(self.type());
           })
      .def("_layout",
           [](phi::DenseTensor &self) {
             return common::DataLayoutToString(self.layout());
           })
      .def("_share_data_with", &phi::DenseTensor::ShareDataWith)
      .def("_share_data_nocheck_with", &phi::DenseTensor::ShareDataNoCheckWith)
      .def("__getitem__", PySliceTensor, py::return_value_policy::reference)
      .def("__str__",
           [](const phi::DenseTensor &self) {
             std::stringstream ostr;
             ostr << self;
             return ostr.str();
           }) /* ------ End of original Tensor ------ */
      .def(py::init([](const std::vector<std::vector<size_t>>
                           &recursive_sequence_lengths) {
        LegacyLoD new_lod;
        new_lod.reserve(recursive_sequence_lengths.size());
        std::copy(recursive_sequence_lengths.begin(),
                  recursive_sequence_lengths.end(),
                  std::back_inserter(new_lod));
        LegacyLoD new_offset_lod = ConvertToOffsetBasedLegacyLoD(new_lod);
        PADDLE_ENFORCE_EQ(
            CheckLegacyLoD(new_offset_lod, -1),
            true,
            common::errors::InvalidArgument(
                "The provided recursive_sequence_lengths info is "
                "invalid, "
                "the LegacyLoD converted by recursive_sequence_lengths is %s",
                new_lod));
        return std::make_unique<phi::DenseTensor>(new_offset_lod);
      }))
      .def(py::init([]() { return std::make_unique<phi::DenseTensor>(); }))
      // We implement offset based LegacyLoD in C++ while we use length based
      // with Python API. The discussion is here:
      // https://github.com/PaddlePaddle/Paddle/issues/10855
      .def(
          "set_lod",
          [](phi::DenseTensor &self,
             const std::vector<std::vector<size_t>> &lod) {
            // the input lod is offset-based level-of-detail info
            LegacyLoD new_lod;
            new_lod.reserve(lod.size());
            std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
            PADDLE_ENFORCE_EQ(
                CheckLegacyLoD(new_lod, common::vectorize(self.dims()).front()),
                true,
                common::errors::InvalidArgument(
                    "The provided LegacyLoD is invalid, the LegacyLoD is %s",
                    new_lod));
            self.set_lod(new_lod);
          },
          py::arg("lod"),
          R"DOC(
           Set LegacyLoD of the Tensor.

           Args:
               lod (list[list[int]]): The lod to set.

           Returns:
                None.

           Examples:
                .. code-block:: python

                    >>> import paddle
                    >>> import numpy as np

                    >>> t = paddle.framework.core.Tensor()
                    >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
                    >>> t.set_lod([[0, 2, 5]])
                    >>> print(t.lod())
                    [[0, 2, 5]]
           )DOC")
      .def(
          "set_recursive_sequence_lengths",
          [](phi::DenseTensor &self,
             const std::vector<std::vector<size_t>>
                 &recursive_sequence_lengths) {
            // the input recursive_sequence_lengths is length-based
            // level-of-detail info
            LegacyLoD new_lod;
            new_lod.reserve(recursive_sequence_lengths.size());
            std::copy(recursive_sequence_lengths.begin(),
                      recursive_sequence_lengths.end(),
                      std::back_inserter(new_lod));
            LegacyLoD new_offset_lod = ConvertToOffsetBasedLegacyLoD(new_lod);
            PADDLE_ENFORCE_EQ(
                CheckLegacyLoD(new_offset_lod,
                               common::vectorize(self.dims()).front()),
                true,
                common::errors::InvalidArgument(
                    "The provided recursive_sequence_lengths info is "
                    "invalid, "
                    "the LegacyLoD converted by recursive_sequence_lengths is "
                    "%s",
                    new_lod));
            self.set_lod(new_offset_lod);
          },
          py::arg("recursive_sequence_lengths"),
          R"DOC(
           Set LegacyLoD of the Tensor according to recursive sequence lengths.

           For example, if recursive_sequence_lengths=[[2, 3]], which means
           there are two sequences with length 2 and 3 respectively, the
           corresponding lod would be [[0, 2, 2+3]], i.e., [[0, 2, 5]].

           Args:
                recursive_sequence_lengths (list[list[int]]): The recursive sequence lengths.

           Returns:
                None.

           Examples:
                .. code-block:: python

                    >>> import paddle
                    >>> import numpy as np

                    >>> t = paddle.framework.core.Tensor()
                    >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
                    >>> t.set_recursive_sequence_lengths([[2, 3]])
                    >>> print(t.recursive_sequence_lengths())
                    [[2, 3]]
                    >>> print(t.lod())
                    [[0, 2, 5]]
           )DOC")
      .def(
          "lod",
          [](phi::DenseTensor &self) -> std::vector<std::vector<size_t>> {
            // output the offset-based lod info
            LegacyLoD lod = self.lod();
            std::vector<std::vector<size_t>> new_lod;
            new_lod.reserve(lod.size());
            std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
            return new_lod;
          },
          R"DOC(
           Return the LegacyLoD of the Tensor.

           Returns:
               list[list[int]]: The lod of the Tensor.

           Examples:
                .. code-block:: python

                    >>> import paddle
                    >>> import numpy as np

                    >>> t = paddle.framework.core.Tensor()
                    >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
                    >>> t.set_lod([[0, 2, 5]])
                    >>> print(t.lod())
                    [[0, 2, 5]]
           )DOC")
      .def("_as_type",
           [](const phi::DenseTensor &self,
              paddle::framework::proto::VarType::Type type) {
             phi::DenseTensor dst;
             if (self.IsInitialized() && self.numel() > 0) {
               TransDataType(self, type, &dst);
             }
             return dst;
           })
      .def("_copy", [](const phi::DenseTensor &self, const phi::Place &place) {
        // follow fetch_op's implementation
        phi::DenseTensor dst;
        if (self.IsInitialized() && self.numel() > 0) {
          TensorCopySync(self, place, &dst);
        } else {
          // Not copy, if the src tensor is empty.
          dst.clear();
          dst.Resize({0});
        }
        return dst;
#ifdef _WIN32
      });
#else
           })
#ifdef PADDLE_WITH_CUDA
      .def("_share_vmm", [](phi::DenseTensor self) {
        PADDLE_ENFORCE_EQ(
            self.IsInitialized() && self.numel() > 0,
            true,
            common::errors::InvalidArgument(
                "Tensor must be initialized and contain elements before "
                "calling _share_vmm."));
        auto *holder = dynamic_cast<memory::allocation::Allocation *>(
          self.Holder().get());
        PADDLE_ENFORCE_EQ(
            phi::is_gpu_place(holder->place()),
            true,
            common::errors::InvalidArgument(
                "_share_vmm only supports tensors placed on GPU, but "
                "the current tensor is on %s.",
                holder->place()));
        paddle::memory::VmmTensorPartsVisitor parts_visitor(holder->ptr());
        paddle::memory::allocation::AllocatorFacade::Instance().Accept(
            holder->place(), &parts_visitor);
        PADDLE_ENFORCE_EQ(
            parts_visitor.Found(),
            true,
            common::errors::Unavailable(
                "Failed to locate VMM allocation metadata for tensor."));
        const auto& parts = parts_visitor.Parts();
        PADDLE_ENFORCE_GT(
            parts.size(),
            0,
            common::errors::Unavailable(
                "Cannot export VMM tensor because no VMM chunks were found."));
        const int &device_id = paddle::platform::GetCurrentDeviceId();
        auto stream =
            paddle::platform::get_current_stream(device_id);
        stream->Synchronize();

        using paddle::memory::allocation::VmmIpcHeader;
        using paddle::memory::allocation::VmmIpcEntry;
        VmmIpcHeader header{};
        header.version     = 1;
        header.flags       = 0x1;  // pidfd
        header.pid         = static_cast<uint32_t>(::getpid());
        header.num_entries = static_cast<uint32_t>(parts.size());
        header.alloc_size  = static_cast<uint64_t>(holder->size());
        header.offset = parts[0].chunk_rel_off;
        header.reserved_size = 0;
        for (const auto& p : parts) {
          header.reserved_size += p.chunk->size;
        }

        std::string blob;
        blob.reserve(sizeof(VmmIpcHeader) +
          parts.size() * (sizeof(VmmIpcEntry) + sizeof(int)));
        blob.resize(sizeof(VmmIpcHeader));
        std::memcpy(blob.data(), &header, sizeof(VmmIpcHeader));

        uint64_t rel_offset = 0;
        for (const auto& p : parts) {
          VmmIpcEntry entry{};
          entry.handle_type = 1;  // POSIX_FD
          entry.rel_offset  = rel_offset;
          entry.chunk_size     = p.chunk->size;
          entry.chunk_rel_off = p.chunk_rel_off;

          int fd = -1;
          auto chunk = p.chunk;
          PADDLE_ENFORCE_NOT_NULL(
              chunk,
              common::errors::InvalidArgument(
                  "Found an empty VMM chunk while exporting tensor."));
          VLOG(10) << "chunk handle="
                  << static_cast<int64_t>(chunk->handle)
                  << " device=" << chunk->device
                  << " chunk_size=" << p.chunk->size
                  << " rel_offset=" << rel_offset
                  << " chunk_rel_off=" << p.chunk_rel_off
                  << " len=" << p.len;
          PADDLE_ENFORCE_NE(
              p.chunk->handle,
              0,
              common::errors::InvalidArgument(
                  "VMM chunk handle must be non-zero when exporting tensor."));
          PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cuMemExportToShareableHandle(
              &fd, p.chunk->handle,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

          const size_t old_size = blob.size();
          blob.resize(old_size + sizeof(VmmIpcEntry) + sizeof(int));
          std::memcpy(blob.data() + old_size,
                      &entry, sizeof(VmmIpcEntry));
          std::memcpy(blob.data() + old_size + sizeof(VmmIpcEntry),
                      &fd, sizeof(int));

          rel_offset += p.chunk->size;
        }

        const int dtype_idx = static_cast<int>(self.type());
        return py::make_tuple(py::bytes(blob),
                              dtype_idx,
                              common::vectorize(self.dims()),
                              self.lod(),
                              device_id);
      })
      .def("_new_shared_vmm", [](py::tuple meta) {
// Fallback definitions for older glibc versions where SYS_pidfd_open and
// SYS_pidfd_getfd are not defined, even though the kernel may support them.
// These x86_64 syscall numbers (434, 438) are only used when the macros are
// missing; newer systems will use the definitions provided by glibc.
#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif
        PADDLE_ENFORCE_EQ(
            meta.size(),
            5,
            common::errors::InvalidArgument(
                "VMM IPC metadata must contain 5 elements, but received %d. "
                "Please make sure the tuple returned by _share_vmm is passed "
                "unchanged.",
                meta.size()));
        std::string blob = meta[0].cast<py::bytes>();
        int dtype_idx = meta[1].cast<int>();
        std::vector<int64_t> dims_vec = meta[2].cast<std::vector<int64_t>>();
        int device_id = meta[4].cast<int>();

        using paddle::memory::allocation::VmmIpcHeader;
        using paddle::memory::allocation::VmmIpcEntry;
        PADDLE_ENFORCE_GE(
            blob.size(),
            sizeof(VmmIpcHeader),
            common::errors::InvalidArgument(
                "Invalid VMM IPC payload: blob size %zu is smaller than header "
                "size %zu.",
                blob.size(),
                sizeof(VmmIpcHeader)));
        const VmmIpcHeader* header =
            reinterpret_cast<const VmmIpcHeader*>(blob.data());
        VLOG(10) << "[VMM-IPC] header: ver="
                 << static_cast<int>(header->version)
                 << " pid=" << header->pid
                 << " num_entries=" << header->num_entries
                 << " alloc_size=" << header->alloc_size
                 << " reserved_size=" << header->reserved_size
                 << " offset=" << header->offset;

        const int cur_dev = paddle::platform::GetCurrentDeviceId();
        VLOG(10) << "[VMM-IPC/import] device_id=" << device_id
                << " cur_dev=" << cur_dev;

        CUdeviceptr base = 0;
        PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cuMemAddressReserve(
            &base, header->reserved_size, 0, 0, 0));

        CUmemAccessDesc desc{};
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id   = device_id;
        desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        std::vector<CUmemGenericAllocationHandle> handles;
        handles.reserve(header->num_entries);

        int pidfd = static_cast<int>(::syscall(SYS_pidfd_open,
                                   (pid_t)header->pid, 0));
        PADDLE_ENFORCE_NE(
            pidfd,
            -1,
            common::errors::Unavailable(
                "pidfd_open failed while importing VMM tensor. errno=%d.",
                errno));
        size_t off = sizeof(VmmIpcHeader);
        for (uint32_t i = 0; i < header->num_entries; ++i) {
          PADDLE_ENFORCE_GE(
              blob.size() - off,
              sizeof(VmmIpcEntry),
              common::errors::InvalidArgument(
                  "Invalid VMM IPC payload: insufficient bytes for entry %u.",
                  i));
          const VmmIpcEntry* e =
              reinterpret_cast<const VmmIpcEntry*>(blob.data() + off);
          off += sizeof(VmmIpcEntry);

          // Only support FD(handle_type==1)
          PADDLE_ENFORCE_GE(
              blob.size() - off,
              sizeof(int),
              common::errors::InvalidArgument(
                  "Invalid VMM IPC payload: missing file descriptor for entry "
                  "%u.",
                  i));
          int remote_fd = *reinterpret_cast<const int*>(blob.data() + off);
          off += sizeof(int);

          int myfd = static_cast<int>(
            ::syscall(SYS_pidfd_getfd, pidfd, remote_fd, 0));
          PADDLE_ENFORCE_NE(
              myfd,
              -1,
              common::errors::Unavailable(
                  "pidfd_getfd failed while importing VMM tensor. errno=%d.",
                  errno));

          CUmemGenericAllocationHandle handle = 0;
          PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cuMemImportFromShareableHandle(
              &handle, reinterpret_cast<void*>(static_cast<intptr_t>(myfd)),
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
          handles.push_back(handle);

          CUmemAllocationProp prop{};
          PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cuMemGetAllocationPropertiesFromHandle(
              &prop, handle));
          VLOG(10) << "[VMM-IPC] prop.type=" << static_cast<int>(prop.type)
                   << " loc.type=" << static_cast<int>(prop.location.type)
                   << " loc.id=" << prop.location.id
                   << " requestedHandleTypes="
                   << static_cast<int>(prop.requestedHandleTypes);

          size_t gran = 0;
          PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cuMemGetAllocationGranularity(
              &gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

          // map + set access
          const size_t map_len = e->chunk_size;
          VLOG(10) << "[VMM-IPC] entry#" << i
                   << " map: va=["
                   << reinterpret_cast<void*>(base + e->rel_offset)
                   << ", "
                   << reinterpret_cast<void*>(base + e->rel_offset + map_len)
                   << ") offsetInHandle=" << e->chunk_rel_off
                   << " rel_off=" << e->rel_offset
                   << " map_len=" << map_len
                   << " (chunk_size=" << e->chunk_size
                   << ", gran=" << gran << ")";
          PADDLE_ENFORCE_EQ(static_cast<size_t>(base + e->rel_offset) % gran,
              0UL, "base + e->rel_offset not aligned");
          PADDLE_ENFORCE_EQ(map_len % gran, 0UL, "map_len not aligned");

          PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemMap(
              base + e->rel_offset, map_len, 0, handle, 0));

          PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemSetAccess(
              base + e->rel_offset, map_len, &desc, 1));
        }

        if (pidfd != -1) ::close(pidfd);

        auto keep = std::make_shared<memory::allocation::ImportedVmmMulti>();
        keep->base  = base;
        keep->reserved_size = header->reserved_size;
        keep->hs    = std::move(handles);

        auto alloc =
        std::make_unique<memory::allocation::VmmImportedAllocation>(
            reinterpret_cast<void*>(base + header->offset),
            header->alloc_size,
            phi::GPUPlace(device_id), keep);

        phi::DenseTensor tensor;
        tensor.Resize(phi::make_ddim(dims_vec));
        tensor.ResetHolder(std::move(alloc));
        tensor.set_type(static_cast<phi::DataType>(dtype_idx));
        return tensor;
      })
      .def("_share_buffer_with",
           [](phi::DenseTensor &self, const phi::DenseTensor src,
              py::tuple t) {
              if (!src.meta().is_contiguous()) {
                PADDLE_THROW(common::errors::InvalidArgument(
                    "Tensor is not contiguous, cannot call "
                    "share_buffer_with on it."));
              }
             auto *cuda_ipc_allocation =
                 dynamic_cast<memory::allocation::CudaIpcAllocation *>(
                     src.Holder().get());

             PADDLE_ENFORCE_NOT_NULL(
                 cuda_ipc_allocation,
                 common::errors::PreconditionNotMet(
                     "Tensor is not Cuda IPC shared tensor. "
                     "Now only Tensor shared by cuda ipc could use this "
                     "api."));

             size_t size = t[0].cast<size_t>();
             auto dtype =
                 static_cast<phi::DataType>(t[1].cast<int>());
             auto dims = common::make_ddim(t[2].cast<std::vector<int>>());
             auto device_id = t[4].cast<int>();

             auto shared_reader_holder =
                 std::make_shared<memory::allocation::Allocation>(
                     cuda_ipc_allocation->ptr(),
                     cuda_ipc_allocation->base_ptr(), size,
                     phi::GPUPlace(device_id));

             self.ResetHolderWithType(shared_reader_holder, dtype);
             self.Resize(dims);

             VLOG(6) << "Reconstructed tensor with buffer shared!";
           },
           R"DOC(
           Deserialize GPU Tensor for existed shared Cuda IPC tensor.

           Params:
               tensor: Shared Cuda IPC tensor.
               tuple: contains data size, data type,
                      tensor dims, lod information, device index.

       )DOC")
      .def("_share_cuda",
           [](phi::DenseTensor self) {
             if (!self.IsInitialized() || self.numel() == 0)
               throw std::runtime_error(
                   "Tensor not initialized or numel is 0.  could not pass "
                   "to shared memory. ");

             auto *holder = dynamic_cast<memory::allocation::Allocation *>(
                 self.Holder().get());
             PADDLE_ENFORCE_EQ(
                 phi::is_gpu_place(holder->place()), true,
                 common::errors::InvalidArgument(
                     "Tensor is not on GPU. share_cuda only support GPU "
                     "Tensor, share_filename is for CPU tensor."));

             void *base_ptr = holder->base_ptr();
             ptrdiff_t offset_bytes = reinterpret_cast<char *>(holder->ptr()) -
                                      reinterpret_cast<char *>(base_ptr);

             cudaIpcMemHandle_t handle;
             PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcGetMemHandle(&handle, base_ptr));

             auto _handle = py::bytes(reinterpret_cast<char *>(&handle),
                                      (py::ssize_t)CUDA_IPC_HANDLE_SIZE);

             // TODO(ZHUI): use cuda event, to avoid sync.
             const auto &device_id = paddle::platform::GetCurrentDeviceId();
             auto stream =
                 paddle::platform::get_current_stream(device_id);
             stream->Synchronize();

             int type_idx = static_cast<int>(self.type());
             size_t data_size =
                 self.numel() *
                 framework::SizeOfType(
                     framework::TransToProtoVarType(self.type()));

             return py::make_tuple(_handle,
                                   (py::size_t)offset_bytes,
                                   data_size,
                                   type_idx,
                                   common::vectorize(self.dims()),
                                   self.lod(),
                                   device_id);
           },
           R"DOC(
           Serialize GPU Tensor by cudaIpcMemHandle.

           Returns:
               tuple: contains handle, data size, data type,
                      tensor dims, lod information, device index.

           Examples:
                .. code-block:: python

                    >>> import paddle

                    >>> tensor = paddle.ones([3,3])
                    >>> metainfo = tensor.value().get_tensor()._share_cuda()
      )DOC")
      .def("_new_shared_cuda",
           [](py::tuple t) {
             if (t.size() != 7)
               throw std::runtime_error(
                   "Invalid Tensor meta info for shared cuda tensor!");

             // 1. Create a new C++ instance
             phi::DenseTensor tensor;

             // 2. Rebuild Allocation from handle
             const std::string &handle = t[0].cast<std::string>();
             ptrdiff_t offset_bytes = (ptrdiff_t)t[1].cast<int64_t>();
             auto device_id = t[6].cast<int>();
             auto base_ptr = memory::allocation::GetIpcBasePtr(handle);
             size_t size = t[2].cast<size_t>();
             void *dev = base_ptr.get();
             dev = reinterpret_cast<char *>(dev) + offset_bytes;

             auto shared_reader_holder =
                 std::make_shared<memory::allocation::CudaIpcAllocation>(
                     dev, size, device_id, std::move(base_ptr));

             // 3. Rebuild Tensor
             tensor.ResetHolderWithType(
                 shared_reader_holder,
                 static_cast<phi::DataType>(t[3].cast<int>()));
             tensor.Resize(common::make_ddim(
                 t[4].cast<std::vector<int64_t>>()));

             return tensor;
           },
           R"DOC(
           Deserialize GPU lod tensor from cudaIpcMemHandle.

           Params:
               tuple: contains handle, data size, data type,
                      tensor dims, lod information, device index.

           Examples:
                .. code-block:: python

                    >>> import paddle

                    >>> tensor = paddle.ones([3,3])
                    >>> metainfo = tensor.value().get_tensor()._share_cuda()
                    >>> tensor_from_shared = paddle.to_tensor(paddle.base.core.DenseTensor._new_shared_cuda(metainfo))
        )DOC")
#endif
#ifdef PADDLE_WITH_XPU
      .def("_share_buffer_with",
           [](phi::DenseTensor &self, const phi::DenseTensor src,
              py::tuple t) {
             if (!src.meta().is_contiguous()) {
               PADDLE_THROW(common::errors::InvalidArgument(
                   "Tensor is not contiguous, cannot call "
                   "share_buffer_with on it."));
             }

             // Get the current device ID.
             int dev_id = platform::GetXPUCurrentDeviceId();
             paddle::platform::SetXPUDeviceId(dev_id);
             VLOG(6) << "[DEBUG XPU] _share_buffer_with: current XPU device = "
                     << dev_id;

             auto *xpu_ipc_allocation =
                 dynamic_cast<memory::allocation::XpuIpcAllocation *>(
                     src.Holder().get());

             PADDLE_ENFORCE_NOT_NULL(
                 xpu_ipc_allocation,
                 common::errors::PreconditionNotMet(
                     "Tensor is not Xpu IPC shared tensor. Now only Tensor "
                     "shared by xpu ipc could use this api."));

             size_t size = t[0].cast<size_t>();
             auto dtype = static_cast<phi::DataType>(t[1].cast<int>());
             auto dims = common::make_ddim(
                 t[2].cast<std::vector<int>>());
             auto device_id = t[4].cast<int>();

             auto shared_reader_holder =
                 std::make_shared<memory::allocation::Allocation>(
                     xpu_ipc_allocation->ptr(),
                     xpu_ipc_allocation->base_ptr(), size,
                     phi::XPUPlace(device_id));

             self.ResetHolderWithType(shared_reader_holder, dtype);
             self.Resize(dims);

             VLOG(6) << "[DEBUG XPU] Reconstructed tensor with buffer shared!";
           },
           R"DOC(
           Deserialize XPU Tensor for existed shared Xpu IPC tensor.

           Params:
               tensor: Shared Xpu IPC tensor.
               tuple: contains data size, data type, tensor dims, lod
                      information, device index.
           )DOC")
      .def("_share_xpu",
           [](phi::DenseTensor &self) {
             if (!self.IsInitialized() || self.numel() == 0)
               throw std::runtime_error(
                   "Tensor not initialized or numel is 0. could not pass to "
                   "shared memory.");

             // Get the current device ID.
             int dev_id = platform::GetXPUCurrentDeviceId();
             paddle::platform::SetXPUDeviceId(dev_id);
             VLOG(6) << "[DEBUG XPU] _share_xpu: current XPU device = "
                     << dev_id;

             auto *holder = dynamic_cast<memory::allocation::Allocation *>(
                 self.Holder().get());
             PADDLE_ENFORCE_EQ(
                 phi::is_xpu_place(holder->place()), true,
                 common::errors::InvalidArgument(
                     "Tensor is not on XPU. share_xpu only supports XPU "
                     "tensors."));
             void *base_ptr = holder->base_ptr();
             ptrdiff_t offset_bytes = reinterpret_cast<char *>(holder->ptr()) -
                                      reinterpret_cast<char *>(base_ptr);
             VLOG(6) << "[DEBUG XPU] _share_xpu: base_ptr = " << base_ptr
                     << ", offset_bytes = " << offset_bytes;
             cudaIpcMemHandle_t handle;
             int ret = cudaIpcGetMemHandle(&handle, base_ptr);
             VLOG(6) << "[DEBUG XPU] _share_xpu: cudaIpcGetMemHandle returned: "
                     << ret;
             PADDLE_ENFORCE_XPU_SUCCESS(ret);
             // Use the correct size for the IPC handle.
             auto _handle = py::bytes(
                 reinterpret_cast<char *>(&handle),
                 (py::ssize_t)sizeof(cudaIpcMemHandle_t));
             VLOG(6) << "[DEBUG XPU] _share_xpu: IPC handle (bytes) = "
                     << _handle;
             const auto &device_id =
                 paddle::platform::GetXPUCurrentDeviceId();
             auto stream = paddle::platform::get_current_stream(device_id);
             xpu_wait(stream->raw_stream());
             int type_idx = static_cast<int>(self.type());
             size_t data_size = self.numel() *
                 framework::SizeOfType(
                     framework::TransToProtoVarType(self.type()));
             VLOG(6) << "[DEBUG XPU] _share_xpu: data_size = " << data_size;
             return py::make_tuple(_handle,
                                   (py::size_t)offset_bytes,
                                   data_size,
                                   type_idx,
                                   common::vectorize(self.dims()),
                                   self.lod(),
                                   device_id);
           },
           R"DOC(
           Serialize XPU Tensor by IPC.

           Returns:
               tuple: contains handle, offset, data size, data type,
                      tensor dims, lod information, and device id.
           )DOC")
      .def("_new_shared_xpu",
           [](py::tuple t) {
             if (t.size() != 7)
               throw std::runtime_error(
                   "Invalid Tensor meta info for shared xpu tensor!");

             // Get the current device ID.
             int dev_id = platform::GetXPUCurrentDeviceId();
             paddle::platform::SetXPUDeviceId(dev_id);
             VLOG(6) << "[DEBUG XPU] _new_shared_xpu: current XPU device = "
                     << dev_id;

             phi::DenseTensor tensor;
             const std::string &handle = t[0].cast<std::string>();
             ptrdiff_t offset_bytes = (ptrdiff_t)t[1].cast<int64_t>();
             auto device_id = t[6].cast<int>();
             VLOG(6) << "[DEBUG XPU] _new_shared_xpu: handle = " << handle
                     << ", offset_bytes = " << offset_bytes;
             auto base_ptr = memory::allocation::GetIpcBasePtr(handle);
             size_t size = t[2].cast<size_t>();
             void *dev = base_ptr.get();
             dev = reinterpret_cast<char *>(dev) + offset_bytes;
             auto shared_holder =
                 std::make_shared<memory::allocation::XpuIpcAllocation>(
                     dev, size, device_id, std::move(base_ptr));
             tensor.ResetHolderWithType(
                 shared_holder,
                 static_cast<phi::DataType>(t[3].cast<int>()));
             tensor.Resize(common::make_ddim(
                 t[4].cast<std::vector<int>>()));
             VLOG(6) << "[DEBUG XPU] _new_shared_xpu: Reshape tensor dims: "
                     << tensor.dims();
             return tensor;
           },
           R"DOC(
           Deserialize XPU Tensor from IPC.

           Params:
               tuple: contains handle, offset, data size, data type,
                      tensor dims, lod information, and device index.

           Returns:
               A new DenseTensor that shares memory via IPC.
           )DOC")
#endif  // PADDLE_WITH_XPU
      .def("_share_filename",
           [](phi::DenseTensor &self, bool use_file_descriptor) {
             if (!self.IsInitialized() || self.numel() == 0)
               throw std::runtime_error(
                   "Tensor not initialized or numel is 0. could not pass to "
                   "shared memory. ");

             auto holder = self.Holder();
             PADDLE_ENFORCE_EQ(
                 phi::is_cpu_place(holder->place()) ||
                     phi::is_cuda_pinned_place(holder->place()),
                 true, common::errors::InvalidArgument(
                           "Tensor is not on CPU. share_filename only "
                           "support CPU Tensor."));

             auto *mmap_allocation = dynamic_cast<
                 memory::allocation::RefcountedMemoryMapAllocation *>(
                 holder.get());
             // If the tensor is not shared, allocate memory map allocation.
             if (mmap_allocation == nullptr) {
               void *data_ptr = self.data();
               size_t data_size =
                   self.numel() *
                   framework::SizeOfType(
                       framework::TransToProtoVarType(self.type()));

               int flags = memory::allocation::MAPPED_SHAREDMEM |
                           memory::allocation::MAPPED_EXCLUSIVE;
               if (use_file_descriptor) {
                   flags = flags | memory::allocation::MAPPED_KEEPFD |
                           memory::allocation::MAPPED_UNLINK;
               }
               std::string handle = memory::allocation::GetIPCName();
               int find_id = -1;
               if (FLAGS_use_shm_cache) {
                 find_id = memory::allocation::MemoryMapAllocationPool::Instance().FindFromCache(flags, data_size); // NOLINT
               }
               if (find_id != -1) {
                 handle = memory::allocation::MemoryMapAllocationPool::Instance().GetById(find_id).file_name_; // NOLINT
               }
               int shared_fd = -1;
               auto shared_holder =
                   memory::allocation::AllocateRefcountedMemoryMapAllocation(
                       handle, shared_fd, flags, data_size, find_id);

               // copy data & reset holder
               if (phi::is_cuda_pinned_place(holder->place())) {
#ifdef PADDLE_WITH_CUDA
                 memory::Copy(phi::CPUPlace(), shared_holder->ptr(),
                              phi::GPUPinnedPlace(), data_ptr, data_size);
#endif
               } else {
                 memory::Copy(phi::CPUPlace(), shared_holder->ptr(),
                              phi::CPUPlace(), data_ptr, data_size);
               }
               self.ResetHolder(shared_holder);
               mmap_allocation = shared_holder.get();
             }
             int type_idx = static_cast<int>(self.type());

             return py::make_tuple(mmap_allocation->ipc_name(),
                                   mmap_allocation->shared_fd(),
                                   mmap_allocation->size(), type_idx,
                                   common::vectorize(self.dims()), self.lod(),
                                   use_file_descriptor);
           },
           R"DOC(
           Serialize CPU lod tensor in shared memory to tuple.
           If the tensor is not in shared memory, we will copy it first.

           Returns:
               tuple: contains ipc name, data size, data type,
                      tensor dims and lod information.

           Examples:
                .. code-block:: python

                    >>> import paddle

                    >>> tensor = paddle.ones([3,3])
                    >>> metainfo = tensor.value().get_tensor()._share_filename()
       )DOC")
      .def("_new_shared_filename",
           [](py::tuple t) {  // __setstate__
             if (t.size() != 7)
               throw std::runtime_error("Invalid Tensor meta info state!");

             phi::DenseTensor tensor;

             // 2. Rebuild Allocation
             const std::string &ipc_name = t[0].cast<std::string>();
             const int shared_fd = t[1].cast<int>();
             const bool use_file_descriptor = t[6].cast<bool>();

             size_t size = t[2].cast<size_t>();
             int flags = memory::allocation::MAPPED_SHAREDMEM |
                         memory::allocation::MAPPED_NOCREATE;
             if (use_file_descriptor) {
                 flags = flags | memory::allocation::MAPPED_KEEPFD |
                         memory::allocation::MAPPED_UNLINK;
             }
             int find_id = -1;
             if (FLAGS_use_shm_cache) {
               find_id = memory::allocation::MemoryMapAllocationPool::Instance().FindFromCache(flags, size, ipc_name, /*check_refcount*/ false); // NOLINT
             }
             auto shared_holder =
                 memory::allocation::AllocateRefcountedMemoryMapAllocation(
                     ipc_name, shared_fd, flags, size, find_id);

             // 3. Rebuild Tensor
             tensor.ResetHolderWithType(
                 shared_holder,
                 static_cast<phi::DataType>(t[3].cast<int>()));
             tensor.Resize(common::make_ddim(t[4].cast<std::vector<int>>()));

             return tensor;
           },
           R"DOC(
           Deserialize CPU lod tensor from shared memory.

           Params:
               tuple: contains ipc file name, data size, data type,
                      tensor dims and lod information.

           Examples:
                .. code-block:: python

                    >>> import paddle

                    >>> tensor = paddle.ones([3,3])
                    >>> metainfo = tensor.value().get_tensor()._share_filename()
                    >>> tensor_from_shared = paddle.to_tensor(paddle.base.core.DenseTensor._new_shared_filename(metainfo))
        )DOC")
      .def("_shared_incref",
           [](phi::DenseTensor &self) {
             auto *mmap_allocation = dynamic_cast<
                 memory::allocation::RefcountedMemoryMapAllocation *>(
                 self.Holder().get());
             if (mmap_allocation) {
               mmap_allocation->incref();
             }
           },
           R"DOC(
            Increase reference count of share_filename tensor.
      )DOC")
      .def("_shared_decref",
           [](phi::DenseTensor &self) {
             auto *mmap_allocation = dynamic_cast<
                 memory::allocation::RefcountedMemoryMapAllocation *>(
                 self.Holder().get());
             if (mmap_allocation) {
               mmap_allocation->decref();
             }
           },
           R"DOC(
            Decrease reference count of share_filename tensor.
      )DOC")
      .def(py::pickle(
          [](const phi::DenseTensor &t) {  // __getstate__
            auto holder = t.Holder();
            PADDLE_ENFORCE_EQ(phi::is_cpu_place(holder->place()), true,
                              common::errors::PreconditionNotMet(
                                  "Tensor is not on CPU."
                                  "Now only Tensor on CPU can be serialized."));
            auto *mmap_writer_allocation =
                dynamic_cast<memory::allocation::MemoryMapWriterAllocation *>(
                    holder.get());
            PADDLE_ENFORCE_NOT_NULL(
                mmap_writer_allocation,
                common::errors::PreconditionNotMet(
                    "Tensor is not in shared memory."
                    "Now only Tensor on shared memory can be serialized."));
            int type_idx = static_cast<int>(t.type());

            return py::make_tuple(mmap_writer_allocation->ipc_name(),
                                  mmap_writer_allocation->size(), type_idx,
                                  common::vectorize(t.dims()), t.lod());
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 5)
              throw std::runtime_error("Invalid Tensor state!");

            // 1. Create a new C++ instance
            phi::DenseTensor tensor;

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
                static_cast<phi::DataType>(t[2].cast<int>()));
            tensor.Resize(common::make_ddim(t[3].cast<std::vector<int>>()));

            return tensor;
          }));
#endif

#ifdef PADDLE_WITH_DISTRIBUTE
  using phi::distributed::DistTensor;
  py::class_<DistTensor>(m, "DistTensor")
      .def(
          "get_tensor",
          [](DistTensor &self) { return self.value(); },
          py::return_value_policy::reference)
      .def("numel",
           [](DistTensor &self) -> int64_t { return self.value().numel(); })
      .def("set",
           [](DistTensor &self, const DistTensor &src) {
             self.unsafe_mutable_value()->ShareDataWith(src.value());
             return self;
           })
      .def("_share_data_nocheck_with",
           [](DistTensor &self, const DistTensor &src) {
             self.unsafe_set_dims(src.dims());
             self.unsafe_set_dist_attr(src.dist_attr());
             self.unsafe_mutable_value()->ShareDataNoCheckWith(src.value());
             return self;
           })
      .def("_numel",
           [](DistTensor &self) -> int64_t { return self.value().numel(); })
      .def("_share_data_with",
           [](DistTensor &self, const DistTensor &src) {
             self.unsafe_set_dims(src.dims());
             self.unsafe_set_dist_attr(src.dist_attr());
             if (!IsCurRankInMesh(self.process_mesh()) &&
                 !IsCurRankInMesh(src.dist_attr().process_mesh())) {
               self.unsafe_mutable_value()->ShareDataNoCheckWith(src.value());
             } else {
               self.unsafe_mutable_value()->ShareDataWith(src.value());
             }
             return self;
           })
      .def("_unsafe_set_skip_check_mesh",
           &DistTensor::unsafe_set_skip_check_mesh)
      .def("_clear", &DistTensor::clear);
#endif

  py::class_<phi::SelectedRows>(m, "SelectedRows")
      .def(py::init([]() { return std::make_unique<phi::SelectedRows>(); }))
      .def(py::init([](const std::vector<int64_t> rows, const int64_t &height) {
        return std::make_unique<phi::SelectedRows>(rows, height);
      }))
      .def(
          "get_tensor",
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
        std::vector<int64_t> new_rows(rows);
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

  py::class_<phi::SparseCooTensor>(m, "SparseCooTensor")
      .def(py::init([]() { return std::make_unique<phi::SparseCooTensor>(); }))
      .def("numel",
           [](const phi::SparseCooTensor &self) -> int64_t {
             return self.numel();
           })
      .def("indices", [](const phi::SparseCooTensor &self) -> phi::DenseTensor {
        return self.indices();
      });
}

}  // namespace paddle::pybind
