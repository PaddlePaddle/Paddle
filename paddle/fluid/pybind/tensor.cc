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
#include "paddle/phi/backends/cpu/cpu_info.h"
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
#include "paddle/fluid/pybind/tensor.h"
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

void BindTensor(pybind11::module &m) {  // NOLINT
  using namespace paddle::framework;    // NOLINT
  py::class_<phi::DenseTensor> framework_tensor(
      m, "Tensor", py::buffer_protocol());
  g_framework_tensor_pytype =
      reinterpret_cast<PyTypeObject *>(framework_tensor.ptr());
  framework_tensor
      .def("__array__",
           [](phi::DenseTensor &self) { return TensorToPyArray(self); })
      .def("_ptr",
           [](const phi::DenseTensor &self) {
             return reinterpret_cast<uintptr_t>(self.data());
           })
      .def("_slice", &phi::DenseTensor::Slice)
      .def("_numel", &phi::DenseTensor::numel)
      .def("_is_initialized",
           [](const phi::DenseTensor &self) { return self.IsInitialized(); })
      .def("_get_dims",
           [](const phi::DenseTensor &self) { return vectorize(self.dims()); })
      .def("_set_dims",
           [](phi::DenseTensor &self, const std::vector<int64_t> &dim) {
             self.Resize(phi::make_ddim(dim));
           })
      .def("_set_layout",
           [](phi::DenseTensor &self, const std::string &layout) {
             self.set_layout(phi::StringToDataLayout(layout));
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, paddle::platform::CustomPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, paddle::platform::CUDAPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, paddle::platform::XPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, paddle::platform::NPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](phi::DenseTensor &self, paddle::platform::MLUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_double",
           [](phi::DenseTensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<double>(place);
           })
      .def("_alloc_int",
           [](phi::DenseTensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](phi::DenseTensor &self, paddle::platform::CustomPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](phi::DenseTensor &self, paddle::platform::XPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](phi::DenseTensor &self, paddle::platform::CUDAPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](phi::DenseTensor &self, paddle::platform::MLUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def(
          "_alloc_int",
          [](phi::DenseTensor &self, paddle::platform::CUDAPinnedPlace &place) {
            self.mutable_data<int>(place);
          })
      .def(
          "_alloc_float",
          [](phi::DenseTensor &self, paddle::platform::CUDAPinnedPlace &place) {
            self.mutable_data<float>(place);
          })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              paddle::platform::CPUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              paddle::platform::CustomPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              paddle::platform::XPUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              paddle::platform::CUDAPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              paddle::platform::CUDAPinnedPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              paddle::platform::MLUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_clear", &phi::DenseTensor::clear)
      .def("_mutable_data",
           [](phi::DenseTensor &self,
              paddle::platform::NPUPlace &place,
              paddle::framework::proto::VarType::Type type) {
             return reinterpret_cast<uintptr_t>(
                 self.mutable_data(place, framework::TransToPhiDataType(type)));
           })
      .def("_copy_from",
           &TensorCopyFrom<paddle::platform::CPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<paddle::platform::CustomPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<paddle::platform::XPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<paddle::platform::CUDAPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<paddle::platform::NPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<paddle::platform::CUDAPinnedPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<paddle::platform::MLUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<paddle::platform::IPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("_copy_from",
           &TensorCopyFrom<paddle::platform::Place>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("batch_size") = -1)
      .def("set",
           SetTensorFromPyArray<paddle::platform::CPUPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<paddle::platform::CustomPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<paddle::platform::XPUPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<paddle::platform::CUDAPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<paddle::platform::NPUPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<paddle::platform::IPUPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<paddle::platform::MLUPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false)
      .def("set",
           SetTensorFromPyArray<paddle::platform::CUDAPinnedPlace>,
           py::arg("array"),
           py::arg("place"),
           py::arg("zero_copy") = false,
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

      .def(
          "shape",
          [](phi::DenseTensor &self) { return vectorize(self.dims()); },
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
           [](phi::DenseTensor &self) {
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
      .def("_place", [](phi::DenseTensor &self) { return self.place(); })
      .def("_dtype",
           [](phi::DenseTensor &self) {
             return framework::TransToProtoVarType(self.type());
           })
      .def("_layout",
           [](phi::DenseTensor &self) {
             return phi::DataLayoutToString(self.layout());
           })
      .def("_share_data_with", &phi::DenseTensor::ShareDataWith)
      .def("__getitem__", PySliceTensor, py::return_value_policy::reference)
      .def("__str__",
           [](const phi::DenseTensor &self) {
             std::stringstream ostr;
             ostr << self;
             return ostr.str();
           }) /* ------ End of original Tensor ------ */
      .def("__init__",
           [](phi::DenseTensor &instance,
              const std::vector<std::vector<size_t>>
                  &recursive_sequence_lengths) {
             LoD new_lod;
             new_lod.reserve(recursive_sequence_lengths.size());
             std::copy(recursive_sequence_lengths.begin(),
                       recursive_sequence_lengths.end(),
                       std::back_inserter(new_lod));
             LoD new_offset_lod = ConvertToOffsetBasedLoD(new_lod);
             PADDLE_ENFORCE_EQ(
                 CheckLoD(new_offset_lod, -1),
                 true,
                 platform::errors::InvalidArgument(
                     "The provided recursive_sequence_lengths info is "
                     "invalid, "
                     "the LoD converted by recursive_sequence_lengths is %s",
                     new_lod));
             new (&instance) phi::DenseTensor(new_offset_lod);
           })
      .def("__init__",
           [](phi::DenseTensor &instance) {
             new (&instance) phi::DenseTensor();
           })
      // We implement offset based LOD in C++ while we use length based with
      // Python API. So we changed set_lod to set_recursive_sequence_lengths
      // to
      // avoid misuse.
      // The discussion is here:
      // https://github.com/PaddlePaddle/Paddle/issues/10855
      .def(
          "set_lod",
          [](phi::DenseTensor &self,
             const std::vector<std::vector<size_t>> &lod) {
            // the input lod is offset-based level-of-detail info
            LoD new_lod;
            new_lod.reserve(lod.size());
            std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
            PADDLE_ENFORCE_EQ(
                CheckLoD(new_lod, vectorize(self.dims()).front()),
                true,
                platform::errors::InvalidArgument(
                    "The provided LoD is invalid, the LoD is %s", new_lod));
            self.set_lod(new_lod);
          },
          py::arg("lod"),
          R"DOC(
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
      .def(
          "set_recursive_sequence_lengths",
          [](phi::DenseTensor &self,
             const std::vector<std::vector<size_t>>
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
                CheckLoD(new_offset_lod, vectorize(self.dims()).front()),
                true,
                platform::errors::InvalidArgument(
                    "The provided recursive_sequence_lengths info is "
                    "invalid, "
                    "the LoD converted by recursive_sequence_lengths is "
                    "%s",
                    new_lod));
            self.set_lod(new_offset_lod);
          },
          py::arg("recursive_sequence_lengths"),
          R"DOC(
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
      .def(
          "lod",
          [](phi::DenseTensor &self) -> std::vector<std::vector<size_t>> {
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
      .def(
          "recursive_sequence_lengths",
          [](phi::DenseTensor &self) -> std::vector<std::vector<size_t>> {
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
      .def(
          "has_valid_recursive_sequence_lengths",
          [](phi::DenseTensor &self) -> bool {
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
           [](const phi::DenseTensor &self,
              paddle::framework::proto::VarType::Type type) {
             phi::DenseTensor dst;
             if (self.IsInitialized() && self.numel() > 0) {
               TransDataType(self, type, &dst);
             }
             return dst;
           })
      .def("_copy",
           [](const phi::DenseTensor &self, const platform::Place &place) {
             // follow fetch_op's inplementation
             phi::DenseTensor dst;
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
#ifdef PADDLE_WITH_CUDA
      .def("_share_buffer_with",
           [](phi::DenseTensor &self, const phi::DenseTensor src,
              py::tuple t) {
             auto *cuda_ipc_allocation =
                 dynamic_cast<memory::allocation::CudaIpcAllocation *>(
                     src.Holder().get());

             PADDLE_ENFORCE_NOT_NULL(
                 cuda_ipc_allocation,
                 platform::errors::PreconditionNotMet(
                     "Tensor is not Cuda IPC shared tensor. "
                     "Now only Tensor shared by cuda ipc could use this "
                     "api."));

             size_t size = t[0].cast<size_t>();
             auto dtype =
                 static_cast<paddle::experimental::DataType>(t[1].cast<int>());
             auto dims = phi::make_ddim(t[2].cast<std::vector<int>>());
             auto lod_info = t[3].cast<framework::LoD>();
             auto device_id = t[4].cast<int>();

             auto shared_reader_holder =
                 std::make_shared<memory::allocation::Allocation>(
                     cuda_ipc_allocation->ptr(),
                     cuda_ipc_allocation->base_ptr(), size,
                     platform::CUDAPlace(device_id));

             self.ResetHolderWithType(shared_reader_holder, dtype);
             self.Resize(dims);
             self.set_lod(lod_info);

             VLOG(6) << "Reconstructed tensor with buffer shared!";
           },
           R"DOC(
           Deserialize GPU Tensor for existed shared Cuda IPC tensor.

           Params:
               tensor: Shared Cuda IPC tensor.
               tuple: contrains data size, data type,
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
                 platform::is_gpu_place(holder->place()), true,
                 platform::errors::InvalidArgument(
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

             return py::make_tuple(_handle, (py::size_t)offset_bytes, data_size,
                                   type_idx, vectorize(self.dims()), self.lod(),
                                   device_id);
           },
           R"DOC(
           Serialize GPU Tensor by cudaIpcMemHandle.

           Returns:
               tuple: contrains handle, data size, data type,
                      tensor dims, lod information, device index.

           Examples:
               .. code-block:: python

                 import paddle
                 tensor = paddle.ones([3,3])
                 metainfo = tensor.value().get_tensor()._share_cuda()

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
                 static_cast<paddle::experimental::DataType>(t[3].cast<int>()));
             tensor.Resize(phi::make_ddim(t[4].cast<std::vector<int>>()));
             tensor.set_lod(t[5].cast<framework::LoD>());

             return tensor;
           },
           R"DOC(
           Deserialize GPU lod tensor from cudaIpcMemHandle.

           Params:
               tuple: contrains handle, data size, data type,
                      tensor dims, lod information, device index.

           Examples:
               .. code-block:: python

                 import paddle
                 tensor = paddle.ones([3,3])
                 metainfo = tensor.value().get_tensor()._share_cuda()
                 tensor_from_shared = paddle.to_tensor(paddle.fluid.core.LoDTensor._new_shared_cuda(metainfo))

        )DOC")
#endif
      .def("_share_filename",
           [](phi::DenseTensor &self) {
             if (!self.IsInitialized() || self.numel() == 0)
               throw std::runtime_error(
                   "Tensor not initialized or numel is 0. could not pass to "
                   "shared memory. ");

             auto holder = self.Holder();
             PADDLE_ENFORCE_EQ(
                 platform::is_cpu_place(holder->place()) ||
                     platform::is_cuda_pinned_place(holder->place()),
                 true, platform::errors::InvalidArgument(
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
               std::string handle = memory::allocation::GetIPCName();
               auto shared_holder =
                   memory::allocation::AllocateRefcountedMemoryMapAllocation(
                       handle, flags, data_size);

               // copy data & reset holder
               if (platform::is_cuda_pinned_place(holder->place())) {
#ifdef PADDLE_WITH_CUDA
                 memory::Copy(platform::CPUPlace(), shared_holder->ptr(),
                              platform::CUDAPinnedPlace(), data_ptr, data_size);
#endif
               } else {
                 memory::Copy(platform::CPUPlace(), shared_holder->ptr(),
                              platform::CPUPlace(), data_ptr, data_size);
               }
               self.ResetHolder(shared_holder);
               mmap_allocation = shared_holder.get();
             }
             int type_idx = static_cast<int>(self.type());

             return py::make_tuple(mmap_allocation->ipc_name(),
                                   mmap_allocation->size(), type_idx,
                                   vectorize(self.dims()), self.lod());
           },
           R"DOC(
           Serialize CPU lod tensor in shared memory to tuple.
           If the tensor is not in shared memory, we will copy it first.

           Returns:
               tuple: contrains ipc name, data size, data type,
                      tensor dims and lod imformation.

           Examples:
               .. code-block:: python

                 import paddle
                 tensor = paddle.ones([3,3])
                 metainfo = tensor.value().get_tensor()._share_filename()

       )DOC")
      .def("_new_shared_filename",
           [](py::tuple t) {  // __setstate__
             if (t.size() != 5)
               throw std::runtime_error("Invalid Tensor meta info state!");

             phi::DenseTensor tensor;

             // 2. Rebuild Allocation
             const std::string &ipc_name = t[0].cast<std::string>();
             size_t size = t[1].cast<size_t>();
             int flags = memory::allocation::MAPPED_SHAREDMEM |
                         memory::allocation::MAPPED_NOCREATE;

             auto shared_holder =
                 memory::allocation::AllocateRefcountedMemoryMapAllocation(
                     ipc_name, flags, size);

             // 3. Rebuild Tensor
             tensor.ResetHolderWithType(
                 shared_holder,
                 static_cast<paddle::experimental::DataType>(t[2].cast<int>()));
             tensor.Resize(phi::make_ddim(t[3].cast<std::vector<int>>()));
             tensor.set_lod(t[4].cast<framework::LoD>());

             return tensor;
           },
           R"DOC(
           Deserialize CPU lod tensor from shared memory.

           Params:
               tuple: contrains ipc file name, data size, data type,
                      tensor dims and lod information.

           Examples:
               .. code-block:: python

                 import paddle
                 tensor = paddle.ones([3,3])
                 metainfo = tensor.value().get_tensor()._share_filename()
                 tensor_from_shared = paddle.to_tensor(paddle.fluid.core.LoDTensor._new_shared_filename(metainfo))

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
           [](phi::SelectedRows &instance,
              const std::vector<int64_t> rows,
              const int64_t &height) {
             new (&instance) phi::SelectedRows(rows, height);
           })
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

  py::class_<phi::SparseCooTensor>(m, "SparseCooTensor")
      .def("__init__",
           [](phi::SparseCooTensor &instance) {
             new (&instance) phi::SparseCooTensor();
           })
      .def("numel",
           [](const phi::SparseCooTensor &self) -> int64_t {
             return self.numel();
           })
      .def("indices", [](const phi::SparseCooTensor &self) -> phi::DenseTensor {
        return self.indices();
      });
}

}  // namespace pybind
}  // namespace paddle
