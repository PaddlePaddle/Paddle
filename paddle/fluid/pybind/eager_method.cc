/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
// disable numpy compile error

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <Python.h>
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/string_array.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/slice_utils.h"
#include "paddle/fluid/pybind/uva_utils.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "pybind11/detail/internals.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "paddle/fluid/eager/amp_utils.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/eager_amp_auto_cast.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/utils/pybind.h"

PHI_DECLARE_bool(set_to_1d);
PHI_DECLARE_bool(use_stride_kernel);

namespace paddle {
namespace pybind {

extern void InitTensorWithNumpyValue(TensorObject* self,
                                     const pybind11::object& array,
                                     const paddle::platform::Place& place,
                                     bool zero_copy);

extern PyTypeObject* p_tensor_type;

Py_ssize_t GetSliceIndexFromPyObject(PyObject* obj) {
  if (PyObject_TypeCheck(obj, p_tensor_type)) {
    VLOG(6) << "Call GetSliceIndexFromTensor in Eager";
    paddle::Tensor tensor = CastPyArg2Tensor(obj, 0);
    PADDLE_ENFORCE_EQ(
        tensor.initialized(),
        true,
        paddle::platform::errors::InvalidArgument(
            "We can only support initialized tensor in slice, however we got "
            "uninitialized tensor %s, please check your code.",
            tensor.name()));
    return GetSliceIndexFromTensor((*static_cast<phi::DenseTensor*>(
        CastPyArg2Tensor(obj, 0).impl().get())));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "We should only get paddle::Tensor or VarBase in this "
        "method, when you reach this means we got another type index."));
  }
}

namespace {
#ifdef PADDLE_WITH_DISTRIBUTE
phi::DenseTensor ReshardXToReplicated(
    phi::distributed::DistTensor* dist_tensor) {
  if (!dist_tensor->dist_attr().is_replicated()) {
    phi::distributed::TensorDistAttr dist_attr(dist_tensor->dist_attr());
    std::vector<int64_t> dims_mapping(dist_tensor->dims().size(), -1);
    dist_attr.set_dims_mapping(dims_mapping);

    // reshard to replicate dist tensor
    auto* func =
        phi::distributed::ChooseProperReshardFunction(*dist_tensor, dist_attr);
    auto* dev_ctx =
        phi::DeviceContextPool::Instance().Get(dist_tensor->place());
    auto out_tensor = func->Eval(dev_ctx, *dist_tensor, dist_attr);
    return out_tensor->value();
  } else {
    return dist_tensor->value();
  }
}
#endif
}  // namespace

PyDoc_STRVAR(tensor_method_numpy__doc__,  // NOLINT
             R"DOC(numpy($self, /)
--

Returns a numpy array shows the value of current Tensor.

Returns:
    ndarray, The numpy value of current Tensor, dtype is
    same as current Tensor.

Examples:
    .. code-block:: python

        import paddle

        data = paddle.uniform([30, 10, 32], dtype="float32", min=-1, max=1)
        linear = paddle.nn.Linear(32, 64)
        data = paddle.to_tensor(data)
        x = linear(data)
        print(x.numpy())
)DOC");

static PyObject* tensor_method_numpy(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  auto& api = pybind11::detail::npy_api::get();
  if (!self->tensor.impl()) {
    Py_intptr_t py_dims[paddle::framework::DDim::kMaxRank];     // NOLINT
    Py_intptr_t py_strides[paddle::framework::DDim::kMaxRank];  // NOLINT
    py_dims[0] = 0;
    py_strides[0] = 0;

    PyObject* array = api.PyArray_NewFromDescr_(
        api.PyArray_Type_,
        api.PyArray_DescrFromType_(pybind11::detail::npy_api::NPY_FLOAT_),
        1,
        py_dims,
        py_strides,
        nullptr,
        pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
            pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
        nullptr);
    return array;
  }
  auto tensor_dims = self->tensor.shape();
  auto numpy_dtype = TensorDtype2NumpyDtype(self->tensor.type());
  auto sizeof_dtype = phi::SizeOf(self->tensor.type());
  Py_intptr_t py_dims[paddle::framework::DDim::kMaxRank];     // NOLINT
  Py_intptr_t py_strides[paddle::framework::DDim::kMaxRank];  // NOLINT
  size_t py_rank = tensor_dims.size();
  size_t numel = 1;
  if (py_rank == 0) {
    Py_ssize_t args_num = PyTuple_Size(args);
    // true by default
    bool set_to_1d = FLAGS_set_to_1d;
    if (args_num == (Py_ssize_t)1) {
      PyObject* obj = PyTuple_GET_ITEM(args, 0);
      if (obj == Py_False) {
        set_to_1d = false;
      }
    }
    if (set_to_1d) {
      // 0D Tensor hack process to 1D numpy, will remove in release 2.6
      VLOG(0)
          << "Warning:: 0D Tensor cannot be used as 'Tensor.numpy()[0]' . In "
             "order to avoid this problem, "
             "0D Tensor will be changed to 1D numpy currently, but it's not "
             "correct and will be "
             "removed in release 2.6. For Tensor contain only one element, "
             "Please "
             "modify "
             " 'Tensor.numpy()[0]' to 'float(Tensor)' as soon as "
             "possible, "
             "otherwise 'Tensor.numpy()[0]' will raise error in release 2.6.";
      py_rank = 1;
      py_dims[0] = 1;
      py_strides[0] = static_cast<Py_intptr_t>(sizeof_dtype * numel);
    }
  } else if (self->tensor.is_dense_tensor()) {
    auto tensor_stride = self->tensor.strides();

    for (int i = static_cast<int>(tensor_dims.size()) - 1; i >= 0; --i) {
      py_dims[i] = static_cast<Py_intptr_t>(tensor_dims[i]);
      py_strides[i] = static_cast<Py_intptr_t>(sizeof_dtype * tensor_stride[i]);
      numel *= py_dims[i];
    }
  } else {
    for (int i = static_cast<int>(tensor_dims.size()) - 1; i >= 0; --i) {
      py_dims[i] = static_cast<Py_intptr_t>(tensor_dims[i]);
      py_strides[i] = static_cast<Py_intptr_t>(sizeof_dtype * numel);
      numel *= py_dims[i];
    }
  }

  if (!self->tensor.impl()->initialized()) {
    PyObject* array = api.PyArray_NewFromDescr_(
        api.PyArray_Type_,
        api.PyArray_DescrFromType_(numpy_dtype),
        static_cast<int>(py_rank),
        py_dims,
        py_strides,
        nullptr,
        pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
            pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
        nullptr);

    if (tensor_dims.empty()) {
      py_dims[0] = 0;
      py_strides[0] = 0;
      PyObject* array = api.PyArray_NewFromDescr_(
          api.PyArray_Type_,
          api.PyArray_DescrFromType_(numpy_dtype),
          1,
          py_dims,
          py_strides,
          nullptr,
          pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
              pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
          nullptr);
      return array;
    }
    return array;
  }

  phi::DenseTensor cpu_tensor;
  platform::CPUPlace cpu_place;

  if (self->tensor.is_cpu() || self->tensor.is_gpu_pinned()) {
    eager_gil_scoped_release guard;
    platform::CPUPlace place;
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor =
          static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      // deep copy
      paddle::memory::Copy(place,
                           cpu_tensor.Holder()->ptr(),
                           place,
                           dense_tensor->Holder()->ptr(),
                           dense_tensor->Holder()->size());
    } else if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
      VLOG(6) << "Getting DistTensor's numpy value";
      auto* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
      auto dense_tensor = ReshardXToReplicated(dist_tensor);

      cpu_tensor.set_meta(dense_tensor.meta());
      // deep copy
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor.Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      // deep copy
      paddle::memory::Copy(place,
                           cpu_tensor.Holder()->ptr(),
                           place,
                           dense_tensor.Holder()->ptr(),
                           dense_tensor.Holder()->size());
#else
      PADDLE_THROW(
          platform::errors::Unavailable("The `numpy()` method of (Dist)Tensor "
                                        "is not supported in the current "
                                        "PaddlePaddle, please recompile and "
                                        "installPaddlePaddle with the option "
                                        "of `WITH_DISTRIBUTE=ON`."));
#endif
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      // deep copy
      paddle::memory::Copy(place,
                           cpu_tensor.Holder()->ptr(),
                           place,
                           dense_tensor->Holder()->ptr(),
                           dense_tensor->Holder()->size());
    }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (self->tensor.is_gpu()) {
    eager_gil_scoped_release guard;
#if defined(PADDLE_WITH_CUDA)
    gpuMemcpyKind kind = cudaMemcpyDeviceToHost;
#elif defined(PADDLE_WITH_HIP)
    gpuMemcpyKind kind = hipMemcpyDeviceToHost;
    phi::DeviceContextPool::Instance().Get(self->tensor.place())->Wait();
#endif
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor =
          static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      paddle::platform::GpuMemcpySync(cpu_tensor.Holder()->ptr(),
                                      dense_tensor->Holder()->ptr(),
                                      dense_tensor->Holder()->size(),
                                      kind);
    } else if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
      VLOG(6) << "Getting DistTensor's numpy value";
      auto* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
      auto dense_tensor = ReshardXToReplicated(dist_tensor);

      cpu_tensor.set_meta(dense_tensor.meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor.Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      paddle::platform::GpuMemcpySync(cpu_tensor.Holder()->ptr(),
                                      dense_tensor.Holder()->ptr(),
                                      dense_tensor.Holder()->size(),
                                      kind);
#else
      PADDLE_THROW(
          platform::errors::Unavailable("The `numpy()` method of (Dist)Tensor "
                                        "is not supported in the current "
                                        "PaddlePaddle, please recompile and "
                                        "installPaddlePaddle with the option "
                                        "of `WITH_DISTRIBUTE=ON`."));
#endif
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      paddle::platform::GpuMemcpySync(cpu_tensor.Holder()->ptr(),
                                      dense_tensor->Holder()->ptr(),
                                      dense_tensor->Holder()->size(),
                                      kind);
    }
#endif
#if defined(PADDLE_WITH_XPU)
  } else if (self->tensor.is_xpu()) {
    platform::CPUPlace place;
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor =
          static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      paddle::memory::Copy(place,
                           cpu_tensor.Holder()->ptr(),
                           dense_tensor->place(),
                           dense_tensor->Holder()->ptr(),
                           dense_tensor->Holder()->size());
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      paddle::memory::Copy(place,
                           cpu_tensor.Holder()->ptr(),
                           dense_tensor->place(),
                           dense_tensor->Holder()->ptr(),
                           dense_tensor->Holder()->size());
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (self->tensor.is_custom_device()) {
    eager_gil_scoped_release guard;
    phi::DeviceContextPool::Instance().Get(self->tensor.place())->Wait();
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor =
          static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      phi::DeviceManager::GetDeviceWithPlace(self->tensor.place())
          ->MemoryCopyD2H(cpu_tensor.Holder()->ptr(),
                          dense_tensor->Holder()->ptr(),
                          dense_tensor->Holder()->size());
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      // TODO(qili93): temporary for ascend npu performance to be removed along
      // with npu_identity op
      paddle::Tensor temp_tensor(std::make_shared<phi::DenseTensor>());
      if (dense_tensor->storage_properties_initialized()) {
        temp_tensor = npu_identity_ad_func(self->tensor, -1);
        dense_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(temp_tensor.impl());
      }
      cpu_tensor.set_meta(dense_tensor->meta());
      auto tmp_allocation_ptr =
          memory::Alloc(cpu_place, dense_tensor->Holder()->size());
      cpu_tensor.ResetHolder(std::shared_ptr<phi::Allocation>(
          tmp_allocation_ptr.release(), tmp_allocation_ptr.get_deleter()));
      phi::DeviceManager::GetDeviceWithPlace(self->tensor.place())
          ->MemoryCopyD2H(cpu_tensor.Holder()->ptr(),
                          dense_tensor->Holder()->ptr(),
                          dense_tensor->Holder()->size());
    }
#endif
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Tensor.numpy() only support cpu tensor."));
    RETURN_PY_NONE
  }

  void* array_buffer = cpu_tensor.Holder()->ptr();
  size_t array_offset = cpu_tensor.offset();

  PyObject* base = ToPyObject(paddle::Tensor(
      std::make_shared<phi::DenseTensor>(std::move(cpu_tensor))));

  PyObject* array = api.PyArray_NewFromDescr_(
      api.PyArray_Type_,
      api.PyArray_DescrFromType_(numpy_dtype),
      static_cast<int>(py_rank),
      py_dims,
      py_strides,
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(array_buffer) +
                              array_offset),
      pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
          pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
      nullptr);

  api.PyArray_SetBaseObject_(array, base);

  return array;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_numpy_for_string_tensor(TensorObject* self,
                                                       PyObject* args,
                                                       PyObject* kwargs) {
  EAGER_TRY
  auto& api = pybind11::detail::npy_api::get();
  if (!self->tensor.impl() || !self->tensor.impl()->initialized()) {
    VLOG(6) << "The StringTensor is uninitialized. Return the empty string "
               "numpy array.";
    Py_intptr_t py_dims[paddle::framework::DDim::kMaxRank];     // NOLINT
    Py_intptr_t py_strides[paddle::framework::DDim::kMaxRank];  // NOLINT
    py_dims[0] = 0;
    py_strides[0] = 0;

    PyObject* array = api.PyArray_NewFromDescr_(
        api.PyArray_Type_,
        api.PyArray_DescrFromType_(pybind11::detail::npy_api::NPY_UNICODE_),
        1,
        py_dims,
        py_strides,
        nullptr,
        pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
            pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
        nullptr);
    return array;
  }

  if (self->tensor.is_cpu()) {
    VLOG(6) << "Getting StringTensor's numpy value";
    auto string_tensor =
        std::dynamic_pointer_cast<phi::StringTensor>(self->tensor.impl());
    const auto* st_ptr = string_tensor->data();
    auto numel = self->tensor.numel();
    auto tensor_dims = self->tensor.shape();
    // Get the max unicode length of StringTensor to create numpy unicode
    // string array.
    auto* longest_pstring = std::max_element(
        st_ptr, st_ptr + numel, [](const auto& a, const auto& b) {
          auto a_unicode_len =
              phi::strings::GetUnicodeStrLen(a.data(), a.size());
          auto b_unicode_len =
              phi::strings::GetUnicodeStrLen(b.data(), b.size());
          return a_unicode_len < b_unicode_len;
        });
    size_t max_unicode_length = phi::strings::GetUnicodeStrLen(
        longest_pstring->data(), longest_pstring->size());
    max_unicode_length = (max_unicode_length == 0) ? 1 : max_unicode_length;
    VLOG(6) << "The max unicode length is " << max_unicode_length;
    auto sp =
        std::make_unique<uint32_t[]>(max_unicode_length * numel);  // NOLINT
    auto py_array_data = sp.get();
    memset(py_array_data, 0, max_unicode_length * numel * sizeof(uint32_t));
    for (int64_t i = 0; i < numel; ++i) {
      auto curr_unicode_len =
          phi::strings::GetUnicodeStrLen(st_ptr[i].data(), st_ptr[i].size());
      phi::strings::GetUnicodeStr(st_ptr[i].data(),
                                  py_array_data + i * max_unicode_length,
                                  curr_unicode_len);
    }
    py::array array(py::dtype("U" + std::to_string(max_unicode_length)),
                    tensor_dims,
                    {},
                    py_array_data);
    return array.release().ptr();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "StringTensor.numpy() only support cpu tensor."));
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__is_initialized(TensorObject* self,
                                               PyObject* args,
                                               PyObject* kwargs) {
  EAGER_TRY
  return ToPyObject(self->tensor.initialized());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__is_dense_tensor_hold_allocation(
    TensorObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  auto dense_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
  if (dense_tensor) {
    return ToPyObject(dense_tensor->IsInitialized());
  } else {
    return ToPyObject(false);
  }

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static void IncreaseTensorReferenceCountUntilCopyComplete(
    const paddle::Tensor& tensor, const platform::Place& place) {
  auto place_ = platform::is_gpu_place(place) ? place : tensor.place();

  auto tracer = egr::Controller::Instance().GetCurrentTracer();
  auto gc = tracer->MutableGarbageCollectorIfNotExists(place_);

  // Note(dev): This is an empty callback, the only way is to "reference"
  // inner memory Holder, so it will not be destructed until the kernels
  // launched at current stream of given place is finished, such as
  // CUDAPinned Mem -> CUDA by cudaMemcpyAsync.
  auto callback = [tensor, place_]() {
    VLOG(3) << "Run callback of Tensor:" << tensor.name() << " at place "
            << place_;
  };
  gc->DirectClearCallback(callback);
}

static PyObject* tensor_method__copy_to(TensorObject* self,
                                        PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 0), 0);
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 1), 1);
  paddle::Tensor cp_tensor;
  {
    eager_gil_scoped_release guard;
    cp_tensor = self->tensor.copy_to(place, blocking);
    if (!blocking) {
      IncreaseTensorReferenceCountUntilCopyComplete(self->tensor, place);
    }
    egr::EagerUtils::autograd_meta(&cp_tensor)->SetStopGradient(true);
    egr::EagerUtils::autograd_meta(&cp_tensor)
        ->SetPersistable(
            egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  }
  return ToPyObject(cp_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_reconstruct_from___doc__,
             R"DOC(reconstruct_from_($self, other/)
--

Reconstruct the self with other Tensor. It is a deep copy of 'self = other'.

Returns:
    None.

Examples:
    .. code-block:: python

      import paddle

      t1 = paddle.to_tensor([1.0], stop_gradient=False)
      t2 = paddle.to_tensor([2.0], stop_gradient=True)

      t1.reconstruct_from_(t2)

      print(t1)
)DOC");

static PyObject* tensor_method_reconstruct_from_(TensorObject* self,
                                                 PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor src_tensor = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  std::string orig_name = self->tensor.name();
  VLOG(6) << "Start Reconstructing Tensor from" << src_tensor.name() << " to "
          << orig_name;
  self->tensor = src_tensor;

  // Recover source name
  self->tensor.set_name(orig_name);

  VLOG(6) << "Finished Reconstructing Tensor from" << src_tensor.name()
          << " to " << self->tensor.name();
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_copy_(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor src_tensor = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 1), 1);
  VLOG(6) << "Start Copy Tensor " << src_tensor.name() << " to "
          << self->tensor.name();
  if (!self->tensor.initialized()) {
    eager_gil_scoped_release guard;
    egr::EagerUtils::autograd_meta(&(self->tensor))
        ->SetStopGradient(
            egr::EagerUtils::autograd_meta(&(src_tensor))->StopGradient());
    egr::EagerUtils::autograd_meta(&(self->tensor))
        ->SetPersistable(
            egr::EagerUtils::autograd_meta(&(src_tensor))->Persistable());
    if (src_tensor.initialized()) {
      self->tensor.copy_(src_tensor, src_tensor.place(), blocking);
    }
  } else {
    if (src_tensor.initialized()) {
      eager_gil_scoped_release guard;
      self->tensor.copy_(src_tensor, self->tensor.place(), blocking);
    }
  }

  VLOG(6) << "Finish Copy Tensor " << src_tensor.name() << " to "
          << self->tensor.name();
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_clone__doc__,  // NOLINT
             R"DOC(clone($self, /)
--

Returns a new Tensor, which is clone of origin Tensor, and it remains in the current graph.
It will always have a Tensor copy.
Tn addition, the cloned Tensor provides gradient propagation.

Returns:
    Tensor, The cloned Tensor.

Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor(1.0, stop_gradient=False)
        clone_x = x.clone()
        y = clone_x**2
        y.backward()
        print(clone_x.stop_gradient) # False
        print(clone_x.grad)          # [2.0], support gradient propagation
        print(x.stop_gradient)       # False
        print(x.grad)                # [2.0], clone_x support gradient propagation for x

        x = paddle.to_tensor(1.0)
        clone_x = x.clone()
        clone_x.stop_gradient = False
        z = clone_x**3
        z.backward()
        print(clone_x.stop_gradient) # False
        print(clone_x.grad)          # [3.0], support gradient propagation
        print(x.stop_gradient) # True
        print(x.grad)          # None
)DOC");

static PyObject* tensor_method_clone(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor out;
  {
    eager_gil_scoped_release guard;
    PADDLE_ENFORCE_EQ(
        self->tensor.initialized(),
        true,
        paddle::platform::errors::InvalidArgument(
            "We can only support initialized tensor in clone, however we got "
            "uninitialized tensor %s, please check your code.",
            self->tensor.name()));

    out = assign_ad_func(self->tensor);
  }
  return ToPyObject(out);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_retain_grads__doc__, R"DOC(retain_grads($self, /)
--

Enables this Tensor to have their grad populated during backward(). It is a no-op for leaf tensors.

Returns:
    None.

Examples:
    .. code-block:: python

      import paddle

      x = paddle.to_tensor([1.0, 2.0, 3.0])
      x.stop_gradient = False
      y = x + x
      y.retain_grads()
      loss = y.sum()
      loss.backward()

      print(y.grad) # [1., 1., 1.]

      x = paddle.to_tensor([1.0, 2.0, 3.0])
      x.stop_gradient = False
      y = x + x
      # y.retain_grads()
      loss = y.sum()
      loss.backward()

      print(y.grad) # None
)DOC");

static PyObject* tensor_retain_grads(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  if (egr::Controller::Instance().HasGrad()) {
    eager_gil_scoped_release guard;
    auto meta = egr::EagerUtils::autograd_meta(&(self->tensor));
    if (!meta->GetMutableGradNode()) {
      VLOG(6) << "Make grad node of tensor: " << self->tensor.name()
              << "become accumulation node";
      meta->SetGradNode(std::make_shared<egr::GradNodeAccumulation>(meta));
    }
    egr::egr_utils_api::RetainGradForTensor(self->tensor);
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_clear_gradient__doc__,  // NOLINT
             R"DOC(clear_gradient($self, set_to_zero=True, /)
--

Only for Tensor that has gradient, normally we use this for Parameters since
other temporary Tensor doesen't has gradient.

The Gradient of current Tensor will be set to ``0`` elementwise or ``None``.

Args:
    set_to_zero (bool, optional): If set to ``True``, the gradient will be set
        to ``0`` elementwise, otherwise the gradient will be set to ``None``.
        Default: ``True``.

Returns:
    None.

Examples:
    .. code-block:: python

        import paddle
        input = paddle.uniform([10, 2])
        linear = paddle.nn.Linear(2, 3)
        out = linear(input)
        out.backward()
        print("Before clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))
        linear.weight.clear_gradient()
        print("After clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))
)DOC");

static PyObject* tensor_clear_gradient(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "ClearGradient " << self->tensor.name();

  Py_ssize_t args_num = PyTuple_Size(args);
  bool set_to_zero = true;
  if (args_num == (Py_ssize_t)1) {
    set_to_zero = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 0), 0);
  }

  paddle::Tensor* grad;
  bool is_leaf = egr::EagerUtils::IsLeafTensor(self->tensor);
  if (is_leaf) {
    grad = egr::EagerUtils::mutable_grad(self->tensor);
    PADDLE_ENFORCE(grad != nullptr,
                   paddle::platform::errors::Fatal(
                       "Detected nullptr grad"
                       "Please check if you have manually cleared"
                       "the grad inside autograd_meta"));
  } else {
    auto meta = egr::EagerUtils::unsafe_autograd_meta(self->tensor);
    grad = meta->MutableGrad();
  }

  if (grad->impl()) {
    eager_gil_scoped_release guard;
    if (grad->is_selected_rows()) {
      auto selected_rows =
          std::dynamic_pointer_cast<phi::SelectedRows>(grad->impl());
      if (selected_rows->mutable_value()->IsInitialized()) {
        selected_rows->mutable_rows()->clear();
        selected_rows->mutable_value()->clear();
      }
    } else if (grad->is_dense_tensor()) {
      if (grad->initialized()) {
        if (set_to_zero) {
          auto* grad_t = static_cast<phi::DenseTensor*>(grad->impl().get());
          auto* dev_ctx =
              platform::DeviceContextPool::Instance().Get(grad_t->place());
          phi::funcs::set_constant(*dev_ctx, grad_t, 0.0);
          if (is_leaf) {
            std::static_pointer_cast<egr::GradNodeAccumulation>(
                egr::EagerUtils::grad_node(self->tensor))
                ->SetFakeEmpty(true);
          }
        } else {
          VLOG(4) << "Gradient of " << self->tensor.name()
                  << " is initialized, will be released.";
          auto dense_tensor =
              std::dynamic_pointer_cast<phi::DenseTensor>(grad->impl());
          dense_tensor->MoveMemoryHolder();
        }
      }
    }
  }

  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__zero_grads(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "ZeroGrads " << self->tensor.name();

  if (egr::EagerUtils::IsLeafTensor(self->tensor)) {
    eager_gil_scoped_release guard;
    // Add RetainGrad as PostHook to AccumulationNode
    paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
    PADDLE_ENFORCE(grad != nullptr,
                   paddle::platform::errors::Fatal(
                       "Detected nullptr grad"
                       "Please check if you have manually cleared"
                       "the grad inside autograd_meta"));
    if (grad->initialized()) {
      if (grad->is_dense_tensor()) {
        auto* t = static_cast<phi::DenseTensor*>(grad->impl().get());
        auto* dev_ctx = platform::DeviceContextPool::Instance().Get(t->place());
        phi::funcs::set_constant(*dev_ctx, t, 0.0);
      } else {
        grad->set_impl(paddle::experimental::zeros_like(*(grad)).impl());
      }
    }
  } else {
    eager_gil_scoped_release guard;
    auto meta = egr::EagerUtils::unsafe_autograd_meta(self->tensor);
    if (meta->MutableGrad()->initialized()) {
      if (meta->MutableGrad()->is_dense_tensor()) {
        auto* t =
            static_cast<phi::DenseTensor*>(meta->MutableGrad()->impl().get());
        auto* dev_ctx = platform::DeviceContextPool::Instance().Get(t->place());
        phi::funcs::set_constant(*dev_ctx, t, 0.0);
      } else {
        meta->MutableGrad()->set_impl(
            paddle::experimental::zeros_like(*(meta->MutableGrad())).impl());
      }
    }
  }

  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__share_buffer_to(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* dst_ptr =
      &(reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor);
  PADDLE_ENFORCE_EQ(self->tensor.initialized(),
                    true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        self->tensor.name()));
  auto* src_tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  if (!dst_ptr->defined()) {
    dst_ptr->set_impl(std::make_shared<phi::DenseTensor>());
  }
  auto dst_tensor = static_cast<phi::DenseTensor*>(dst_ptr->impl().get());
  dst_tensor->ShareBufferWith(*src_tensor);
  dst_tensor->ShareDataTypeWith(*src_tensor);
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__is_shared_buffer_with(TensorObject* self,
                                               PyObject* args,
                                               PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* dst_ptr =
      &(reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor);
  PADDLE_ENFORCE_EQ(self->tensor.initialized(),
                    true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        self->tensor.name()));
  bool res = false;
  if (!self->tensor.defined() || !dst_ptr->defined()) {
    return ToPyObject(res);
  }
  auto* self_ptr = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  auto dst_tensor = static_cast<phi::DenseTensor*>(dst_ptr->impl().get());
  res = dst_tensor->IsSharedBufferWith(*self_ptr);
  return ToPyObject(res);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__share_underline_tensor_to(TensorObject* self,
                                                   PyObject* args,
                                                   PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* src_ptr =
      &(reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor);
  PADDLE_ENFORCE_EQ(self->tensor.initialized(),
                    true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        self->tensor.name()));
  src_ptr->set_impl(self->tensor.impl());
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__is_shared_underline_tensor_with(TensorObject* self,
                                                         PyObject* args,
                                                         PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor src_tensor = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  PADDLE_ENFORCE_EQ(src_tensor.initialized(),
                    true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        src_tensor.name()));
  bool res = false;
  if (!self->tensor.defined() || !src_tensor.defined()) {
    return ToPyObject(res);
  }
  res = (self->tensor.impl().get() == src_tensor.impl().get());
  return ToPyObject(res);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_detach__doc__,  // NOLINT
             R"DOC(detach($self, /)
--

Returns a new Tensor, detached from the current graph.
It will share data with origin Tensor and always doesn't have a Tensor copy.
In addition, the detached Tensor doesn't provide gradient propagation.

Returns:
    Tensor, The detached Tensor.

Examples:
    .. code-block:: python

      import paddle

      x = paddle.to_tensor([1.0], stop_gradient=False)
      detach_x = x.detach()
      detach_x[0] = 10.0
      print(x)  # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=False,
                  #        [10.])
      y = x**2
      y.backward()
      print(x.grad)         # [20.0]
      print(detach_x.grad)  # None, 'stop_gradient=True' by default

      detach_x.stop_gradient = False # Set stop_gradient to be False, supported auto-grad
      z = detach_x**3
      z.backward()

      print(x.grad)         # [20.0], detach_x is detached from x's graph, not affect each other
      print(detach_x.grad)  # [300.0], detach_x has its own graph

      # Due to sharing of data with origin Tensor, There are some unsafe operations:
      # y = 2 * x
      # detach_x[:] = 5.0
      # y.backward()
      # It will raise Error:
      #   one of the variables needed for gradient computation has been modified by an inplace operation.
)DOC");

static PyObject* tensor_method_detach(TensorObject* self,
                                      PyObject* args,
                                      PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE_EQ(
      self->tensor.defined(),
      true,
      platform::errors::InvalidArgument("Tensor %s has not been initialized!",
                                        self->tensor.name()));

  PyObject* obj = p_tensor_type->tp_alloc(p_tensor_type, 0);
  if (obj) {
    auto v = reinterpret_cast<TensorObject*>(obj);
    new (&(v->tensor)) paddle::Tensor();
    v->tensor.set_impl(self->tensor.impl());
    v->tensor.set_name(egr::Controller::Instance().GenerateUniqueName());
    auto autograd_meta_src = egr::EagerUtils::autograd_meta(&(self->tensor));
    auto autograd_meta = egr::EagerUtils::autograd_meta(&(v->tensor));
    autograd_meta->SetPersistable(autograd_meta_src->Persistable());
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "tp_alloc return null, can not new a PyObject."));
  }

  return obj;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_detach___doc__, R"DOC(detach_($self, /)
--

Detach self from the current graph, and returns self Tensor.
In addition, the detached Tensor doesn't provide gradient propagation.

Returns:
    Tensor, The detached Tensor.
)DOC");

static PyObject* tensor_method_detach_(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE_EQ(
      self->tensor.defined(),
      true,
      platform::errors::InvalidArgument("Tensor %s has not been initialized!",
                                        self->tensor.name()));

  auto autograd_meta = std::make_shared<egr::AutogradMeta>();
  autograd_meta->SetPersistable(
      egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  self->tensor.set_autograd_meta(autograd_meta);
  Py_INCREF(reinterpret_cast<PyObject*>(self));
  return reinterpret_cast<PyObject*>(self);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_get_tensor__doc__, R"DOC(get_tensor($self, /)
--

Returns the underline tensor in the origin Tensor.

Returns:
    Underline tensor.

Examples:
    .. code-block:: python

      import paddle

      x = paddle.to_tensor([1.0], stop_gradient=False)
      underline_x = x.get_tensor()
      print(underline_x) # a Dense Tensor info
)DOC");

static PyObject* tensor_method_get_underline_tensor(TensorObject* self,
                                                    PyObject* args,
                                                    PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    // The original `get_tensor` method of Variable will create a empty tensor
    phi::DenseTensor empty_tensor;
    return ToPyObject(&empty_tensor);
  }
  if (self->tensor.is_dense_tensor()) {
    auto* tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
    VLOG(6) << "tensor: " << tensor->IsInitialized();
    return ToPyObject(tensor);
  } else if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
    auto* tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
    VLOG(6) << "dist tensor: " << tensor->defined();
    return ToPyObject(tensor);
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "The `get_tensor()` method of (Dist)Tensor is not supported in the "
        "current PaddlePaddle, please recompile and installPaddlePaddle "
        "with the option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_get_underline_selected_rows(TensorObject* self,
                                                           PyObject* args,
                                                           PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    RETURN_PY_NONE
  }
  if (self->tensor.is_selected_rows()) {
    auto* selected_rows =
        static_cast<phi::SelectedRows*>(self->tensor.impl().get());
    return ToPyObject(selected_rows);
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__get_tensor_from_selected_rows(
    TensorObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_selected_rows(),
                 paddle::platform::errors::Fatal(
                     "this method is only effective for SelectedRows."));

  auto* selected_rows =
      static_cast<phi::SelectedRows*>(self->tensor.impl().get());

  PADDLE_ENFORCE(
      selected_rows->initialized(),
      paddle::platform::errors::Fatal("SelectedRows must be initialized."));

  auto* dense_tensor =
      static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
  VLOG(4) << "dense_tensor: " << dense_tensor->IsInitialized();

  auto t = paddle::Tensor(egr::Controller::Instance().GenerateUniqueName());
  t.set_impl(std::make_shared<phi::DenseTensor>(*dense_tensor));

  return ToPyObject(t);

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__getitem_index_not_tensor(TensorObject* self,
                                                  PyObject* args,
                                                  PyObject* kwargs) {
  EAGER_TRY
  PyObject* _index = PyTuple_GET_ITEM(args, 0);
  VLOG(4) << "Call _getitem_index_not_tensor";
  std::vector<int> slice_axes, slice_starts, slice_ends, slice_strides,
      decrease_axis, none_axes, infer_flags;
  std::vector<int64_t> list_select_idxs;
  // if index is a list, list_select_flag will be true
  bool list_select_flag = false;
  // Note(0x45f): Using defined() instead of initialized()
  // to support slice tensor which shape like [0, 0, 0].
  PADDLE_ENFORCE_EQ(
      self->tensor.defined(),
      true,
      platform::errors::InvalidArgument(
          "tensor %s has not been initialized, we can only slice initialized "
          "tensor please init it first with numpy or other tensor.",
          self->tensor.name()));
  auto tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  ParseIndexingSlice(tensor,
                     _index,
                     &slice_axes,
                     &slice_starts,
                     &slice_ends,
                     &slice_strides,
                     &decrease_axis,
                     &none_axes,
                     &infer_flags,
                     &list_select_idxs,
                     &list_select_flag);

  auto out =
      slice_axes.empty() && !list_select_flag
          ? self->tensor
          : paddle::Tensor(egr::Controller::Instance().GenerateUniqueName());

  if (!slice_axes.empty()) {
    framework::AttributeMap attrs = {{"axes", slice_axes},
                                     {"starts", slice_starts},
                                     {"ends", slice_ends},
                                     {"infer_flags", infer_flags},
                                     {"decrease_axis", decrease_axis}};
    std::string op_type = "slice";
    for (auto stride : slice_strides) {
      if (stride != 1) {
        op_type = "strided_slice";
        attrs.insert({"strides", slice_strides});
        attrs.erase("decrease_axis");
        break;
      }
    }
    std::vector<int64_t> slice_axes_tmp(slice_axes.begin(), slice_axes.end());
    std::vector<int64_t> infer_flags_tmp(infer_flags.begin(),
                                         infer_flags.end());
    std::vector<int64_t> decrease_axis_tmp(decrease_axis.begin(),
                                           decrease_axis.end());

    if (op_type == "slice") {
      eager_gil_scoped_release guard;
      out = slice_ad_func(self->tensor,
                          slice_axes_tmp,
                          slice_starts,
                          slice_ends,
                          infer_flags_tmp,
                          decrease_axis_tmp);
    } else if (op_type == "strided_slice") {
      eager_gil_scoped_release guard;
      out = strided_slice_ad_func(
          self->tensor, slice_axes, slice_starts, slice_ends, slice_strides);
      if (!decrease_axis_tmp.empty()) {
        out = squeeze_ad_func(out, decrease_axis_tmp);
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Slice is only support slice and strided_slice, but we got %s which "
          "is impossible, please check your code first or contact us by "
          "issue. ",
          op_type));
    }
  }

  bool set_to_1d = FLAGS_set_to_1d;

  if (set_to_1d) {
    // NOTE(zoooo0820): When all axes are decreased, the output will be 1-D
    // with FLAGS_set_to_1d=True. In this case, one `None` should be pop out,
    // otherwise the output shape will be not correct.
    if (static_cast<int>(decrease_axis.size()) == tensor->dims().size()) {
      VLOG(1)
          << "Warning: In Tensor '__getitem__', if the number of scalar "
             "elements "
             "in the index is equal to the rank of the Tensor, the output "
             "should "
             "be 0-D. In order to be consistent with the behavior of previous "
             "versions, it will be processed to 1-D. But it is not correct and "
             "will be "
             "removed in release 2.6. "
             "If 1-D is still wanted, please modify the index element from "
             "scalar to slice "
             "(e.g. 'x[i]' => 'x[i:i+1]'). ";
      if (!none_axes.empty()) {
        none_axes.pop_back();
      }
    }
  }
  if (!none_axes.empty()) {
    paddle::Tensor new_out;
    {
      eager_gil_scoped_release guard;
      // Deal with cases that decrease_axes is not empty
      // For example:
      // # x.shape: (2,3,4)
      // out = x[0, 0:2, None] # out.shape : (2, 1, 4)
      for (auto& axis : none_axes) {
        int len = 0;
        for (int da : decrease_axis) {
          if (da < axis) {
            len++;
          }
        }
        axis -= len;
      }
      new_out = unsqueeze_ad_func(out, none_axes);
    }
    return ToPyObject(new_out);
  }

  // the index is a list
  if (list_select_flag) {
    eager_gil_scoped_release guard;
    if (FLAGS_use_stride_kernel && list_select_idxs.size() == 1) {
      out = index_select_strided_ad_func(self->tensor, list_select_idxs[0], 0);
    } else {
      auto select_index =
          paddle::Tensor(egr::Controller::Instance().GenerateUniqueName());
      auto idx_tensor = std::make_shared<phi::DenseTensor>();
      select_index.set_impl(idx_tensor);
      auto* dev_ctx = platform::DeviceContextPool::Instance().Get(
          egr::Controller::Instance().GetExpectedPlace());
      paddle::framework::TensorFromVector(
          list_select_idxs, *dev_ctx, idx_tensor.get());
      out = index_select_ad_func(self->tensor, select_index, 0);
    }
  }

  return ToPyObject(out);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__getitem_from_offset(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  phi::DenseTensor* ptr = nullptr;
  if (self->tensor.is_selected_rows()) {
    auto* selected_rows =
        static_cast<phi::SelectedRows*>(self->tensor.impl().get());
    ptr = static_cast<phi::DenseTensor*>(selected_rows->mutable_value());
  } else {
    ptr = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  }
  PADDLE_ENFORCE_NOT_NULL(ptr,
                          platform::errors::InvalidArgument(
                              "%s is not a DenseTensor.", self->tensor.name()));
  const auto& tensor = *ptr;
  PADDLE_ENFORCE_EQ(
      tensor.IsInitialized(),
      true,
      platform::errors::InvalidArgument(
          "Tensor of %s is Empty, please check if it has no data.",
          self->tensor.name()));

  const auto& tensor_dims = tensor.dims();

  std::vector<size_t> dims(tensor_dims.size());
  std::vector<size_t> stride = phi::vectorize<size_t>(tensor.strides());

  size_t numel = 1;
  for (int i = tensor_dims.size() - 1; i >= 0; --i) {
    dims[i] = static_cast<size_t>(tensor_dims[i]);
    numel *= dims[i];
  }
  size_t offset = 0;
  if (PyTuple_Size(args) == 0) {
    PADDLE_ENFORCE_EQ(numel,
                      1,
                      platform::errors::InvalidArgument(
                          "only one element tensors can be converted to Python "
                          "scalars when no input coordinates"));
  } else if (PyTuple_Size(args) == 1) {
    offset = CastPyArg2AttrLong(PyTuple_GET_ITEM(args, 0), 0);
    PADDLE_ENFORCE_LT(
        offset,
        numel,
        platform::errors::InvalidArgument(
            "index %d is out of bounds for size %d", offset, numel));
  } else {
    PADDLE_ENFORCE_EQ(PyTuple_Size(args),
                      dims.size(),
                      platform::errors::InvalidArgument(
                          "incorrect number of indices for Tensor"));

    for (Py_ssize_t i = 0; i < PyTuple_Size(args); ++i) {
      size_t index = CastPyArg2AttrLong(PyTuple_GET_ITEM(args, i), i);
      PADDLE_ENFORCE_LT(
          index,
          dims[i],
          platform::errors::InvalidArgument(
              "index %d is out fo bounds for axis %d with size %d",
              index,
              i,
              dims[i]));
      offset += index * stride[i];
    }
  }
#define PD_FOR_EACH_DENSE_TENSOR_DATA_TYPE(_) \
  _(bool, DataType::BOOL)                     \
  _(int8_t, DataType::INT8)                   \
  _(uint8_t, DataType::UINT8)                 \
  _(int16_t, DataType::INT16)                 \
  _(uint16_t, DataType::UINT16)               \
  _(int32_t, DataType::INT32)                 \
  _(uint32_t, DataType::UINT32)               \
  _(int64_t, DataType::INT64)                 \
  _(uint64_t, DataType::UINT64)               \
  _(bfloat16, DataType::BFLOAT16)             \
  _(float16, DataType::FLOAT16)               \
  _(float, DataType::FLOAT32)                 \
  _(double, DataType::FLOAT64)                \
  _(complex64, DataType::COMPLEX64)           \
  _(complex128, DataType::COMPLEX128)

#define TENSOR_TO_PY_SCALAR(T, proto_type)                                   \
  if (tensor.dtype() == proto_type) {                                        \
    auto numpy_dtype = TensorDtype2NumpyDtype(proto_type);                   \
    T b = paddle::pybind::TensorGetElement<T>(tensor, offset);               \
    Py_intptr_t py_dims[paddle::framework::DDim::kMaxRank];    /* NOLINT */  \
    Py_intptr_t py_strides[paddle::framework::DDim::kMaxRank]; /* NOLINT */  \
    auto& api = pybind11::detail::npy_api::get();                            \
    PyObject* array = api.PyArray_NewFromDescr_(                             \
        api.PyArray_Type_,                                                   \
        api.PyArray_DescrFromType_(numpy_dtype),                             \
        0,                                                                   \
        py_dims,                                                             \
        py_strides,                                                          \
        nullptr,                                                             \
        pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |                      \
            pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,                 \
        nullptr);                                                            \
    std::memcpy(                                                             \
        reinterpret_cast<void*>(pybind11::detail::array_proxy(array)->data), \
        static_cast<void*>(&b),                                              \
        sizeof(b));                                                          \
    return array;                                                            \
  }

  PD_FOR_EACH_DENSE_TENSOR_DATA_TYPE(TENSOR_TO_PY_SCALAR);
#undef TENSOR_TO_PY_SCALAR
  PADDLE_THROW(platform::errors::Unimplemented(
      "Unsupported tensor data type: %s", tensor.dtype()));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__setitem_eager_tensor(TensorObject* self,
                                                     PyObject* args,
                                                     PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Call __setitem_eager_tensor";

  auto self_tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());

  PyObject* _index = PyTuple_GET_ITEM(args, 0);
  PyObject* value_obj = PyTuple_GET_ITEM(args, 1);
  // NOTE(zhiqiu): PyTuple_Pack increases refcount while PyTuple_New
  // https://github.com/python/cpython/blob/24b63c695ae0a95b06379eaadace66735abac1e2/Objects/tupleobject.c#L251
  PyObject* index_ptr =
      !PyTuple_Check(_index) ? PyTuple_Pack(1, _index) : _index;
  DEFINE_PADDLE_SCOPE_GUARD([index_ptr, &_index]() {
    if (!PyTuple_Check(_index)) {
      Py_DECREF(index_ptr);
      VLOG(4) << "Call Py_DECREF";
    }
  });

  // 1. Check argumnets
  bool parse_index = true;

  // Check whether _index can be parsed.
  const int size = PyTuple_GET_SIZE(index_ptr);
  for (int dim = 0; dim < size; ++dim) {
    PyObject* slice_item = PyTuple_GetItem(index_ptr, dim);
    if (!(PyCheckInteger(slice_item) || PySlice_Check(slice_item) ||
          slice_item == Py_Ellipsis || slice_item == Py_None)) {
      parse_index = false;
      break;
    }
  }

  // 2. Call op set_value to speed up if the condition is met,
  // otherwise call TensorToPyArray.
  // TODO(liym27): Try not to call TensorToPyArray because it always
  // copys data to cpu place, which reduces performance.
  if (parse_index) {
    std::vector<int> axes, starts, ends, steps, decrease_axes, none_axes,
        infer_flags;
    std::vector<int64_t> list_select_idxs;
    // if index is a list, list_select_flag will be true
    bool list_select_flag = false;
    ParseIndexingSlice(self_tensor,
                       index_ptr,
                       &axes,
                       &starts,
                       &ends,
                       &steps,
                       &decrease_axes,
                       &none_axes,
                       &infer_flags,
                       &list_select_idxs,
                       &list_select_flag);

    framework::AttributeMap attrs = {{"axes", axes},
                                     {"starts", starts},
                                     {"ends", ends},
                                     {"steps", steps},
                                     {"decrease_axes", decrease_axes},
                                     {"none_axes", none_axes}};

    if (egr::Controller::Instance().HasGrad()) {
      PADDLE_ENFORCE_EQ(
          egr::EagerUtils::IsLeafTensor(self->tensor) &&
              !egr::EagerUtils::autograd_meta(&self->tensor)->StopGradient(),
          false,
          platform::errors::InvalidArgument(
              "Leaf Tensor (%s) that doesn't stop gradient can't use "
              "inplace strategy.",
              self->tensor.name()));
    }

    paddle::Tensor value_tensor;

    if (PyCheckTensor(value_obj)) {
      value_tensor = reinterpret_cast<TensorObject*>(value_obj)->tensor;
    } else if (py::isinstance<py::array>(value_obj)) {
      paddle::Tensor value_tensor_tmp(
          std::make_shared<phi::DenseTensor>(),
          egr::Controller::Instance().GenerateUniqueName());
      py::object value_obj_tmp(py::handle(value_obj), true);
      py::object value = value_obj_tmp;
      if (self->tensor.dtype() == phi::DataType::FLOAT32) {
        if (!py::isinstance<py::array_t<float>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<float>(value_obj_tmp);
        }
      } else if (self->tensor.dtype() == phi::DataType::FLOAT64) {
        if (!py::isinstance<py::array_t<double>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<double>(value_obj_tmp);
        }
      } else if (self->tensor.dtype() == phi::DataType::INT32) {
        if (!py::isinstance<py::array_t<int32_t>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<int32_t>(value_obj_tmp);
        }
      } else if (self->tensor.dtype() == phi::DataType::INT64) {
        if (!py::isinstance<py::array_t<int64_t>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<int64_t>(value_obj_tmp);
        }
      } else if (self->tensor.dtype() == phi::DataType::BOOL) {
        if (!py::isinstance<py::array_t<bool>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<bool>(value_obj_tmp);
        }
      } else if (self->tensor.dtype() == phi::DataType::COMPLEX64) {
        if (!py::isinstance<py::array_t<std::complex<float>>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<std::complex<float>>(
              value_obj_tmp);
        }
      } else if (self->tensor.dtype() == phi::DataType::COMPLEX128) {
        if (!py::isinstance<py::array_t<std::complex<double>>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<std::complex<double>>(
              value_obj_tmp);
        }
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "When assign a numpy.np value to a paddle.Tensor, "
            "the data type of the paddle.Tensor must be bool, "
            "float32, float64, complex64, complex128, int32 or int64, "
            "please check the type of tensor."));
      }

      SetTensorFromPyArray(
          static_cast<phi::DenseTensor*>(value_tensor_tmp.impl().get()),
          value,
          self->tensor.place(),
          false);

      value_tensor = value_tensor_tmp;
    } else {
      py::object value_obj_tmp(py::handle(value_obj), true);
      // convert the value to self data type
      if (py::isinstance<py::float_>(value_obj_tmp) ||
          py::isinstance<py::int_>(value_obj_tmp) ||
          py::isinstance<py::bool_>(value_obj_tmp) ||
          PyComplex_Check(value_obj)) {
        if (self->tensor.dtype() == phi::DataType::FLOAT32 ||
            self->tensor.dtype() == phi::DataType::FLOAT16) {
          attrs["values"] = std::vector<paddle::experimental::Scalar>{
              value_obj_tmp.cast<float>()};
        } else if (self->tensor.dtype() == phi::DataType::FLOAT64) {
          attrs["values"] = std::vector<paddle::experimental::Scalar>{
              value_obj_tmp.cast<double>()};
        } else if (self->tensor.dtype() == phi::DataType::INT32) {
          attrs["values"] = std::vector<paddle::experimental::Scalar>{
              value_obj_tmp.cast<int32_t>()};
        } else if (self->tensor.dtype() == phi::DataType::INT64) {
          attrs["values"] = std::vector<paddle::experimental::Scalar>{
              value_obj_tmp.cast<int64_t>()};
        } else if (self->tensor.dtype() == phi::DataType::BOOL) {
          attrs["values"] = std::vector<paddle::experimental::Scalar>{
              value_obj_tmp.cast<bool>()};
        } else if (self->tensor.dtype() == phi::DataType::COMPLEX64) {
          attrs["values"] = std::vector<paddle::experimental::Scalar>{
              value_obj_tmp.cast<std::complex<float>>()};
        } else if (self->tensor.dtype() == phi::DataType::COMPLEX128) {
          attrs["values"] = std::vector<paddle::experimental::Scalar>{
              value_obj_tmp.cast<std::complex<double>>()};
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "When assign a value to a paddle.Tensor, "
              "the data type of the paddle.Tensor must be bool, "
              "float32, float64, complex64, complex128, int32, int64 or "
              "float16, "
              "please check the type of tensor."));
        }
        attrs["shape"] = std::vector<int64_t>{1};

      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Value type error. The assign value allows "
            "numpy.ndarray, integer, float, complex  or bool, "
            "but received %s.",
            Py_TYPE(value_obj)));
      }
    }
    {
      // Release gil and do tracing
      py::gil_scoped_release release;
      // use inplace set_value_ operator
      if (value_tensor.initialized() &&
          (self->tensor.dtype() != value_tensor.dtype())) {
        if (egr::Controller::Instance().GetAMPLevel() !=
            paddle::imperative::AmpLevel::O0) {
          paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
              tmps = {{self->tensor}, {value_tensor}};
          auto amp_dtype = egr::GetAmpDestDtype("set_value", tmps);
          self->tensor = egr::EagerAmpAutoCast(
              self->tensor.name(), self->tensor, amp_dtype, "set_value");
          value_tensor = egr::EagerAmpAutoCast(
              value_tensor.name(), value_tensor, amp_dtype, "set_value");
        }
        if (self->tensor.dtype() != value_tensor.dtype()) {
          value_tensor = cast_ad_func(value_tensor, self->tensor.dtype());
        }
      }
      self->tensor = set_value__dygraph_function(
          self->tensor, value_tensor, {}, {}, {}, attrs);
    }
    if (PyCheckTensor(value_obj)) {
      // pass the stop_gradient from value to tensor.
      // pass stop gradient should be done after CheckInplace in
      // set_value__dygraph_function.
      if (!egr::EagerUtils::autograd_meta(&value_tensor)->StopGradient() &&
          egr::EagerUtils::autograd_meta(&self->tensor)->StopGradient()) {
        egr::EagerUtils::autograd_meta(&self->tensor)->SetStopGradient(false);
      }
    }
  } else {
    auto self_numpy = TensorToPyArray(*self_tensor, true);
    VLOG(4) << "parse_index is false";
    if (PyCheckTensor(_index)) {
      VLOG(4) << "index is tensor";
      auto index_tensor = static_cast<phi::DenseTensor*>(
          reinterpret_cast<TensorObject*>(_index)->tensor.impl().get());
      auto index_numpy = TensorToPyArray(*index_tensor);
      self_numpy[index_numpy] = py::object(py::handle(value_obj), true);
    } else {
      VLOG(4) << "index is not tensor";
      self_numpy[_index] = py::object(py::handle(value_obj), true);
    }
    if (!self->tensor.initialized()) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      SetTensorFromPyArray(self_tensor,
                           self_numpy,
                           platform::Place(platform::CUDAPlace(0)),
                           false);
#else
      SetTensorFromPyArray(self_tensor,
                           self_numpy,
                           platform::Place(platform::CPUPlace()),
                           false);
#endif
    } else {
      SetTensorFromPyArray(
          self_tensor, self_numpy, self->tensor.place(), false);
    }
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_register_grad_hook(TensorObject* self,
                                           PyObject* args,
                                           PyObject* kwargs) {
  EAGER_TRY
  int64_t hook_id;
  if (egr::EagerUtils::IsLeafTensor(self->tensor)) {
    VLOG(6) << "Register hook for leaf tensor: " << self->tensor.name();

    auto autograd_meta = egr::EagerUtils::unsafe_autograd_meta(self->tensor);

    if (autograd_meta && !autograd_meta->StopGradient()) {
      if (!autograd_meta->GetMutableGradNode()) {
        VLOG(6) << "Detected nullptr grad_node, Leaf tensor should have had "
                   "grad_node with type: GradNodeAccumulation.";
        autograd_meta->SetGradNode(
            std::make_shared<egr::GradNodeAccumulation>(autograd_meta));
      }
    }

    std::shared_ptr<egr::GradNodeBase> grad_node =
        egr::EagerUtils::grad_node(self->tensor);
    auto rank_info =
        egr::EagerUtils::unsafe_autograd_meta(self->tensor)->OutRankInfo();
    PyObject* hook_func = PyTuple_GET_ITEM(args, 0);

    auto accumulation_grad_node =
        std::dynamic_pointer_cast<egr::GradNodeAccumulation>(grad_node);
    hook_id = accumulation_grad_node->RegisterGradientHook(
        rank_info.first,
        rank_info.second,
        std::make_shared<PyTensorHook>(hook_func));

  } else {
    VLOG(6) << "Register hook for non leaf tensor: " << self->tensor.name();
    std::shared_ptr<egr::GradNodeBase> grad_node =
        egr::EagerUtils::grad_node(self->tensor);
    auto rank_info =
        egr::EagerUtils::unsafe_autograd_meta(self->tensor)->OutRankInfo();

    PyObject* hook_func = PyTuple_GET_ITEM(args, 0);

    hook_id = grad_node->RegisterGradientHook(
        rank_info.first,
        rank_info.second,
        std::make_shared<PyTensorHook>(hook_func));
  }
  return ToPyObject(hook_id);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_remove_grad_hook(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  VLOG(6) << "Remove the registered hook for tensor: " << self->tensor.name();
  std::shared_ptr<egr::GradNodeBase> grad_node =
      egr::EagerUtils::grad_node(self->tensor);

  int64_t hook_id = pybind::CastPyArg2AttrLong(PyTuple_GET_ITEM(args, 0), 0);

  return ToPyObject(grad_node->RemoveGradientHook(hook_id));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_inplace_assign(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  VLOG(6) << "inplace assign for tensor:" << self->tensor.name();
  PyObject* other = PyTuple_GET_ITEM(args, 0);
  PyObject* self_obj = reinterpret_cast<PyObject*>(self);
  ShareTensor(self_obj, other);
  RETURN_PY_NONE;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method__register_reduce_hook__doc__,  // NOLINT
             R"DOC(_register_backward_hook($self, hook, /)
--

Registers a backward hook for current Tensor.

This hook will be called every time the gradient of current Tensor has been fully calculated.

There are two differences with `_register_grad_hook`:
1. This backward hook will be executed after the gradient accumulation completed across batches,
  but the hook registered by `_register_grad_hook` will be executed the gradient accumulation
  completed in current batch.
2. This backward hook function should have the following signature:

    hook() -> None

  It requires no input and no return value.

Args:
    hook(function): A backward hook to be registered for Tensor.gradient

Returns:
    None
)DOC");
static PyObject* tensor_register_reduce_hook(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Register reduce hook for tensor: " << self->tensor.name();

  std::shared_ptr<egr::GradNodeBase> grad_node =
      egr::EagerUtils::grad_node(self->tensor);
  PADDLE_ENFORCE_EQ(egr::EagerUtils::IsLeafTensor(self->tensor),
                    true,
                    platform::errors::InvalidArgument(
                        "Only can register backward hook for leaf Tensor."));
  PADDLE_ENFORCE_EQ(
      !egr::EagerUtils::unsafe_autograd_meta(self->tensor)->StopGradient(),
      true,
      platform::errors::InvalidArgument(
          "Cannot register backward hook on a Tensor that stop "
          "gradient."));
  PADDLE_ENFORCE(
      grad_node.get() != nullptr,
      paddle::platform::errors::Fatal("Detected nullptr grad_node,"
                                      "Leaf tensor should have had grad_node "
                                      "with type: GradNodeAccumulation."));
  PyObject* hook_func = PyTuple_GET_ITEM(args, 0);

  auto accumulation_grad_node =
      std::dynamic_pointer_cast<egr::GradNodeAccumulation>(grad_node);
  accumulation_grad_node->RegisterReduceHook(
      std::make_shared<PyVoidHook>(hook_func));

  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__set_grad_type(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  auto var_type = pybind::CastPyArg2ProtoType(PyTuple_GET_ITEM(args, 0), 0);
  auto grad_tensor =
      egr::EagerUtils::autograd_meta(&self->tensor)->MutableGrad();
  if (var_type == framework::proto::VarType::LOD_TENSOR) {
    grad_tensor->set_impl(std::make_shared<phi::DenseTensor>());
  } else if (var_type == framework::proto::VarType::SELECTED_ROWS) {
    grad_tensor->set_impl(std::make_shared<phi::SelectedRows>());
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__clear(TensorObject* self,
                               PyObject* args,
                               PyObject* kwargs) {
  EAGER_TRY
  self->tensor.reset();
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__clear_dataptr(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  self->tensor.set_impl(nullptr);
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__copy_gradient_from(TensorObject* self,
                                            PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  auto src = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  if (self->tensor.initialized()) {
    PADDLE_ENFORCE_EQ(self->tensor.dtype(),
                      src.dtype(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different data type with Tensor %s",
                          self->tensor.name(),
                          src.name()));
    PADDLE_ENFORCE_EQ(self->tensor.impl()->type_info().id(),
                      src.impl()->type_info().id(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different type with Tensor %s, Tensor "
                          "ShareGradientDataWith cannot be performed!",
                          self->tensor.name(),
                          src.name()));
  }
  VLOG(6) << "Tensor copy gradient from: " << src.name();
  auto* p_grad = egr::EagerUtils::mutable_grad(self->tensor);
  if (p_grad) {
    PADDLE_ENFORCE_EQ(src.initialized(),
                      true,
                      platform::errors::InvalidArgument(
                          "Tensor %s has not been initialized", src.name()));
    p_grad->set_impl(src.impl());
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__use_gpudnn(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.defined() && self->tensor.is_dense_tensor(),
                 paddle::platform::errors::Fatal(
                     "function _use_gpudnn is only effective for DenseTensor"));

  bool use_gpudnn = pybind::CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 0), 0);

  // Set the same use_gpudnn attribute, return directly
  phi::DenseTensor* dense_tensor =
      static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  phi::DenseTensorMeta* dense_tensor_meta =
      phi::DenseTensorUtils::GetMutableMeta(dense_tensor);
  if (use_gpudnn == dense_tensor_meta->use_gpudnn) {
    return ToPyObject(self->tensor);
  }

  // Share all other members of Tensor except use_gpudnn
  phi::DenseTensorMeta target_dense_meta = *dense_tensor_meta;
  target_dense_meta.use_gpudnn = use_gpudnn;
  phi::DenseTensor target_dense_tensor;
  target_dense_tensor.ShareDataWith(*dense_tensor);
  target_dense_tensor.set_meta(target_dense_meta);
  // Construct returned tensor
  paddle::Tensor target_tensor(
      std::make_shared<phi::DenseTensor>(target_dense_tensor),
      self->tensor.name());
  target_tensor.set_autograd_meta(self->tensor.mutable_autograd_meta());
  VLOG(4) << "Tensor: " << target_tensor.name()
          << " set use_gpudnn = " << use_gpudnn;

  return ToPyObject(target_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_set_vocab(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  using Vocab = paddle::framework::Vocab;
  auto vocab = CastPyArg2Vocab(PyTuple_GET_ITEM(args, 0), 0);
  auto var_tensor = std::make_shared<egr::VariableCompatTensor>();
  *var_tensor->GetMutable<Vocab>() = vocab;
  self->tensor.set_impl(var_tensor);
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_set_string_list(TensorObject* self,
                                               PyObject* args,
                                               PyObject* kwargs) {
  EAGER_TRY
  using Strings = paddle::framework::Strings;
  auto strings = CastPyArg2VectorOfString(PyTuple_GET_ITEM(args, 0), 0);
  auto var_tensor = std::make_shared<egr::VariableCompatTensor>();
  *var_tensor->GetMutable<Strings>() = strings;
  self->tensor.set_impl(var_tensor);
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_get_map_tensor(TensorObject* self,
                                              PyObject* args,
                                              PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE_EQ(
      egr::IsVariableCompatTensor(self->tensor),
      true,
      paddle::platform::errors::Fatal(
          "this method is only effective for VariableCompatTensor"));
  using Vocab = paddle::framework::Vocab;
  auto* var_tensor =
      static_cast<const egr::VariableCompatTensor*>(self->tensor.impl().get());
  return ToPyObject(var_tensor->Get<Vocab>());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_nnz__doc__,
             R"DOC(nnz($self, /)
--

Note:
    **This API is only available for SparseCooTensor or SparseCsrTensor.**

Returns the total number of non zero elements in input SparseCooTensor/SparseCsrTensor.

Returns:
    int

Examples:
    .. code-block:: python

        import paddle

        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        coo.nnz()
        # 3

)DOC");

static PyObject* tensor_method_get_non_zero_nums(TensorObject* self,
                                                 PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(
      self->tensor.is_sparse_coo_tensor() ||
          self->tensor.is_sparse_csr_tensor(),
      paddle::platform::errors::Fatal("this method is only effective for "
                                      "SparseCooTensor or SparseCsrTensor"));
  if (self->tensor.is_sparse_coo_tensor()) {
    auto sparse_coo_tensor =
        std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
    return ToPyObject(sparse_coo_tensor->nnz());
  } else {
    auto sparse_csr_tensor =
        std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
    return ToPyObject(sparse_csr_tensor->nnz());
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_indices__doc__,
             R"DOC(indices($self, /)
--

Note:
    **This API is only available for SparseCooTensor.**

Returns the indices of non zero elements in input SparseCooTensor.

Returns:
    DenseTesnor

Examples:
    .. code-block:: python

        import paddle

        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        coo.indices()
        # Tensor(shape=[2, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
        #        [[0, 1, 2],
        #         [1, 2, 0]])

)DOC");

static PyObject* tensor_method_get_non_zero_indices(TensorObject* self,
                                                    PyObject* args,
                                                    PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_coo_tensor(),
                 paddle::platform::errors::Fatal(
                     "this method is only effective for SparseCooTensor"));
  auto sparse_coo_tensor =
      std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
  paddle::Tensor tensor(std::make_shared<phi::DenseTensor>(
      sparse_coo_tensor->non_zero_indices()));
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_values__doc__,
             R"DOC(values($self, /)
--

Note:
    **This API is only available for SparseCooTensor or SparseCsrTensor.**

Returns the values of non zero elements in input SparseCooTensor.

Returns:
    DenseTesnor

Examples:
    .. code-block:: python

        import paddle

        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        coo.values()
        # Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
        #        [1., 2., 3.])

)DOC");

static PyObject* tensor_method_get_non_zero_elements(TensorObject* self,
                                                     PyObject* args,
                                                     PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(
      self->tensor.is_sparse_coo_tensor() ||
          self->tensor.is_sparse_csr_tensor(),
      paddle::platform::errors::Fatal("this method is only effective for "
                                      "SparseCooTensor or SparseCsrTensor"));
  if (self->tensor.is_sparse_coo_tensor()) {
    auto sparse_coo_tensor =
        std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
    paddle::Tensor tensor(std::make_shared<phi::DenseTensor>(
        sparse_coo_tensor->non_zero_elements()));
    return ToPyObject(tensor);
  } else {
    auto sparse_csr_tensor =
        std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
    paddle::Tensor tensor(std::make_shared<phi::DenseTensor>(
        sparse_csr_tensor->non_zero_elements()));
    return ToPyObject(tensor);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_crows__doc__,
             R"DOC(crows($self, /)
--

Note:
    **This API is only available for SparseCsrTensor.**

Returns the compressed row index of non zero elements in input SparseCsrTensor.

Returns:
    DenseTesnor

Examples:
    .. code-block:: python

        import paddle

        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
        csr.crows()
        # Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
        #        [0, 2, 3, 5])

)DOC");

static PyObject* tensor_method_get_non_zero_crows(TensorObject* self,
                                                  PyObject* args,
                                                  PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_csr_tensor(),
                 paddle::platform::errors::Fatal(
                     "this method is only effective for SparseCsrTensor"));
  auto sparse_csr_tensor =
      std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
  paddle::Tensor tensor(
      std::make_shared<phi::DenseTensor>(sparse_csr_tensor->non_zero_crows()));
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_cols__doc__,
             R"DOC(cols($self, /)
--

Note:
    **This API is only available for SparseCsrTensor.**

Returns the column index of non zero elements in input SparseCsrTensor.

Returns:
    DenseTesnor

Examples:
    .. code-block:: python

        import paddle

        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
        csr.cols()
        # Tensor(shape=[5], dtype=int64, place=Place(gpu:0), stop_gradient=True,
        #        [1, 3, 2, 0, 1])

)DOC");

static PyObject* tensor_method_get_non_zero_cols(TensorObject* self,
                                                 PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_csr_tensor(),
                 paddle::platform::errors::Fatal(
                     "this method is only effective for SparseCsrTensor"));
  auto sparse_csr_tensor =
      std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
  paddle::Tensor tensor(
      std::make_shared<phi::DenseTensor>(sparse_csr_tensor->non_zero_cols()));
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_is_dense__doc__, R"DOC(is_dense($self, /)
--

Whether the Tensor is a Dense Tensor.

Returns:
    Whether the Tensor is a Dense Tensor.

Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([1.0], stop_gradient=False)
        print(x.is_dense())
)DOC");

static PyObject* tensor_method_is_dense(TensorObject* self,
                                        PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_dense_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_is_dist__doc__, R"DOC(is_dist($self, /)
--

Whether the Tensor is a Distributed Tensor.

Returns:
    Whether the Tensor is a Distributed Tensor.

Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([1.0], stop_gradient=False)
        print(x.is_dist()) # False
)DOC");

static PyObject* tensor_method_is_dist(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_dist_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_sparse__doc__,
             R"DOC(is_sparse($self, /)
--

Returns whether the input Tensor is SparseCooTensor or SparseCsrTensor.

When input is SparseCooTensor/SparseCsrTensor, will return True. When input is DenseTensor, will return False.

Returns:
    bool

Examples:
    .. code-block:: python

        import paddle

        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        coo.is_sparse()
        # True

)DOC");
static PyObject* tensor_method_is_sparse(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_sparse_coo_tensor() ||
                    self->tensor.is_sparse_csr_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_sparse_coo__doc__,
             R"DOC(is_sparse_coo($self, /)
--

Returns whether the input Tensor is SparseCooTensor.

When input is SparseCooTensor, will return True. When input is DenseTensor/SparseCsrTensor, will return False.

Returns:
    bool

Examples:
    .. code-block:: python

        import paddle

        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        coo.is_sparse_coo()
        # True

)DOC");

static PyObject* tensor_method_is_sparse_coo(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_sparse_coo_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_sparse_csr__doc__,
             R"DOC(is_sparse_csr($self, /)
--

Returns whether the input Tensor is SparseCsrTensor.

When input is SparseCsrTensor, will return True. When input is DenseTensor/SparseCooTensor, will return False.

Returns:
    bool

Examples:
    .. code-block:: python

        import paddle

        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
        csr.is_sparse_csr()
        # True

)DOC");

static PyObject* tensor_method_is_sparse_csr(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_sparse_csr_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_to_sparse_csr__doc__,
             R"DOC(to_sparse_csr($self, /)
--

Note:
    **This API is only available for DenseTensor or SparseCooTensor.**

Convert input Tensor to SparseCsrTensor.

When input is SparseCooTensor, will convert `COO` to `CSR` . When input is DenseTensor, will convert `Dense` to `CSR` .

Returns:
    SparseCsrTensor

Examples:
    .. code-block:: python

        import paddle

        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        coo.to_sparse_csr()
        # Tensor(shape=[3, 3], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
        #        crows=[0, 1, 2, 3],
        #        cols=[1, 2, 0],
        #        values=[1., 2., 3.])

)DOC");

static PyObject* tensor_method_to_sparse_csr(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  auto csr_tensor = self->tensor.to_sparse_csr();
  egr::EagerUtils::autograd_meta(&csr_tensor)
      ->SetStopGradient(
          egr::EagerUtils::autograd_meta(&self->tensor)->StopGradient());
  egr::EagerUtils::autograd_meta(&csr_tensor)
      ->SetPersistable(
          egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  return ToPyObject(csr_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_same_shape__doc__,
             R"DOC(is_same_shape($self, y, /)
--

Return the results of shape comparison between two Tensors, check whether x.shape equal to y.shape.
Any two type Tensor among DenseTensor/SparseCooTensor/SparseCsrTensor are supported.

Args:
    x (Tensor): The input tensor. It can be DenseTensor/SparseCooTensor/SparseCsrTensor.
    y (Tensor): The input tensor. It can be DenseTensor/SparseCooTensor/SparseCsrTensor.

Returns:
    bool: True for same shape and False for different shape.

Examples:

    .. code-block:: python

        import paddle

        x = paddle.rand([2, 3, 8])
        y = paddle.rand([2, 3, 8])
        y = y.to_sparse_csr()
        z = paddle.rand([2, 5])

        x.is_same_shape(y)
        # True
        x.is_same_shape(z)
        # False

)DOC");

static PyObject* tensor_method_is_same_shape(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  auto other = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  return ToPyObject(self->tensor.shape() == other.shape());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__inplace_version(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  uint32_t inplace_version = self->tensor.current_inplace_version();

  return ToPyObject(inplace_version);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method_element_size__doc__,  // NOLINT
             R"DOC(element_size($self, /)
--

Returns the size in bytes of an element in the Tensor.

Returns:
    int, The size in bytes of an element in the Tensor.

Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor(1, dtype='bool')
        x.element_size() # 1

        x = paddle.to_tensor(1, dtype='float16')
        x.element_size() # 2

        x = paddle.to_tensor(1, dtype='float32')
        x.element_size() # 4

        x = paddle.to_tensor(1, dtype='float64')
        x.element_size() # 8

        x = paddle.to_tensor(1, dtype='complex128')
        x.element_size() # 16
)DOC");

static PyObject* tensor_method_element_size(TensorObject* self,
                                            PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  uint32_t element_size = phi::SizeOf(self->tensor.dtype());

  return ToPyObject(element_size);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_method__bump_inplace_version__doc__,  // NOLINT
             R"DOC(_bump_inplace_version($self, /)
--

Note:
    **This API is ONLY available in Dygraph mode.**
    **This is a very low level API. Users should not use it directly. **

  Bump the version whenever the Tensor is modified through an inplace operation.
)DOC");
static PyObject* tensor__bump_inplace_version(TensorObject* self,
                                              PyObject* args,
                                              PyObject* kwargs) {
  EAGER_TRY
  self->tensor.bump_inplace_version();
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_is_selected_rows(TensorObject* self,
                                                PyObject* args,
                                                PyObject* kwargs) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    return ToPyObject(false);
  }
  return ToPyObject(self->tensor.is_selected_rows());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_get_rows(TensorObject* self,
                                        PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_selected_rows(),
                 paddle::platform::errors::Fatal(
                     "this method is only effective for SelectedRows"));
  auto selected_rows =
      std::dynamic_pointer_cast<phi::SelectedRows>(self->tensor.impl());
  return ToPyObject(selected_rows->rows());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__reset_grad_inplace_version(TensorObject* self,
                                                    PyObject* args,
                                                    PyObject* kwargs) {
  EAGER_TRY
  Py_ssize_t args_num = PyTuple_Size(args);
  bool set_to_zero = true;
  if (args_num == (Py_ssize_t)1) {
    set_to_zero = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 0), 0);
  }

  paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
  if (grad && grad->defined() && grad->is_dense_tensor() &&
      grad->initialized()) {
    grad->reset_inplace_version(set_to_zero);
  }
  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__share_memory(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
#ifndef _WIN32
  PADDLE_ENFORCE_EQ(platform::is_cpu_place(self->tensor.place()),
                    true,
                    platform::errors::InvalidArgument(
                        "Sharing memory only support CPU Tensor currently"));
  // 1. get LoDTensor
  auto* t =
      std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl()).get();
  // 2. allocate shared memory
  void* data_ptr = t->data();
  size_t data_size =
      t->numel() *
      framework::SizeOfType(framework::TransToProtoVarType(t->dtype()));
  auto shared_writer_holder =
      memory::allocation::AllocateMemoryMapWriterAllocation(data_size);
  // 3. maintain mmap fd set & backup ipc_name
  const std::string& ipc_name = shared_writer_holder->ipc_name();
  memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);
  // 4. copy data & reset holder
  memory::Copy(platform::CPUPlace(),
               shared_writer_holder->ptr(),
               platform::CPUPlace(),
               data_ptr,
               data_size);
  t->ResetHolder(shared_writer_holder);
  return ToPyObject(t);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "Sharing memory in Windows OS is not supported currently"));
  RETURN_PY_NONE

#endif
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__offset(TensorObject* self,
                                PyObject* args,
                                PyObject* kwargs) {
  EAGER_TRY
  auto t = std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
  PADDLE_ENFORCE_EQ(
      t->IsInitialized(),
      true,
      platform::errors::InvalidArgument("Tensor %s has not been initialized!",
                                        self->tensor.name()));

  return ToPyObject(t->offset());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__grad_name(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE_EQ(
      grad != nullptr,
      true,
      platform::errors::InvalidArgument(
          "Detected nullptr grad. Please check if you have manually "
          "cleared the grad inside autograd_meta"));
  return ToPyObject(grad->name());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__grad_value(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE_EQ(
      grad != nullptr,
      true,
      platform::errors::InvalidArgument(
          "Detected nullptr grad. Please check if you have manually "
          "cleared the grad inside autograd_meta"));

  if (!grad->defined()) {
    RETURN_PY_NONE
  }
  if (grad->is_dense_tensor()) {
    auto* grad_tensor = static_cast<phi::DenseTensor*>(grad->impl().get());
    return ToPyObject(grad_tensor);
  } else {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "this method is only supported for DenseTensor"));
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__local_value(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.is_dist_tensor()) {
#ifdef PADDLE_WITH_DISTRIBUTE
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(self->tensor.impl().get());
    paddle::Tensor result(
        std::make_shared<phi::DenseTensor>(dist_tensor->value()));
    return ToPyObject(result);
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "The `_local_value` method of (Dist)Tensor is not supported "
        "in the current PaddlePaddle, please recompile and install "
        "PaddlePaddle "
        "with the option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__unset_fake_empty(TensorObject* self,
                                          PyObject* args,
                                          PyObject* kwargs) {
  EAGER_TRY
  paddle::Tensor* grad = egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE_EQ(
      grad != nullptr,
      true,
      platform::errors::InvalidArgument(
          "Detected nullptr grad. Please check if you have manually "
          "cleared the grad inside autograd_meta"));

  bool is_leaf = egr::EagerUtils::IsLeafTensor(self->tensor);
  if (is_leaf) {
    std::static_pointer_cast<egr::GradNodeAccumulation>(
        egr::EagerUtils::grad_node(self->tensor))
        ->SetFakeEmpty(false);
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_data_ptr__doc__,
             R"DOC(data_ptr($self, /)
--

Returns the address of the first element of current Tensor.

Returns:
    int, The address of the first element of current Tensor.

Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([1, 2, 3])
        print(x.data_ptr())
)DOC");

static PyObject* tensor_data_ptr(TensorObject* self,
                                 PyObject* args,
                                 PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.initialized() && self->tensor.is_dense_tensor()) {
    return ToPyObject(
        (int64_t)std::dynamic_pointer_cast<phi::DenseTensor>(  // NOLINT
            self->tensor.impl())
            ->data());
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__grad_ivar(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  EAGER_TRY
  VLOG(6) << "Get grad for tensor: " << self->tensor.name();
  auto meta = egr::EagerUtils::nullable_autograd_meta(self->tensor);
  VLOG(6) << meta << " initialized: " << meta->Grad().initialized();
  if (meta && meta->Grad().initialized()) {
    return ToPyObject(meta->Grad());
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_get_strides__doc__,
             R"DOC(get_strides($self, /)
--

Returns the strides of current Tensor.

Returns:
    List, the strides of current Tensor.

Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([1, 2, 3])
        y = x[1]
        print(y.get_strides())
)DOC");

static PyObject* tensor_method_strides(TensorObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  std::vector<int64_t> value;
  if (!self->tensor.defined() || !self->tensor.is_dense_tensor()) {
    return ToPyObject(value);
  }
  auto stride = self->tensor.strides();
  int rank = static_cast<int>(stride.size());
  value.resize(rank);
  for (int i = 0; i < rank; i++) {
    value[i] = stride[i];
  }
  return ToPyObject(value);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_contiguous__doc__,
             R"DOC(contiguous($self, /)
--

Returns a contiguous in memory tensor containing the same data as current Tensor.
If self tensor is already contiguous, this function returns the current Tensor.

Returns:
    Tensor, The contiguous Tensor.

Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([1, 2, 3])
        y = x[1]
        y = y.contiguous()
        print(y)
)DOC");

static PyObject* tensor_contiguous(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.is_dense_tensor()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
    if (dense_tensor->meta().is_contiguous()) {
      Py_INCREF(self);
      return reinterpret_cast<PyObject*>(self);
    } else {
      eager_gil_scoped_release guard;
      self->tensor.set_impl(std::make_shared<phi::DenseTensor>(std::move(
          paddle::experimental::Trans2Contiguous(*(dense_tensor.get())))));
      Py_INCREF(self);
      return reinterpret_cast<PyObject*>(self);
    }

  } else {
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyDoc_STRVAR(tensor_is_contiguous__doc__,
             R"DOC(is_contiguous($self, /)
--

Whether the Tensor is contiguous.

Returns:
    Bool, Whether the Tensor is contiguous.

Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([1, 2, 3])
        y = x[1]
        print(y.is_contiguous())
)DOC");
static PyObject* tensor_is_contiguous(TensorObject* self,
                                      PyObject* args,
                                      PyObject* kwargs) {
  EAGER_TRY
  if (self->tensor.is_dense_tensor()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
    return ToPyObject(dense_tensor->meta().is_contiguous());
  } else {
    return ToPyObject(true);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__set_impl(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Running in tensor_method__set_impl: set Tensor impl form the "
             "other Tensor.";
  auto tensor = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  self->tensor.set_impl(tensor.impl());
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

#if defined(PADDLE_WITH_CUDA)
static PyObject* tensor_method__uva(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Running in tensor_method__uva.";
  PADDLE_ENFORCE_EQ(self->tensor.is_dense_tensor(),
                    true,
                    platform::errors::InvalidArgument(
                        "Unified virtual addressing only support "
                        "DenseTensor currently."));
  PADDLE_ENFORCE_EQ(platform::is_cpu_place(self->tensor.place()),
                    true,
                    platform::errors::InvalidArgument(
                        "Unified virtual addressing only support "
                        "CPU Tensor currently."));
  int device_id = pybind::CastPyArg2AttrLong(PyTuple_GET_ITEM(args, 0), 0);
  auto* self_tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  tensor_uva(self_tensor, device_id);

  RETURN_PY_NONE

  EAGER_CATCH_AND_THROW_RETURN_NULL
}
#endif
static PyObject* tensor_method__is_string_tensor_hold_allocation(
    TensorObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  auto string_tensor =
      std::dynamic_pointer_cast<phi::StringTensor>(self->tensor.impl());
  if (string_tensor) {
    return ToPyObject(string_tensor->initialized());
  } else {
    return ToPyObject(false);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef variable_methods[] = {  // NOLINT
    {"numpy",
     (PyCFunction)(void (*)())tensor_method_numpy,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_numpy__doc__},
    {"_is_initialized",
     (PyCFunction)(void (*)())tensor_method__is_initialized,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_dense_tensor_hold_allocation",
     (PyCFunction)(void (*)(
         void))tensor_method__is_dense_tensor_hold_allocation,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_copy_to",
     (PyCFunction)(void (*)())tensor_method__copy_to,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"copy_",
     (PyCFunction)(void (*)())tensor_method_copy_,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"clone",
     (PyCFunction)(void (*)())tensor_method_clone,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_clone__doc__},
    {"reconstruct_from_",
     (PyCFunction)(void (*)())tensor_method_reconstruct_from_,
     METH_VARARGS | METH_KEYWORDS,
     tensor_reconstruct_from___doc__},
    {"retain_grads",
     (PyCFunction)(void (*)())tensor_retain_grads,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_retain_grads__doc__},
    {"clear_gradient",
     (PyCFunction)(void (*)())tensor_clear_gradient,
     METH_VARARGS | METH_KEYWORDS,
     tensor_clear_gradient__doc__},
    {"is_dense",
     (PyCFunction)(void (*)())tensor_method_is_dense,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_is_dense__doc__},
    {"is_dist",
     (PyCFunction)(void (*)())tensor_method_is_dist,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_is_dist__doc__},
    {"_zero_grads",
     (PyCFunction)(void (*)())tensor__zero_grads,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_share_buffer_to",
     (PyCFunction)(void (*)())tensor__share_buffer_to,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_shared_buffer_with",
     (PyCFunction)(void (*)())tensor__is_shared_buffer_with,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_share_underline_tensor_to",
     (PyCFunction)(void (*)())tensor__share_underline_tensor_to,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_shared_underline_tensor_with",
     (PyCFunction)(void (*)())tensor__is_shared_underline_tensor_with,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"detach",
     (PyCFunction)(void (*)())tensor_method_detach,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_detach__doc__},
    {"detach_",
     (PyCFunction)(void (*)(void))tensor_method_detach_,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_detach___doc__},
    {"get_tensor",
     (PyCFunction)(void (*)())tensor_method_get_underline_tensor,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_get_tensor__doc__},
    {"get_selected_rows",
     (PyCFunction)(void (*)())tensor_method_get_underline_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_get_tensor_from_selected_rows",
     (PyCFunction)(void (*)())tensor_method__get_tensor_from_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_getitem_index_not_tensor",
     (PyCFunction)(void (*)())tensor__getitem_index_not_tensor,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_getitem_from_offset",
     (PyCFunction)(void (*)())tensor__getitem_from_offset,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__setitem_eager_tensor__",
     (PyCFunction)(void (*)())tensor_method__setitem_eager_tensor,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_register_grad_hook",
     (PyCFunction)(void (*)())tensor_register_grad_hook,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_inplace_assign",  // NOTE(xiongkun03): only used in sot.
     (PyCFunction)(void (*)())tensor_inplace_assign,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_remove_grad_hook",
     (PyCFunction)(void (*)())tensor_remove_grad_hook,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_register_backward_hook",
     (PyCFunction)(void (*)())tensor_register_reduce_hook,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method__register_reduce_hook__doc__},
    {"_set_grad_type",
     (PyCFunction)(void (*)())tensor__set_grad_type,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_clear",
     (PyCFunction)(void (*)())tensor__clear,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_clear_dataptr",
     (PyCFunction)(void (*)())tensor__clear_dataptr,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_copy_gradient_from",
     (PyCFunction)(void (*)())tensor__copy_gradient_from,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_tensor_use_gpudnn",
     (PyCFunction)(void (*)())tensor__use_gpudnn,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    /** the methods to adapt old dygraph, will be removed in the future **/
    {"set_string_list",
     (PyCFunction)(void (*)())tensor_method_set_string_list,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"set_vocab",
     (PyCFunction)(void (*)())tensor_method_set_vocab,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"get_map_tensor",
     (PyCFunction)(void (*)())tensor_method_get_map_tensor,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    /***the method of sparse tensor****/
    {"nnz",
     (PyCFunction)(void (*)())tensor_method_get_non_zero_nums,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_nnz__doc__},
    {"indices",
     (PyCFunction)(void (*)())tensor_method_get_non_zero_indices,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_indices__doc__},
    {"values",
     (PyCFunction)(void (*)())tensor_method_get_non_zero_elements,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_values__doc__},
    {"crows",
     (PyCFunction)(void (*)())tensor_method_get_non_zero_crows,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_crows__doc__},
    {"cols",
     (PyCFunction)(void (*)())tensor_method_get_non_zero_cols,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_cols__doc__},
    {"is_sparse",
     (PyCFunction)(void (*)())tensor_method_is_sparse,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_sparse__doc__},
    {"is_sparse_coo",
     (PyCFunction)(void (*)())tensor_method_is_sparse_coo,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_sparse_coo__doc__},
    {"is_sparse_csr",
     (PyCFunction)(void (*)())tensor_method_is_sparse_csr,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_sparse_csr__doc__},
    {"is_same_shape",
     (PyCFunction)(void (*)())tensor_method_is_same_shape,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_same_shape__doc__},
    {"to_sparse_csr",
     (PyCFunction)(void (*)())tensor_method_to_sparse_csr,
     METH_VARARGS | METH_KEYWORDS,
     tensor_to_sparse_csr__doc__},
    /***the method of sparse tensor****/
    {"element_size",
     (PyCFunction)(void (*)())tensor_method_element_size,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method_element_size__doc__},
    {"_inplace_version",
     (PyCFunction)(void (*)())tensor__inplace_version,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_bump_inplace_version",
     (PyCFunction)(void (*)())tensor__bump_inplace_version,
     METH_VARARGS | METH_KEYWORDS,
     tensor_method__bump_inplace_version__doc__},
    {"is_selected_rows",
     (PyCFunction)(void (*)())tensor_method_is_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"rows",
     (PyCFunction)(void (*)())tensor_method_get_rows,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_reset_grad_inplace_version",
     (PyCFunction)(void (*)())tensor__reset_grad_inplace_version,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_share_memory",
     (PyCFunction)(void (*)())tensor_method__share_memory,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_offset",
     (PyCFunction)(void (*)())tensor__offset,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_grad_name",
     (PyCFunction)(void (*)())tensor__grad_name,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_grad_value",
     (PyCFunction)(void (*)())tensor__grad_value,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_local_value",
     (PyCFunction)(void (*)())tensor__local_value,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_unset_fake_empty",
     (PyCFunction)(void (*)())tensor__unset_fake_empty,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"data_ptr",
     (PyCFunction)(void (*)())tensor_data_ptr,
     METH_VARARGS | METH_KEYWORDS,
     tensor_data_ptr__doc__},
    {"_grad_ivar",
     (PyCFunction)(void (*)())tensor__grad_ivar,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"contiguous",
     (PyCFunction)(void (*)(void))tensor_contiguous,
     METH_VARARGS | METH_KEYWORDS,
     tensor_contiguous__doc__},
    {"is_contiguous",
     (PyCFunction)(void (*)(void))tensor_is_contiguous,
     METH_VARARGS | METH_KEYWORDS,
     tensor_is_contiguous__doc__},
    {"get_strides",
     (PyCFunction)(void (*)(void))tensor_method_strides,
     METH_VARARGS | METH_KEYWORDS,
     tensor_get_strides__doc__},
    {"_set_impl",
     (PyCFunction)(void (*)(void))tensor_method__set_impl,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
#if defined(PADDLE_WITH_CUDA)
    {"_tensor_uva",
     (PyCFunction)(void (*)())tensor_method__uva,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
#endif
    {nullptr, nullptr, 0, nullptr}};

// variable_methods for core.eager.StringTensor
PyMethodDef string_tensor_variable_methods[] = {  // NOLINT
    {"numpy",
     (PyCFunction)(void (*)())tensor_method_numpy_for_string_tensor,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_initialized",
     (PyCFunction)(void (*)())tensor_method__is_initialized,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_string_tensor_hold_allocation",
     (PyCFunction)(void (*)(
         void))tensor_method__is_string_tensor_hold_allocation,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    // TODO(zhoushunjie): Need to add _copy_to, copy_ for StringTensor.
    {nullptr, nullptr, 0, nullptr}};

}  // namespace pybind
}  // namespace paddle
