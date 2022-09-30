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
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace pybind {

extern void InitTensorWithNumpyValue(TensorObject* self,
                                     const pybind11::object& array,
                                     const paddle::platform::Place& place,
                                     bool zero_copy);

extern PyTypeObject* p_tensor_type;

Py_ssize_t GetSliceIndexFromPyObject(PyObject* obj) {
  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type))) {
    VLOG(6) << "Call GetSliceIndexFromTensor in Eager";
    paddle::experimental::Tensor tensor = CastPyArg2Tensor(obj, 0);
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
        "We should only get paddle::experimental::Tensor or VarBase in this "
        "method, when you reach this means we got another type index."));
  }
}

bool PyCheckTensor(PyObject* obj) {
  return PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type));
}

static PyObject* tensor_method_numpy(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  auto& api = pybind11::detail::npy_api::get();
  if (!self->tensor.impl()) {
    Py_intptr_t py_dims[paddle::framework::DDim::kMaxRank];
    Py_intptr_t py_strides[paddle::framework::DDim::kMaxRank];
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
  auto sizeof_dtype = paddle::framework::DataTypeSize(self->tensor.type());
  Py_intptr_t py_dims[paddle::framework::DDim::kMaxRank];
  Py_intptr_t py_strides[paddle::framework::DDim::kMaxRank];
  size_t numel = 1;
  for (int i = tensor_dims.size() - 1; i >= 0; --i) {
    py_dims[i] = static_cast<size_t>(tensor_dims[i]);
    py_strides[i] = sizeof_dtype * numel;
    numel *= py_dims[i];
  }

  PyObject* array = api.PyArray_NewFromDescr_(
      api.PyArray_Type_,
      api.PyArray_DescrFromType_(numpy_dtype),
      tensor_dims.size(),
      py_dims,
      py_strides,
      nullptr,
      pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
          pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
      nullptr);

  if (!self->tensor.impl()->initialized()) {
    if (tensor_dims.size() == 0) {
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

  if (self->tensor.is_cpu() || self->tensor.is_gpu_pinned()) {
    eager_gil_scoped_release guard;
    platform::CPUPlace place;
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor = static_cast<paddle::framework::LoDTensor*>(
          selected_rows->mutable_value());

      // deep copy
      paddle::memory::Copy(
          place,
          reinterpret_cast<void*>(pybind11::detail::array_proxy(array)->data),
          place,
          dense_tensor->data(),
          sizeof_dtype * numel);
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      // deep copy
      paddle::memory::Copy(
          place,
          reinterpret_cast<void*>(pybind11::detail::array_proxy(array)->data),
          place,
          dense_tensor->data(),
          sizeof_dtype * numel);
    }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (self->tensor.is_gpu()) {
    eager_gil_scoped_release guard;
#if defined(PADDLE_WITH_CUDA)
    gpuMemcpyKind kind = cudaMemcpyDeviceToHost;
#elif defined(PADDLE_WITH_HIP)
    gpuMemcpyKind kind = hipMemcpyDeviceToHost;
#endif
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor = static_cast<paddle::framework::LoDTensor*>(
          selected_rows->mutable_value());
      paddle::platform::GpuMemcpySync(
          pybind11::detail::array_proxy(array)->data,
          dense_tensor->data(),
          paddle::framework::DataTypeSize(dense_tensor->dtype()) *
              dense_tensor->numel(),
          kind);
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      paddle::platform::GpuMemcpySync(
          pybind11::detail::array_proxy(array)->data,
          dense_tensor->data(),
          paddle::framework::DataTypeSize(dense_tensor->dtype()) *
              dense_tensor->numel(),
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
      auto* dense_tensor = static_cast<paddle::framework::LoDTensor*>(
          selected_rows->mutable_value());
      paddle::memory::Copy(
          place,
          reinterpret_cast<void*>(pybind11::detail::array_proxy(array)->data),
          dense_tensor->place(),
          dense_tensor->data(),
          sizeof_dtype * numel);
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      paddle::memory::Copy(
          place,
          reinterpret_cast<void*>(pybind11::detail::array_proxy(array)->data),
          dense_tensor->place(),
          dense_tensor->data(),
          sizeof_dtype * numel);
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (self->tensor.is_custom_device()) {
    eager_gil_scoped_release guard;
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor = static_cast<paddle::framework::LoDTensor*>(
          selected_rows->mutable_value());
      phi::DeviceManager::GetDeviceWithPlace(self->tensor.place())
          ->MemoryCopyD2H(
              pybind11::detail::array_proxy(array)->data,
              dense_tensor->data(),
              paddle::framework::DataTypeSize(dense_tensor->dtype()) *
                  dense_tensor->numel());
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      phi::DeviceManager::GetDeviceWithPlace(self->tensor.place())
          ->MemoryCopyD2H(
              pybind11::detail::array_proxy(array)->data,
              dense_tensor->data(),
              paddle::framework::DataTypeSize(dense_tensor->dtype()) *
                  dense_tensor->numel());
    }
#endif
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Tensor.numpy() only support cpu tensor."));
    RETURN_PY_NONE
  }

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
    Py_intptr_t py_dims[paddle::framework::DDim::kMaxRank];
    Py_intptr_t py_strides[paddle::framework::DDim::kMaxRank];
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
    auto sp = std::make_unique<uint32_t[]>(max_unicode_length * numel);
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
    const paddle::experimental::Tensor& tensor, const platform::Place& place) {
  auto place_ = platform::is_gpu_place(place) ? place : tensor.place();

  auto tracer = egr::Controller::Instance().GetCurrentTracer();
  auto gc = tracer->MutableGarbageCollectorIfNotExists(place_);

  // Note(dev): This is an empty callback, the only way is to "reference"
  // inner memory Holder, so it will not be destructed until the kernels
  // launched at current stream of given place is finished, such as
  // CUDAPinned Mem -> CUDA by cudamemcpyAsync.
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
  paddle::experimental::Tensor cp_tensor;
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

static PyObject* tensor_method_cpu(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor cp_tensor;
  {
    eager_gil_scoped_release guard;
    cp_tensor = self->tensor.copy_to(phi::CPUPlace(), true);
    egr::EagerUtils::autograd_meta(&cp_tensor)->SetStopGradient(true);
    egr::EagerUtils::autograd_meta(&cp_tensor)
        ->SetPersistable(
            egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  }
  return ToPyObject(cp_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_reconstruct_from_(TensorObject* self,
                                                 PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor src_tensor =
      CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
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
  paddle::experimental::Tensor src_tensor =
      CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
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

static PyObject* tensor_method_clone(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor out;
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

  paddle::experimental::Tensor* grad;
  bool is_leaf = egr::egr_utils_api::IsLeafTensor(self->tensor);
  if (is_leaf) {
    grad = egr::EagerUtils::mutable_grad(self->tensor);
    PADDLE_ENFORCE(grad != nullptr,
                   paddle::platform::errors::Fatal(
                       "Detected NULL grad"
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

  if (egr::egr_utils_api::IsLeafTensor(self->tensor)) {
    eager_gil_scoped_release guard;
    // Add RetainGrad as PostHook to AccumulationNode
    paddle::experimental::Tensor* grad =
        egr::EagerUtils::mutable_grad(self->tensor);
    PADDLE_ENFORCE(grad != nullptr,
                   paddle::platform::errors::Fatal(
                       "Detected NULL grad"
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
  paddle::experimental::Tensor* dst_ptr =
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
  paddle::experimental::Tensor* dst_ptr =
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
  paddle::experimental::Tensor* src_ptr =
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
  paddle::experimental::Tensor src_tensor =
      CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
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

static PyObject* tensor_method_detach(TensorObject* self,
                                      PyObject* args,
                                      PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE_EQ(
      self->tensor.initialized(),
      true,
      platform::errors::InvalidArgument("Tensor %s has not been initialized!",
                                        self->tensor.name()));

  PyObject* obj = p_tensor_type->tp_alloc(p_tensor_type, 0);
  if (obj) {
    auto v = reinterpret_cast<TensorObject*>(obj);
    new (&(v->tensor)) paddle::experimental::Tensor();
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
    auto* tensor =
        static_cast<paddle::framework::LoDTensor*>(self->tensor.impl().get());
    VLOG(6) << "tensor: " << tensor->IsInitialized();
    return ToPyObject(tensor);
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

  auto* dense_tensor = static_cast<paddle::framework::LoDTensor*>(
      selected_rows->mutable_value());
  VLOG(1) << "dense_tensor: " << dense_tensor->IsInitialized();

  auto t = paddle::experimental::Tensor(
      egr::Controller::Instance().GenerateUniqueName());
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
      decrease_axis, none_axes, infer_flags, list_select_idxs;
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

  auto out = slice_axes.empty() && !list_select_flag
                 ? self->tensor
                 : paddle::experimental::Tensor(
                       egr::Controller::Instance().GenerateUniqueName());

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
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Slice is only support slice and strided_slice, but we got %s which "
          "is impossible, please check your code first or contact us by "
          "issue. ",
          op_type));
    }
  }

  if (!none_axes.empty()) {
    // Deal with cases when all axes are decreased.
    // After slice, the shape of out is [1], which should have been
    // [], but Paddle doesn't support scalar.
    // In order to ensure the correctness of the final shape of out,
    // one dimension of out needs to be decreased.
    // For example:
    // # x.shape: (2,3,4)
    // out = x[0, 1, 1, None] # out.shape : (1)
    if (static_cast<int>(decrease_axis.size()) == tensor->dims().size()) {
      none_axes.pop_back();
    }
    if (!none_axes.empty()) {
      paddle::experimental::Tensor new_out;
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
  }

  // the index is a list
  if (list_select_flag) {
    eager_gil_scoped_release guard;
    auto select_index = paddle::experimental::Tensor(
        egr::Controller::Instance().GenerateUniqueName());
    auto idx_tensor = std::make_shared<phi::DenseTensor>();
    select_index.set_impl(idx_tensor);
    auto* dev_ctx = platform::DeviceContextPool::Instance().Get(
        egr::Controller::Instance().GetExpectedPlace());
    paddle::framework::TensorFromVector(
        list_select_idxs, *dev_ctx, idx_tensor.get());
    framework::AttributeMap attrs = {{"dim", 0}};
    out = index_select_ad_func(self->tensor, select_index, 0);
  }

  return ToPyObject(out);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__getitem_from_offset(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  auto ptr = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
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
  std::vector<size_t> strides(tensor_dims.size());

  size_t numel = 1;
  for (int i = tensor_dims.size() - 1; i >= 0; --i) {
    strides[i] = numel;
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
      offset += index * strides[i];
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
    Py_intptr_t py_dims[paddle::framework::DDim::kMaxRank];                  \
    Py_intptr_t py_strides[paddle::framework::DDim::kMaxRank];               \
    py_dims[0] = 1;                                                          \
    py_strides[0] = 1;                                                       \
    auto& api = pybind11::detail::npy_api::get();                            \
    PyObject* array = api.PyArray_NewFromDescr_(                             \
        api.PyArray_Type_,                                                   \
        api.PyArray_DescrFromType_(numpy_dtype),                             \
        1,                                                                   \
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
        infer_flags, list_select_idxs;
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
          egr::egr_utils_api::IsLeafTensor(self->tensor) &&
              !egr::EagerUtils::autograd_meta(&self->tensor)->StopGradient(),
          false,
          platform::errors::InvalidArgument(
              "Leaf Tensor (%s) that doesn't stop gradient can't use "
              "inplace strategy.",
              self->tensor.name()));
    }

    paddle::experimental::Tensor value_tensor;

    if (PyCheckTensor(value_obj)) {
      value_tensor = reinterpret_cast<TensorObject*>(value_obj)->tensor;
    } else if (py::isinstance<py::array>(value_obj)) {
      paddle::experimental::Tensor value_tensor_tmp(
          std::make_shared<phi::DenseTensor>(),
          egr::Controller::Instance().GenerateUniqueName());
      py::object value_obj_tmp(py::handle(value_obj), true);
      py::object value = value_obj_tmp;
      if (self->tensor.dtype() == paddle::experimental::DataType::FLOAT32) {
        if (!py::isinstance<py::array_t<float>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<float>(value_obj_tmp);
        }
      } else if (self->tensor.dtype() ==
                 paddle::experimental::DataType::FLOAT64) {
        if (!py::isinstance<py::array_t<double>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<double>(value_obj_tmp);
        }
      } else if (self->tensor.dtype() ==
                 paddle::experimental::DataType::INT32) {
        if (!py::isinstance<py::array_t<int32_t>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<int32_t>(value_obj_tmp);
        }
      } else if (self->tensor.dtype() ==
                 paddle::experimental::DataType::INT64) {
        if (!py::isinstance<py::array_t<int64_t>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<int64_t>(value_obj_tmp);
        }
      } else if (self->tensor.dtype() == paddle::experimental::DataType::BOOL) {
        if (!py::isinstance<py::array_t<bool>>(value_obj_tmp)) {
          value = pybind11::detail::CastNumpyArray<bool>(value_obj_tmp);
        }
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "When assign a numpy.np value to a paddle.Tensor, "
            "the data type of the paddle.Tensor must be bool, "
            "float32, int32 or int64, "
            "please check the type of tensor."));
      }

      if (!value_tensor_tmp.initialized()) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        SetTensorFromPyArray(
            static_cast<phi::DenseTensor*>(value_tensor_tmp.impl().get()),
            value,
            platform::Place(platform::CUDAPlace(0)),
            false);
#else
        SetTensorFromPyArray(
            static_cast<phi::DenseTensor*>(value_tensor_tmp.impl().get()),
            value,
            platform::Place(platform::CPUPlace()),
            false);
#endif
      } else {
        SetTensorFromPyArray(
            static_cast<phi::DenseTensor*>(value_tensor_tmp.impl().get()),
            value,
            value_tensor_tmp.place(),
            false);
      }

      value_tensor = value_tensor_tmp;
    } else {
      py::object value_obj_tmp(py::handle(value_obj), true);
      // convert the value to self data type
      if (py::isinstance<py::float_>(value_obj_tmp) ||
          py::isinstance<py::int_>(value_obj_tmp) ||
          py::isinstance<py::bool_>(value_obj_tmp)) {
        if (self->tensor.dtype() == paddle::experimental::DataType::FLOAT32) {
          attrs["fp32_values"] =
              std::vector<float>{value_obj_tmp.cast<float>()};
        } else if (self->tensor.dtype() ==
                   paddle::experimental::DataType::FLOAT64) {
          attrs["fp64_values"] =
              std::vector<double>{value_obj_tmp.cast<double>()};
        } else if (self->tensor.dtype() ==
                   paddle::experimental::DataType::INT32) {
          attrs["int32_values"] =
              std::vector<int32_t>{value_obj_tmp.cast<int32_t>()};
        } else if (self->tensor.dtype() ==
                   paddle::experimental::DataType::INT64) {
          attrs["int64_values"] =
              std::vector<int64_t>{value_obj_tmp.cast<int64_t>()};
        } else if (self->tensor.dtype() ==
                   paddle::experimental::DataType::BOOL) {
          attrs["bool_values"] = std::vector<int>{value_obj_tmp.cast<bool>()};
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "When assign a value to a paddle.Tensor, "
              "the data type of the paddle.Tensor must be bool, "
              "float32, int32 or int64, "
              "please check the type of tensor."));
        }
        attrs["shape"] = std::vector<int64_t>{1};

      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Value type error. The assign value allows "
            "numpy.ndarray, integer, float or bool, "
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
        paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                             egr::kSlotSmallVectorSize>
            tmps = {{self->tensor}, {value_tensor}};
        auto amp_dtype = egr::GetAmpDestDtype("set_value", tmps);
        self->tensor = egr::EagerAmpAutoCast(
            self->tensor.name(), self->tensor, amp_dtype, "set_value");
        value_tensor = egr::EagerAmpAutoCast(
            value_tensor.name(), value_tensor, amp_dtype, "set_value");
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
    auto self_numpy = TensorToPyArray(*self_tensor);
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
  if (egr::egr_utils_api::IsLeafTensor(self->tensor)) {
    VLOG(6) << "Register hook for leaf tensor: " << self->tensor.name();

    auto autograd_meta = egr::EagerUtils::unsafe_autograd_meta(self->tensor);

    if (autograd_meta && !autograd_meta->StopGradient()) {
      if (!autograd_meta->GetMutableGradNode()) {
        VLOG(6) << "Detected NULL grad_node, Leaf tensor should have had "
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

static PyObject* tensor_register_reduce_hook(TensorObject* self,
                                             PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Register reduce hook for tensor: " << self->tensor.name();

  std::shared_ptr<egr::GradNodeBase> grad_node =
      egr::EagerUtils::grad_node(self->tensor);
  PADDLE_ENFORCE_EQ(egr::egr_utils_api::IsLeafTensor(self->tensor),
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
      paddle::platform::errors::Fatal("Detected NULL grad_node,"
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

static PyObject* tensor_method_set_vocab(TensorObject* self,
                                         PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  using Vocab = std::unordered_map<std::wstring, int>;
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
  using Strings = std::vector<std::string>;
  auto strings = CastPyArg2Strings(PyTuple_GET_ITEM(args, 0), 0);
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
  using Vocab = std::unordered_map<std::wstring, int>;
  auto* var_tensor =
      static_cast<const egr::VariableCompatTensor*>(self->tensor.impl().get());
  return ToPyObject(var_tensor->Get<Vocab>());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

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

static PyObject* tensor_method_get_non_zero_indices(TensorObject* self,
                                                    PyObject* args,
                                                    PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_coo_tensor(),
                 paddle::platform::errors::Fatal(
                     "this method is only effective for SparseCooTensor"));
  auto sparse_coo_tensor =
      std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
  paddle::experimental::Tensor tensor(std::make_shared<phi::DenseTensor>(
      sparse_coo_tensor->non_zero_indices()));
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

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
    paddle::experimental::Tensor tensor(std::make_shared<phi::DenseTensor>(
        sparse_coo_tensor->non_zero_elements()));
    return ToPyObject(tensor);
  } else {
    auto sparse_csr_tensor =
        std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
    paddle::experimental::Tensor tensor(std::make_shared<phi::DenseTensor>(
        sparse_csr_tensor->non_zero_elements()));
    return ToPyObject(tensor);
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_get_non_zero_crows(TensorObject* self,
                                                  PyObject* args,
                                                  PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_csr_tensor(),
                 paddle::platform::errors::Fatal(
                     "this method is only effective for SparseCsrTensor"));
  auto sparse_csr_tensor =
      std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
  paddle::experimental::Tensor tensor(
      std::make_shared<phi::DenseTensor>(sparse_csr_tensor->non_zero_crows()));
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_get_non_zero_cols(TensorObject* self,
                                                 PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE(self->tensor.is_sparse_csr_tensor(),
                 paddle::platform::errors::Fatal(
                     "this method is only effective for SparseCsrTensor"));
  auto sparse_csr_tensor =
      std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
  paddle::experimental::Tensor tensor(
      std::make_shared<phi::DenseTensor>(sparse_csr_tensor->non_zero_cols()));
  return ToPyObject(tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

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

static PyObject* tensor_method_element_size(TensorObject* self,
                                            PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  uint32_t element_size = framework::DataTypeSize(self->tensor.dtype());

  return ToPyObject(element_size);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

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

static PyObject* tensor_methon_element_size(TensorObject* self,
                                            PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  return ToPyObject(paddle::experimental::SizeOf(self->tensor.dtype()));
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

  paddle::experimental::Tensor* grad =
      egr::EagerUtils::mutable_grad(self->tensor);
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
  paddle::experimental::Tensor* grad =
      egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE_EQ(grad != nullptr,
                    true,
                    platform::errors::InvalidArgument(
                        "Detected NULL grad. Please check if you have manually "
                        "cleared the grad inside autograd_meta"));
  return ToPyObject(grad->name());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__grad_value(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor* grad =
      egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE_EQ(grad != nullptr,
                    true,
                    platform::errors::InvalidArgument(
                        "Detected NULL grad. Please check if you have manually "
                        "cleared the grad inside autograd_meta"));

  if (!grad->defined()) {
    RETURN_PY_NONE
  }
  if (grad->is_dense_tensor()) {
    auto* grad_tensor =
        static_cast<paddle::framework::LoDTensor*>(grad->impl().get());
    return ToPyObject(grad_tensor);
  } else {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "this method is only supported for DenseTensor"));
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__unset_fake_empty(TensorObject* self,
                                          PyObject* args,
                                          PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor* grad =
      egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE_EQ(grad != nullptr,
                    true,
                    platform::errors::InvalidArgument(
                        "Detected NULL grad. Please check if you have manually "
                        "cleared the grad inside autograd_meta"));

  bool is_leaf = egr::egr_utils_api::IsLeafTensor(self->tensor);
  if (is_leaf) {
    std::static_pointer_cast<egr::GradNodeAccumulation>(
        egr::EagerUtils::grad_node(self->tensor))
        ->SetFakeEmpty(false);
  }
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
  auto* self_tensor =
      static_cast<paddle::framework::LoDTensor*>(self->tensor.impl().get());
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

PyMethodDef variable_methods[] = {
    {"numpy",
     (PyCFunction)(void (*)(void))tensor_method_numpy,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_is_initialized",
     (PyCFunction)(void (*)(void))tensor_method__is_initialized,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_is_dense_tensor_hold_allocation",
     (PyCFunction)(void (*)(
         void))tensor_method__is_dense_tensor_hold_allocation,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_copy_to",
     (PyCFunction)(void (*)(void))tensor_method__copy_to,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"copy_",
     (PyCFunction)(void (*)(void))tensor_method_copy_,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"clone",
     (PyCFunction)(void (*)(void))tensor_method_clone,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"reconstruct_from_",
     (PyCFunction)(void (*)(void))tensor_method_reconstruct_from_,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"retain_grads",
     (PyCFunction)(void (*)(void))tensor_retain_grads,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"clear_gradient",
     (PyCFunction)(void (*)(void))tensor_clear_gradient,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"is_dense",
     (PyCFunction)(void (*)(void))tensor_method_is_dense,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_zero_grads",
     (PyCFunction)(void (*)(void))tensor__zero_grads,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_share_buffer_to",
     (PyCFunction)(void (*)(void))tensor__share_buffer_to,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_is_shared_buffer_with",
     (PyCFunction)(void (*)(void))tensor__is_shared_buffer_with,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_share_underline_tensor_to",
     (PyCFunction)(void (*)(void))tensor__share_underline_tensor_to,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_is_shared_underline_tensor_with",
     (PyCFunction)(void (*)(void))tensor__is_shared_underline_tensor_with,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"detach",
     (PyCFunction)(void (*)(void))tensor_method_detach,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"get_tensor",
     (PyCFunction)(void (*)(void))tensor_method_get_underline_tensor,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"get_selected_rows",
     (PyCFunction)(void (*)(void))tensor_method_get_underline_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_get_tensor_from_selected_rows",
     (PyCFunction)(void (*)(void))tensor_method__get_tensor_from_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_getitem_index_not_tensor",
     (PyCFunction)(void (*)(void))tensor__getitem_index_not_tensor,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_getitem_from_offset",
     (PyCFunction)(void (*)(void))tensor__getitem_from_offset,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"__setitem_eager_tensor__",
     (PyCFunction)(void (*)(void))tensor_method__setitem_eager_tensor,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_register_grad_hook",
     (PyCFunction)(void (*)(void))tensor_register_grad_hook,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_remove_grad_hook",
     (PyCFunction)(void (*)(void))tensor_remove_grad_hook,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_register_backward_hook",
     (PyCFunction)(void (*)(void))tensor_register_reduce_hook,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_set_grad_type",
     (PyCFunction)(void (*)(void))tensor__set_grad_type,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_clear",
     (PyCFunction)(void (*)(void))tensor__clear,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_copy_gradient_from",
     (PyCFunction)(void (*)(void))tensor__copy_gradient_from,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    /** the methods to adapt old dygraph, will be removed in the future **/
    {"set_string_list",
     (PyCFunction)(void (*)(void))tensor_method_set_string_list,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"set_vocab",
     (PyCFunction)(void (*)(void))tensor_method_set_vocab,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"get_map_tensor",
     (PyCFunction)(void (*)(void))tensor_method_get_map_tensor,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    /***the method of sparse tensor****/
    {"nnz",
     (PyCFunction)(void (*)(void))tensor_method_get_non_zero_nums,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"indices",
     (PyCFunction)(void (*)(void))tensor_method_get_non_zero_indices,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"values",
     (PyCFunction)(void (*)(void))tensor_method_get_non_zero_elements,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"crows",
     (PyCFunction)(void (*)(void))tensor_method_get_non_zero_crows,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"cols",
     (PyCFunction)(void (*)(void))tensor_method_get_non_zero_cols,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"is_sparse",
     (PyCFunction)(void (*)(void))tensor_method_is_sparse,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"is_sparse_coo",
     (PyCFunction)(void (*)(void))tensor_method_is_sparse_coo,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"is_sparse_csr",
     (PyCFunction)(void (*)(void))tensor_method_is_sparse_csr,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"is_same_shape",
     (PyCFunction)(void (*)(void))tensor_method_is_same_shape,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"to_sparse_csr",
     (PyCFunction)(void (*)(void))tensor_method_to_sparse_csr,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"element_size",
     (PyCFunction)(void (*)(void))tensor_method_element_size,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    /***the method of sparse tensor****/
    {"_inplace_version",
     (PyCFunction)(void (*)(void))tensor__inplace_version,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_bump_inplace_version",
     (PyCFunction)(void (*)(void))tensor__bump_inplace_version,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"is_selected_rows",
     (PyCFunction)(void (*)(void))tensor_method_is_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"rows",
     (PyCFunction)(void (*)(void))tensor_method_get_rows,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"element_size",
     (PyCFunction)(void (*)(void))tensor_methon_element_size,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_reset_grad_inplace_version",
     (PyCFunction)(void (*)(void))tensor__reset_grad_inplace_version,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_share_memory",
     (PyCFunction)(void (*)(void))tensor_method__share_memory,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_offset",
     (PyCFunction)(void (*)(void))tensor__offset,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_grad_name",
     (PyCFunction)(void (*)(void))tensor__grad_name,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_grad_value",
     (PyCFunction)(void (*)(void))tensor__grad_value,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_unset_fake_empty",
     (PyCFunction)(void (*)(void))tensor__unset_fake_empty,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
#if defined(PADDLE_WITH_CUDA)
    {"_tensor_uva",
     (PyCFunction)(void (*)(void))tensor_method__uva,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
#endif
    {NULL, NULL, 0, NULL}};

// variable_methods for core.eager.StringTensor
PyMethodDef string_tensor_variable_methods[] = {
    {"numpy",
     (PyCFunction)(void (*)(void))tensor_method_numpy_for_string_tensor,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_is_initialized",
     (PyCFunction)(void (*)(void))tensor_method__is_initialized,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"_is_string_tensor_hold_allocation",
     (PyCFunction)(void (*)(
         void))tensor_method__is_string_tensor_hold_allocation,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    // TODO(zhoushunjie): Need to add _copy_to, copy_ for StringTensor.
    {NULL, NULL, 0, NULL}};

}  // namespace pybind
}  // namespace paddle
