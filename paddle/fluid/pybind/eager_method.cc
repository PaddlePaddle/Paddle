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
#include <Python.h>

#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
namespace paddle {
namespace pybind {

extern void InitTensorWithNumpyValue(TensorObject* self,
                                     const pybind11::object& array,
                                     bool zero_copy);

extern PyTypeObject* p_tensor_type;

static PyObject* tensor_method_numpy(TensorObject* self, PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE_EQ(
      self->tensor.initialized(), true,
      platform::errors::InvalidArgument(
          "Tensor data of %s is Empty that indicates we have null tensor for "
          "now, please check if it has no data and initialize it first.",
          self->tensor.name()));
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
  auto& api = pybind11::detail::npy_api::get();
  PyObject* array = api.PyArray_NewFromDescr_(
      api.PyArray_Type_, api.PyArray_DescrFromType_(numpy_dtype),
      tensor_dims.size(), py_dims, py_strides, nullptr,
      pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
          pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
      nullptr);

  if (self->tensor.is_cpu()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
    platform::CPUPlace place;
    // deep copy
    paddle::memory::Copy(place, reinterpret_cast<void*>(
                                    pybind11::detail::array_proxy(array)->data),
                         place, dense_tensor->data(), sizeof_dtype * numel);
#if defined(PADDLE_WITH_CUDA)
  } else if (self->tensor.is_cuda()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());

    paddle::platform::GpuMemcpySync(
        pybind11::detail::array_proxy(array)->data, dense_tensor->data(),
        paddle::framework::DataTypeSize(dense_tensor->dtype()) *
            dense_tensor->numel(),
        cudaMemcpyDeviceToHost);
#endif
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Tensor.numpy() only support cpu tensor."));
    Py_INCREF(Py_None);
    return Py_None;
  }

  return array;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__is_initialized(TensorObject* self,
                                               PyObject* args,
                                               PyObject* kwargs) {
  EAGER_TRY
  return ToPyObject(self->tensor.initialized());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method__copy_to(TensorObject* self, PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 0), 0);
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
  auto cp_tensor =
      self->tensor.copy_to(phi::TransToPtenBackend(place), blocking);
  egr::EagerUtils::autograd_meta(&cp_tensor)->SetStopGradient(true);
  egr::EagerUtils::autograd_meta(&cp_tensor)
      ->SetPersistable(
          egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
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
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_copy_(TensorObject* self, PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor src_tensor =
      CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 1), 1);
  VLOG(6) << "Start Copy Tensor " << src_tensor.name() << " to "
          << self->tensor.name();
  if (!self->tensor.defined()) {
    egr::EagerUtils::autograd_meta(&(self->tensor))
        ->SetStopGradient(
            egr::EagerUtils::autograd_meta(&(src_tensor))->StopGradient());
    egr::EagerUtils::autograd_meta(&(self->tensor))
        ->SetPersistable(
            egr::EagerUtils::autograd_meta(&(src_tensor))->Persistable());
  }

  self->tensor.copy_(src_tensor, blocking);

  VLOG(6) << "Finish Copy Tensor " << src_tensor.name() << " to "
          << self->tensor.name();
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_retain_grads(TensorObject* self, PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  if (egr::Controller::Instance().HasGrad()) {
    auto meta = egr::EagerUtils::autograd_meta(&(self->tensor));
    if (!meta->GetMutableGradNode()) {
      VLOG(6) << "Make grad node of tensor: " << self->tensor.name()
              << "become accumulation node";
      meta->SetGradNode(std::make_shared<egr::GradNodeAccumulation>(meta));
    }
    egr::egr_utils_api::RetainGradForTensor(self->tensor);
  }
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_clear_gradient(TensorObject* self, PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "ClearGradient " << self->tensor.name();

  Py_ssize_t args_num = PyTuple_Size(args);
  bool set_to_zero = true;
  if (args_num == (Py_ssize_t)1) {
    CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 0), 0);
  }

  paddle::experimental::Tensor* grad;
  if (egr::egr_utils_api::IsLeafTensor(self->tensor)) {
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
        grad->set_impl(paddle::experimental::zeros_like(*grad).impl());
      } else {
        VLOG(4) << "Gradient of " << self->tensor.name()
                << " is initialized, will be released.";
        auto dense_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(grad->impl());
        dense_tensor->MoveMemoryHolder();
      }
    }
  }

  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__zero_grads(TensorObject* self, PyObject* args,
                                    PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "ZeroGrads " << self->tensor.name();

  if (egr::egr_utils_api::IsLeafTensor(self->tensor)) {
    // Add RetainGrad as PostHook to AccumulationNode
    paddle::experimental::Tensor* grad =
        egr::EagerUtils::mutable_grad(self->tensor);
    PADDLE_ENFORCE(grad != nullptr,
                   paddle::platform::errors::Fatal(
                       "Detected NULL grad"
                       "Please check if you have manually cleared"
                       "the grad inside autograd_meta"));
    if (grad->initialized()) {
      grad->set_impl(paddle::experimental::zeros_like(*(grad)).impl());
    }
  } else {
    auto meta = egr::EagerUtils::unsafe_autograd_meta(self->tensor);
    if (meta->MutableGrad()->initialized()) {
      meta->MutableGrad()->set_impl(
          paddle::experimental::zeros_like(*(meta->MutableGrad())).impl());
    }
  }

  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__share_buffer_to(TensorObject* self, PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor* dst_ptr =
      &(reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor);
  PADDLE_ENFORCE_EQ(self->tensor.initialized(), true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        self->tensor.name()));
  auto* src_tensor =
      static_cast<paddle::framework::Tensor*>(self->tensor.impl().get());
  auto dst_tensor =
      static_cast<paddle::framework::Tensor*>(dst_ptr->impl().get());
  dst_tensor->ShareDataWith(*src_tensor);
  dst_tensor->ShareDataTypeWith(*src_tensor);
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__is_shared_buffer_with(TensorObject* self,
                                               PyObject* args,
                                               PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor* dst_ptr =
      &(reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor);
  PADDLE_ENFORCE_EQ(self->tensor.initialized(), true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        self->tensor.name()));
  bool res = false;
  if (!self->tensor.defined() || !dst_ptr->defined()) {
    return ToPyObject(res);
  }
  auto* self_ptr =
      static_cast<paddle::framework::Tensor*>(self->tensor.impl().get());
  auto dst_tensor =
      static_cast<paddle::framework::Tensor*>(dst_ptr->impl().get());
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
  PADDLE_ENFORCE_EQ(self->tensor.initialized(), true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        self->tensor.name()));
  src_ptr->set_impl(self->tensor.impl());
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__is_shared_underline_tensor_with(TensorObject* self,
                                                         PyObject* args,
                                                         PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor src_tensor =
      CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  PADDLE_ENFORCE_EQ(src_tensor.initialized(), true,
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

static PyObject* tensor_method_detach(TensorObject* self, PyObject* args,
                                      PyObject* kwargs) {
  EAGER_TRY
  PADDLE_ENFORCE_EQ(
      self->tensor.initialized(), true,
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
  if (self->tensor.is_dense_tensor()) {
    auto* tensor =
        static_cast<paddle::framework::LoDTensor*>(self->tensor.impl().get());
    VLOG(6) << "tensor: " << tensor->IsInitialized();
    return ToPyObject(tensor);
  } else {
    Py_IncRef(Py_None);
    return Py_None;
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

// NOTE(wuweilong): Set value and not change self's original place
static PyObject* tensor_method_set_value(TensorObject* self, PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Value " << self->tensor.name();
  pybind11::object numpy_value =
      pybind11::object(pybind11::handle(PyTuple_GET_ITEM(args, 0)), true);
  InitTensorWithNumpyValue(self, numpy_value, false);
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef variable_methods[] = {
    {"numpy", (PyCFunction)(void (*)(void))tensor_method_numpy,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_is_initialized",
     (PyCFunction)(void (*)(void))tensor_method__is_initialized,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_copy_to", (PyCFunction)(void (*)(void))tensor_method__copy_to,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"copy_", (PyCFunction)(void (*)(void))tensor_method_copy_,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"reconstruct_from_",
     (PyCFunction)(void (*)(void))tensor_method_reconstruct_from_,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"retain_grads", (PyCFunction)(void (*)(void))tensor_retain_grads,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"clear_gradient", (PyCFunction)(void (*)(void))tensor_clear_gradient,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_zero_grads", (PyCFunction)(void (*)(void))tensor__zero_grads,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_share_buffer_to", (PyCFunction)(void (*)(void))tensor__share_buffer_to,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_is_shared_buffer_with",
     (PyCFunction)(void (*)(void))tensor__is_shared_buffer_with,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_share_underline_tensor_to",
     (PyCFunction)(void (*)(void))tensor__share_underline_tensor_to,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_is_shared_underline_tensor_with",
     (PyCFunction)(void (*)(void))tensor__is_shared_underline_tensor_with,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"detach", (PyCFunction)(void (*)(void))tensor_method_detach,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_tensor",
     (PyCFunction)(void (*)(void))tensor_method_get_underline_tensor,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_set_value", (PyCFunction)(void (*)(void))tensor_method_set_value,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

}  // namespace pybind
}  // namespace paddle
