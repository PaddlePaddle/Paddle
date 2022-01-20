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
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/pten/api/include/api.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
namespace paddle {
namespace pybind {

extern PyTypeObject* p_eager_tensor_type;

static PyObject* eager_tensor_method_numpy(EagerTensorObject* self,
                                           PyObject* args, PyObject* kwargs) {
  EAGER_SYNC_TRY
  PADDLE_ENFORCE_EQ(
      self->eager_tensor.initialized(), true,
      platform::errors::InvalidArgument(
          "Tensor data of %s is Empty that indicates we have null tensor for "
          "now, please check if it has no data and initialize it first.",
          self->eager_tensor.name()));
  auto tensor_dims = self->eager_tensor.shape();
  auto numpy_dtype = TensorDtype2NumpyDtype(self->eager_tensor.type());
  auto sizeof_dtype = pten::DataTypeSize(self->eager_tensor.type());
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

  if (self->eager_tensor.is_cpu()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<pten::DenseTensor>(self->eager_tensor.impl());
    platform::CPUPlace place;
    // deep copy
    paddle::memory::Copy(place, reinterpret_cast<void*>(
                                    pybind11::detail::array_proxy(array)->data),
                         place, dense_tensor->data(), sizeof_dtype * numel);
#if defined(PADDLE_WITH_CUDA)
  } else if (self->eager_tensor.is_cuda()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<pten::DenseTensor>(self->eager_tensor.impl());

    paddle::platform::GpuMemcpySync(
        pybind11::detail::array_proxy(array)->data, dense_tensor->data(),
        pten::DataTypeSize(dense_tensor->dtype()) * dense_tensor->numel(),
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

static PyObject* eager_tensor_method__is_initialized(EagerTensorObject* self,
                                                     PyObject* args,
                                                     PyObject* kwargs) {
  EAGER_SYNC_TRY
  return ToPyObject(self->eager_tensor.initialized());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor_method__copy_to(EagerTensorObject* self,
                                              PyObject* args,
                                              PyObject* kwargs) {
  EAGER_SYNC_TRY
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 0), 0);
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
  auto cp_tensor =
      self->eager_tensor.copy_to(pten::TransToPtenBackend(place), blocking);
  egr::EagerUtils::autograd_meta(&cp_tensor)->SetStopGradient(true);
  egr::EagerUtils::autograd_meta(&cp_tensor)
      ->SetPersistable(
          egr::EagerUtils::autograd_meta(&(self->eager_tensor))->Persistable());
  return ToPyObject(cp_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor_method_copy_(EagerTensorObject* self,
                                           PyObject* args, PyObject* kwargs) {
  EAGER_SYNC_TRY
  egr::EagerTensor src_tensor =
      CastPyArg2EagerTensor(PyTuple_GET_ITEM(args, 0), 0);
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 1), 1);
  VLOG(6) << "Start Copy Tensor " << src_tensor.name() << " to "
          << self->eager_tensor.name();
  if (!self->eager_tensor.defined()) {
    egr::EagerUtils::autograd_meta(&(self->eager_tensor))
        ->SetStopGradient(
            egr::EagerUtils::autograd_meta(&(src_tensor))->StopGradient());
    egr::EagerUtils::autograd_meta(&(self->eager_tensor))
        ->SetPersistable(
            egr::EagerUtils::autograd_meta(&(src_tensor))->Persistable());
  }

  self->eager_tensor.copy_(src_tensor, blocking);

  VLOG(6) << "Finish Copy Tensor " << src_tensor.name() << " to "
          << self->eager_tensor.name();
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor_retain_grads(EagerTensorObject* self,
                                           PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  if (egr::Controller::Instance().HasGrad()) {
    auto meta = egr::EagerUtils::autograd_meta(&(self->eager_tensor));
    if (!meta->GetMutableGradNode()) {
      VLOG(6) << "Make grad node of tensor: " << self->eager_tensor.name()
              << "become accumulation node";
      meta->SetGradNode(std::make_shared<egr::GradNodeAccumulation>());
    }
    egr::egr_utils_api::RetainGradForTensor(self->eager_tensor);
  }
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor__clear_gradient(EagerTensorObject* self,
                                              PyObject* args,
                                              PyObject* kwargs) {
  EAGER_SYNC_TRY
  VLOG(4) << "ClearGradient " << self->eager_tensor.name();

  egr::EagerTensor* grad;
  if (egr::egr_utils_api::IsLeafTensor(self->eager_tensor)) {
    // Add RetainGrad as PostHook to AccumulationNode
    std::shared_ptr<egr::GradNodeBase> grad_node =
        egr::EagerUtils::grad_node(self->eager_tensor);
    PADDLE_ENFORCE(
        grad_node.get() != nullptr,
        paddle::platform::errors::Fatal("Detected NULL grad_node"
                                        "Leaf tensor should have had grad_node "
                                        "with type: GradNodeAccumulation"));
    auto accumulation_grad_node =
        std::dynamic_pointer_cast<egr::GradNodeAccumulation>(grad_node);
    grad = accumulation_grad_node->Grad();
  } else {
    auto meta = egr::EagerUtils::unsafe_autograd_meta(self->eager_tensor);
    grad = meta->MutableGrad();
  }

  if (grad->initialized()) {
    VLOG(4) << "Gradient of " << self->eager_tensor.name()
            << " is initialized, will be released.";
    auto dense_tensor =
        std::dynamic_pointer_cast<pten::DenseTensor>(grad->impl());
    dense_tensor->MoveMemoryHolder();
  }
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor__zero_grads(EagerTensorObject* self,
                                          PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "ZeroGrads " << self->eager_tensor.name();

  if (egr::egr_utils_api::IsLeafTensor(self->eager_tensor)) {
    // Add RetainGrad as PostHook to AccumulationNode
    std::shared_ptr<egr::GradNodeBase> grad_node =
        egr::EagerUtils::grad_node(self->eager_tensor);
    PADDLE_ENFORCE(
        grad_node.get() != nullptr,
        paddle::platform::errors::Fatal("Detected NULL grad_node"
                                        "Leaf tensor should have had grad_node "
                                        "with type: GradNodeAccumulation"));
    auto accumulation_grad_node =
        std::dynamic_pointer_cast<egr::GradNodeAccumulation>(grad_node);
    if (accumulation_grad_node->Grad()->initialized()) {
      accumulation_grad_node->Grad()->set_tensor(
          std::make_shared<paddle::experimental::Tensor>(
              paddle::experimental::zeros_like(
                  *(accumulation_grad_node->Grad()->Tensor().get()))));
    }
  } else {
    auto meta = egr::EagerUtils::unsafe_autograd_meta(self->eager_tensor);
    if (meta->MutableGrad()->initialized()) {
      meta->MutableGrad()->set_tensor(
          std::make_shared<paddle::experimental::Tensor>(
              paddle::experimental::zeros_like(
                  *(meta->MutableGrad()->Tensor().get()))));
    }
  }

  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor__share_buffer_to(EagerTensorObject* self,
                                               PyObject* args,
                                               PyObject* kwargs) {
  EAGER_SYNC_TRY
  egr::EagerTensor* dst_ptr =
      &(reinterpret_cast<EagerTensorObject*>(PyTuple_GET_ITEM(args, 0))
            ->eager_tensor);
  PADDLE_ENFORCE_EQ(self->eager_tensor.initialized(), true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        self->eager_tensor.name()));
  auto* src_tensor =
      static_cast<paddle::framework::Tensor*>(self->eager_tensor.impl().get());
  auto dst_tensor =
      static_cast<paddle::framework::Tensor*>(dst_ptr->impl().get());
  dst_tensor->ShareDataWith(*src_tensor);
  dst_tensor->ShareDataTypeWith(*src_tensor);
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor__is_shared_buffer_with(EagerTensorObject* self,
                                                     PyObject* args,
                                                     PyObject* kwargs) {
  EAGER_SYNC_TRY
  egr::EagerTensor* dst_ptr =
      &(reinterpret_cast<EagerTensorObject*>(PyTuple_GET_ITEM(args, 0))
            ->eager_tensor);
  PADDLE_ENFORCE_EQ(self->eager_tensor.initialized(), true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        self->eager_tensor.name()));
  bool res = false;
  if (!self->eager_tensor.defined() || !dst_ptr->defined()) {
    return ToPyObject(res);
  }
  auto* self_ptr =
      static_cast<paddle::framework::Tensor*>(self->eager_tensor.impl().get());
  auto dst_tensor =
      static_cast<paddle::framework::Tensor*>(dst_ptr->impl().get());
  res = dst_tensor->IsSharedBufferWith(*self_ptr);
  return ToPyObject(res);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor__share_underline_tensor_to(
    EagerTensorObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_SYNC_TRY
  egr::EagerTensor* src_ptr =
      &(reinterpret_cast<EagerTensorObject*>(PyTuple_GET_ITEM(args, 0))
            ->eager_tensor);
  PADDLE_ENFORCE_EQ(self->eager_tensor.initialized(), true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        self->eager_tensor.name()));
  src_ptr->set_impl(self->eager_tensor.impl());
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor__is_shared_underline_tensor_with(
    EagerTensorObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_SYNC_TRY
  egr::EagerTensor src_tensor =
      CastPyArg2EagerTensor(PyTuple_GET_ITEM(args, 0), 0);
  PADDLE_ENFORCE_EQ(src_tensor.initialized(), true,
                    platform::errors::InvalidArgument(
                        "Tensor %s has not been initialized! please initialize "
                        "src tensor before share_buffer_with to other.",
                        src_tensor.name()));
  bool res = false;
  if (!self->eager_tensor.defined() || !src_tensor.defined()) {
    return ToPyObject(res);
  }
  res = (self->eager_tensor.impl().get() == src_tensor.impl().get());
  return ToPyObject(res);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor_method_detach(EagerTensorObject* self,
                                            PyObject* args, PyObject* kwargs) {
  EAGER_SYNC_TRY
  PADDLE_ENFORCE_EQ(
      self->eager_tensor.initialized(), true,
      platform::errors::InvalidArgument("Tensor %s has not been initialized!",
                                        self->eager_tensor.name()));

  PyObject* obj = p_eager_tensor_type->tp_alloc(p_eager_tensor_type, 0);
  if (obj) {
    auto v = reinterpret_cast<EagerTensorObject*>(obj);
    new (&(v->eager_tensor)) egr::EagerTensor();
    v->eager_tensor.set_impl(self->eager_tensor.impl());
    v->eager_tensor.set_name(egr::Controller::Instance().GenerateUniqueName());
    auto autograd_meta_src =
        egr::EagerUtils::autograd_meta(&(self->eager_tensor));
    auto autograd_meta = egr::EagerUtils::autograd_meta(&(v->eager_tensor));
    autograd_meta->SetPersistable(autograd_meta_src->Persistable());
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "tp_alloc return null, can not new a PyObject."));
  }

  return obj;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_tensor_method_get_underline_tensor(
    EagerTensorObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_SYNC_TRY
  if (self->eager_tensor.is_dense_tensor()) {
    auto* tensor = static_cast<paddle::framework::LoDTensor*>(
        self->eager_tensor.impl().get());
    VLOG(6) << "tensor: " << tensor->IsInitialized();
    return ToPyObject(tensor);
  } else {
    Py_IncRef(Py_None);
    return Py_None;
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef variable_methods[] = {
    {"numpy", (PyCFunction)(void (*)(void))eager_tensor_method_numpy,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_is_initialized",
     (PyCFunction)(void (*)(void))eager_tensor_method__is_initialized,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_copy_to", (PyCFunction)(void (*)(void))eager_tensor_method__copy_to,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"copy_", (PyCFunction)(void (*)(void))eager_tensor_method_copy_,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"retain_grads", (PyCFunction)(void (*)(void))eager_tensor_retain_grads,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_clear_gradient",
     (PyCFunction)(void (*)(void))eager_tensor__clear_gradient,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_zero_grads", (PyCFunction)(void (*)(void))eager_tensor__zero_grads,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_share_buffer_to",
     (PyCFunction)(void (*)(void))eager_tensor__share_buffer_to,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_is_shared_buffer_with",
     (PyCFunction)(void (*)(void))eager_tensor__is_shared_buffer_with,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_share_underline_tensor_to",
     (PyCFunction)(void (*)(void))eager_tensor__share_underline_tensor_to,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_is_shared_underline_tensor_with",
     (PyCFunction)(void (*)(void))eager_tensor__is_shared_underline_tensor_with,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"detach", (PyCFunction)(void (*)(void))eager_tensor_method_detach,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_tensor",
     (PyCFunction)(void (*)(void))eager_tensor_method_get_underline_tensor,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

}  // namespace pybind
}  // namespace paddle
