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
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "pybind11/detail/internals.h"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/pybind/tensor_py.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

class PyTensorHook : public egr::TensorHook {
 public:
  explicit PyTensorHook(PyObject* func) : py_func_(func) {
    Py_INCREF(py_func_);
  }

  ~PyTensorHook() {
    py::gil_scoped_acquire gil;
    Py_DECREF(py_func_);
  }

  paddle::experimental::Tensor operator()(
      const paddle::experimental::Tensor& var) override {
    py::gil_scoped_acquire gil;
    VLOG(3) << "Call PyTensorHook for var " << var.name();

    PyObject* res = nullptr;
    try {
      res = PyObject_CallFunctionObjArgs(py_func_, ToPyObject(var), nullptr);
    } catch (platform::EnforceNotMet& e) {
      throw std::move(e);
    } catch (std::exception& e) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Hook function of Tensor raises an exception: %s.", e.what()));
    } catch (...) {
      PADDLE_THROW(platform::errors::Fatal(
          "Hook function of Tensor raises an unknown exception."));
    }

    PADDLE_ENFORCE_NOT_NULL(res,
                            platform::errors::Unavailable(
                                "Hook function of Tensor return a nullptr."));
    if (res == Py_None) {
      return var;
    }
    return reinterpret_cast<TensorObject*>(res)->tensor;
  }

 private:
  PyObject* py_func_;
};

class PyTensorVoidHook : public egr::TensorVoidHook {
 public:
  explicit PyTensorVoidHook(PyObject* func) : py_func_(func) {
    Py_INCREF(py_func_);
  }

  ~PyTensorVoidHook() {
    py::gil_scoped_acquire gil;
    Py_DECREF(py_func_);
  }

  void operator()() override {
    py::gil_scoped_acquire gil;
    VLOG(3) << "Call PyTensorVoidHook";

    try {
      PyObject_CallFunctionObjArgs(py_func_, nullptr);
    } catch (platform::EnforceNotMet& e) {
      throw std::move(e);
    } catch (std::exception& e) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Hook function of Tensor raises an exception: %s.", e.what()));
    } catch (...) {
      PADDLE_THROW(platform::errors::Fatal(
          "Hook function of Tensor raises an unknown exception."));
    }
  }

 private:
  PyObject* py_func_;
};

extern void InitTensorWithNumpyValue(TensorObject* self,
                                     const pybind11::object& array,
                                     bool zero_copy);

extern PyTypeObject* p_tensor_type;

Py_ssize_t GetSliceIndexFromPyObject(PyObject* obj) {
  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type))) {
    VLOG(6) << "Call GetSliceIndexFromTensor in Eager";
    paddle::experimental::Tensor tensor = CastPyArg2Tensor(obj, 0);
    PADDLE_ENFORCE_EQ(
        tensor.initialized(), true,
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

static PyObject* tensor_method_numpy(TensorObject* self, PyObject* args,
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
        api.PyArray_DescrFromType_(pybind11::detail::npy_api::NPY_FLOAT_), 1,
        py_dims, py_strides, nullptr,
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
      api.PyArray_Type_, api.PyArray_DescrFromType_(numpy_dtype),
      tensor_dims.size(), py_dims, py_strides, nullptr,
      pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |
          pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,
      nullptr);

  if (!self->tensor.impl()->initialized()) {
    return array;
  }

  if (self->tensor.is_cpu() || self->tensor.is_gpu_pinned()) {
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
          place, dense_tensor->data(), sizeof_dtype * numel);
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      // deep copy
      paddle::memory::Copy(
          place,
          reinterpret_cast<void*>(pybind11::detail::array_proxy(array)->data),
          place, dense_tensor->data(), sizeof_dtype * numel);
    }

#if defined(PADDLE_WITH_CUDA)
  } else if (self->tensor.is_gpu()) {
    if (self->tensor.is_selected_rows()) {
      VLOG(6) << "Getting SelectedRows's numpy value";
      auto* selected_rows =
          static_cast<phi::SelectedRows*>(self->tensor.impl().get());
      auto* dense_tensor = static_cast<paddle::framework::LoDTensor*>(
          selected_rows->mutable_value());
      paddle::platform::GpuMemcpySync(
          pybind11::detail::array_proxy(array)->data, dense_tensor->data(),
          paddle::framework::DataTypeSize(dense_tensor->dtype()) *
              dense_tensor->numel(),
          cudaMemcpyDeviceToHost);
    } else {
      VLOG(6) << "Getting DenseTensor's numpy value";
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(self->tensor.impl());
      paddle::platform::GpuMemcpySync(
          pybind11::detail::array_proxy(array)->data, dense_tensor->data(),
          paddle::framework::DataTypeSize(dense_tensor->dtype()) *
              dense_tensor->numel(),
          cudaMemcpyDeviceToHost);
    }
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

static PyObject* tensor_method__copy_to(TensorObject* self, PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 0), 0);
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 1), 1);
  auto cp_tensor = self->tensor.copy_to(place, blocking);
  egr::EagerUtils::autograd_meta(&cp_tensor)->SetStopGradient(true);
  egr::EagerUtils::autograd_meta(&cp_tensor)
      ->SetPersistable(
          egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  return ToPyObject(cp_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_cpu(TensorObject* self, PyObject* args,
                                   PyObject* kwargs) {
  EAGER_TRY
  auto cp_tensor = self->tensor.copy_to(phi::CPUPlace(), true);
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
    if (src_tensor.defined()) {
      self->tensor.copy_(src_tensor, src_tensor.inner_place(), blocking);
    }
  } else {
    if (src_tensor.defined()) {
      self->tensor.copy_(src_tensor, self->tensor.inner_place(), blocking);
    }
  }

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
    set_to_zero = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 0), 0);
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

  if (grad->impl()) {
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
  PADDLE_ENFORCE_EQ(
      self->tensor.is_initialized(), true,
      platform::errors::InvalidArgument(
          "tensor %s has not been initialized, we can only slice initialized "
          "tensor please init it first with numpy or other tensor.",
          self->tensor.name()));
  auto tensor = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  ParseIndexingSlice(tensor, _index, &slice_axes, &slice_starts, &slice_ends,
                     &slice_strides, &decrease_axis, &none_axes, &infer_flags,
                     &list_select_idxs, &list_select_flag);

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
    if (op_type == "slice") {
      out = slice_dygraph_function(self->tensor, paddle::experimental::Tensor(),
                                   paddle::experimental::Tensor(), {}, {},
                                   std::move(attrs));
    } else if (op_type == "strided_slice") {
      out = strided_slice_dygraph_function(
          self->tensor, paddle::experimental::Tensor(),
          paddle::experimental::Tensor(), paddle::experimental::Tensor(), {},
          {}, {}, attrs);
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

      paddle::experimental::Tensor new_out;
      framework::AttributeMap attrs = {{"axes", none_axes}};
      new_out = std::get<0>(unsqueeze2_dygraph_function(out, std::move(attrs)));
      return ToPyObject(new_out);
    }
  }

  // the index is a list
  if (list_select_flag) {
    auto select_index = paddle::experimental::Tensor(
        egr::Controller::Instance().GenerateUniqueName());
    auto idx_tensor = std::make_shared<phi::DenseTensor>();
    select_index.set_impl(idx_tensor);
    auto* dev_ctx = platform::DeviceContextPool::Instance().Get(
        egr::Controller::Instance().GetExpectedPlace());
    paddle::framework::TensorFromVector(list_select_idxs, *dev_ctx,
                                        idx_tensor.get());
    framework::AttributeMap attrs = {{"dim", 0}};
    out = index_select_dygraph_function(self->tensor, select_index,
                                        std::move(attrs));
  }

  return ToPyObject(out);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__getitem_from_offset(TensorObject* self, PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  auto ptr = static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  PADDLE_ENFORCE_NOT_NULL(
      ptr, platform::errors::InvalidArgument("%s is not a DenseTensor.",
                                             self->tensor.name()));
  const auto& tensor = *ptr;
  PADDLE_ENFORCE_EQ(
      tensor.IsInitialized(), true,
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
    PADDLE_ENFORCE_EQ(numel, 1,
                      platform::errors::InvalidArgument(
                          "only one element tensors can be converted to Python "
                          "scalars when no input coordinates"));
  } else if (PyTuple_Size(args) == 1) {
    offset = CastPyArg2AttrLong(PyTuple_GET_ITEM(args, 0), 0);
    PADDLE_ENFORCE_LT(
        offset, numel,
        platform::errors::InvalidArgument(
            "index %d is out of bounds for size %d", offset, numel));
  } else {
    PADDLE_ENFORCE_EQ(PyTuple_Size(args), dims.size(),
                      platform::errors::InvalidArgument(
                          "incorrect number of indices for Tensor"));

    for (Py_ssize_t i = 0; i < PyTuple_Size(args); ++i) {
      size_t index = CastPyArg2AttrLong(PyTuple_GET_ITEM(args, i), i);
      PADDLE_ENFORCE_LT(
          index, dims[i],
          platform::errors::InvalidArgument(
              "index %d is out fo bounds for axis %d with size %d", index, i,
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
        api.PyArray_Type_, api.PyArray_DescrFromType_(numpy_dtype), 1,       \
        py_dims, py_strides, nullptr,                                        \
        pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_ |                      \
            pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_,                 \
        nullptr);                                                            \
    std::memcpy(                                                             \
        reinterpret_cast<void*>(pybind11::detail::array_proxy(array)->data), \
        static_cast<void*>(&b), sizeof(b));                                  \
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
    ParseIndexingSlice(self_tensor, index_ptr, &axes, &starts, &ends, &steps,
                       &decrease_axes, &none_axes, &infer_flags,
                       &list_select_idxs, &list_select_flag);

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
          false, platform::errors::InvalidArgument(
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

      if (value_tensor_tmp.place() == paddle::PlaceType::kUNK) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        SetTensorFromPyArray(
            static_cast<phi::DenseTensor*>(value_tensor_tmp.impl().get()),
            value, platform::Place(platform::CUDAPlace(0)), false);
#else
        SetTensorFromPyArray(
            static_cast<phi::DenseTensor*>(value_tensor_tmp.impl().get()),
            value, platform::Place(platform::CPUPlace()), false);
#endif
      } else {
        SetTensorFromPyArray(
            static_cast<phi::DenseTensor*>(value_tensor_tmp.impl().get()),
            value, value_tensor_tmp.inner_place(), false);
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
      self->tensor = set_value__dygraph_function(self->tensor, value_tensor, {},
                                                 {}, {}, attrs);
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
    if (self->tensor.place() == paddle::PlaceType::kUNK) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      SetTensorFromPyArray(self_tensor, self_numpy,
                           platform::Place(platform::CUDAPlace(0)), false);
#else
      SetTensorFromPyArray(self_tensor, self_numpy,
                           platform::Place(platform::CPUPlace()), false);
#endif
    } else {
      SetTensorFromPyArray(self_tensor, self_numpy, self->tensor.inner_place(),
                           false);
    }
  }
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_register_grad_hook(TensorObject* self, PyObject* args,
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
        rank_info.first, rank_info.second,
        std::make_shared<PyTensorHook>(hook_func));

  } else {
    VLOG(6) << "Register hook for non leaf tensor: " << self->tensor.name();
    std::shared_ptr<egr::GradNodeBase> grad_node =
        egr::EagerUtils::grad_node(self->tensor);
    auto rank_info =
        egr::EagerUtils::unsafe_autograd_meta(self->tensor)->OutRankInfo();

    PyObject* hook_func = PyTuple_GET_ITEM(args, 0);

    hook_id = grad_node->RegisterGradientHook(
        rank_info.first, rank_info.second,
        std::make_shared<PyTensorHook>(hook_func));
  }
  return ToPyObject(hook_id);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_remove_grad_hook(TensorObject* self, PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  VLOG(6) << "Remove the registered hook for tensor: " << self->tensor.name();
  std::shared_ptr<egr::GradNodeBase> grad_node =
      egr::EagerUtils::grad_node(self->tensor);

  int64_t hook_id = pybind::CastPyArg2AttrLong(PyTuple_GET_ITEM(args, 0), 0);

  return ToPyObject(grad_node->RemoveGradientHook(hook_id));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_register_reduce_hook(TensorObject* self, PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  VLOG(4) << "Register reduce hook for tensor: " << self->tensor.name();

  std::shared_ptr<egr::GradNodeBase> grad_node =
      egr::EagerUtils::grad_node(self->tensor);
  PADDLE_ENFORCE_EQ(egr::egr_utils_api::IsLeafTensor(self->tensor), true,
                    platform::errors::InvalidArgument(
                        "Only can register backward hook for leaf Tensor."));
  PADDLE_ENFORCE_EQ(
      !egr::EagerUtils::unsafe_autograd_meta(self->tensor)->StopGradient(),
      true, platform::errors::InvalidArgument(
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
      std::make_shared<PyTensorVoidHook>(hook_func));

  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__set_grad_type(TensorObject* self, PyObject* args,
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
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__clear(TensorObject* self, PyObject* args,
                               PyObject* kwargs) {
  EAGER_TRY
  self->tensor.reset();
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__copy_gradient_from(TensorObject* self, PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  auto src = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);
  if (self->tensor.is_initialized()) {
    PADDLE_ENFORCE_EQ(self->tensor.dtype(), src.dtype(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different data type with Tensor %s",
                          self->tensor.name(), src.name()));
    PADDLE_ENFORCE_EQ(self->tensor.impl()->type_info().id(),
                      src.impl()->type_info().id(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different type with Tensor %s, Tensor "
                          "ShareGradientDataWith cannot be performed!",
                          self->tensor.name(), src.name()));
  }
  VLOG(6) << "Tensor copy gradient from: " << src.name();
  auto* p_grad = egr::EagerUtils::mutable_grad(self->tensor);
  if (p_grad) {
    PADDLE_ENFORCE_EQ(src.initialized(), true,
                      platform::errors::InvalidArgument(
                          "Tensor %s has not been initialized", src.name()));
    p_grad->set_impl(src.impl());
  }
  Py_INCREF(Py_None);
  return Py_None;
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

static PyObject* tensor_method_is_sparse(TensorObject* self, PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  return ToPyObject(self->tensor.is_sparse_coo_tensor() ||
                    self->tensor.is_sparse_csr_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_is_sparse_coo(TensorObject* self, PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  return ToPyObject(self->tensor.is_sparse_coo_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_is_sparse_csr(TensorObject* self, PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  return ToPyObject(self->tensor.is_sparse_csr_tensor());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_to_sparse_coo(TensorObject* self, PyObject* args,
                                             PyObject* kwargs) {
  EAGER_TRY
  int64_t sparse_dim = CastPyArg2AttrLong(PyTuple_GET_ITEM(args, 0), 0);
  auto coo_tensor = self->tensor.to_sparse_coo(sparse_dim);
  egr::EagerUtils::autograd_meta(&coo_tensor)
      ->SetStopGradient(
          egr::EagerUtils::autograd_meta(&self->tensor)->StopGradient());
  egr::EagerUtils::autograd_meta(&coo_tensor)
      ->SetPersistable(
          egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  return ToPyObject(coo_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_to_sparse_csr(TensorObject* self, PyObject* args,
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

static PyObject* tensor_method_to_dense(TensorObject* self, PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  auto dense_tensor = self->tensor.to_dense();
  egr::EagerUtils::autograd_meta(&dense_tensor)
      ->SetStopGradient(
          egr::EagerUtils::autograd_meta(&self->tensor)->StopGradient());
  egr::EagerUtils::autograd_meta(&dense_tensor)
      ->SetPersistable(
          egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  return ToPyObject(dense_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__inplace_version(TensorObject* self, PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  uint32_t inplace_version = self->tensor.current_inplace_version();

  return ToPyObject(inplace_version);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_element_size(TensorObject* self, PyObject* args,
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
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_is_selected_rows(TensorObject* self,
                                                PyObject* args,
                                                PyObject* kwargs) {
  EAGER_TRY
  return ToPyObject(self->tensor.is_selected_rows());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor_method_get_rows(TensorObject* self, PyObject* args,
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

  paddle::experimental::Tensor* grad =
      egr::EagerUtils::mutable_grad(self->tensor);
  if (grad && grad->defined() && grad->is_dense_tensor() &&
      grad->initialized()) {
    grad->reset_inplace_version(set_to_zero);
  }
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
    {"_is_dense_tensor_hold_allocation",
     (PyCFunction)(
         void (*)(void))tensor_method__is_dense_tensor_hold_allocation,
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
    {"_getitem_index_not_tensor",
     (PyCFunction)(void (*)(void))tensor__getitem_index_not_tensor,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_getitem_from_offset",
     (PyCFunction)(void (*)(void))tensor__getitem_from_offset,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"__setitem_eager_tensor__",
     (PyCFunction)(void (*)(void))tensor_method__setitem_eager_tensor,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_register_grad_hook",
     (PyCFunction)(void (*)(void))tensor_register_grad_hook,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_remove_grad_hook", (PyCFunction)(void (*)(void))tensor_remove_grad_hook,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_register_backward_hook",
     (PyCFunction)(void (*)(void))tensor_register_reduce_hook,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_set_grad_type", (PyCFunction)(void (*)(void))tensor__set_grad_type,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_clear", (PyCFunction)(void (*)(void))tensor__clear,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_copy_gradient_from",
     (PyCFunction)(void (*)(void))tensor__copy_gradient_from,
     METH_VARARGS | METH_KEYWORDS, NULL},
    /***the method of sparse tensor****/
    {"non_zero_indices",
     (PyCFunction)(void (*)(void))tensor_method_get_non_zero_indices,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"non_zero_elements",
     (PyCFunction)(void (*)(void))tensor_method_get_non_zero_elements,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"non_zero_crows",
     (PyCFunction)(void (*)(void))tensor_method_get_non_zero_crows,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"non_zero_cols",
     (PyCFunction)(void (*)(void))tensor_method_get_non_zero_cols,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_sparse", (PyCFunction)(void (*)(void))tensor_method_is_sparse,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_sparse_coo", (PyCFunction)(void (*)(void))tensor_method_is_sparse_coo,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_sparse_csr", (PyCFunction)(void (*)(void))tensor_method_is_sparse_csr,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"to_sparse_coo", (PyCFunction)(void (*)(void))tensor_method_to_sparse_coo,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"to_sparse_csr", (PyCFunction)(void (*)(void))tensor_method_to_sparse_csr,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"to_dense", (PyCFunction)(void (*)(void))tensor_method_to_dense,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"element_size", (PyCFunction)(void (*)(void))tensor_method_element_size,
     METH_VARARGS | METH_KEYWORDS, NULL},
    /***the method of sparse tensor****/
    {"_inplace_version", (PyCFunction)(void (*)(void))tensor__inplace_version,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_bump_inplace_version",
     (PyCFunction)(void (*)(void))tensor__bump_inplace_version,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_selected_rows",
     (PyCFunction)(void (*)(void))tensor_method_is_selected_rows,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"rows", (PyCFunction)(void (*)(void))tensor_method_get_rows,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_reset_grad_inplace_version",
     (PyCFunction)(void (*)(void))tensor__reset_grad_inplace_version,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

}  // namespace pybind
}  // namespace paddle
