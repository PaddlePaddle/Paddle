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

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/include/core.h"
namespace paddle {
namespace pybind {

extern PyTypeObject* pEagerTensorType;

static PyObject* eager_tensor_method_numpy(EagerTensorObject* self,
                                           PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  self->eagertensor.SyncToTensor();
  if (!self->eagertensor.initialized()) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  auto tensor_dims = self->eagertensor.shape();
  auto numpy_dtype = TensorDtype2NumpyDtype(self->eagertensor.type());
  auto sizeof_dtype = pten::DataTypeSize(self->eagertensor.type());
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

  if (self->eagertensor.is_cpu()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<pten::DenseTensor>(self->eagertensor.impl());
    platform::CPUPlace place;
    // deep copy
    paddle::memory::Copy(place, reinterpret_cast<void*>(
                                    pybind11::detail::array_proxy(array)->data),
                         place, dense_tensor->data(), sizeof_dtype * numel);
#if defined(PADDLE_WITH_CUDA)
  } else if (self->eagertensor.is_cuda()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<pten::DenseTensor>(self->eagertensor.impl());

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

static PyObject* eager_tensor_method_is_initialized(EagerTensorObject* self,
                                                    PyObject* args,
                                                    PyObject* kwargs) {
  EAGER_TRY
  if (self->eagertensor.Var().IsInitialized()) {
    self->eagertensor.SyncToTensor();
  }
  return ToPyObject(self->eagertensor.initialized());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef variable_methods[] = {
    {"numpy", (PyCFunction)(void (*)(void))eager_tensor_method_numpy,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_is_initialized",
     (PyCFunction)(void (*)(void))eager_tensor_method_is_initialized,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

}  // namespace pybind
}  // namespace paddle
