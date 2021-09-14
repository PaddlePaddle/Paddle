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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL Paddle_PyArray_API_M
#define INIT_NUMPY_ARRAY_CPP

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <Python.h>

#include <string>
#include <vector>

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/top/api/include/tensor.h"
#include "paddle/top/core/convert_utils.h"
#include "paddle/top/core/dense_tensor.h"
#include "paddle/top/core/dtype.h"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wconversion-null"

namespace paddle {
namespace pybind {

int init_numpy_m() {
  import_array();
  return 0;
}
static const int numpy_initialized_m = init_numpy_m();

extern PyTypeObject* pEagerTensorType;

static PyObject* eager_tensor_method_numpy(EagerTensorObject* self,
                                           PyObject* args, PyObject* kwargs) {
  if (!self->eagertensor.initialized()) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  auto tensor_dims = self->eagertensor.shape();
  auto numpy_dtype = pt::TensorDtype2NumpyDtype(self->eagertensor.type());
  auto sizeof_dtype = pt::DataTypeSize(self->eagertensor.type());
  npy_intp py_dims[paddle::framework::DDim::kMaxRank];
  npy_intp py_strides[paddle::framework::DDim::kMaxRank];

  size_t numel = 1;
  for (int i = tensor_dims.size() - 1; i >= 0; --i) {
    py_dims[i] = static_cast<size_t>(tensor_dims[i]);
    py_strides[i] = sizeof_dtype * numel;
    numel *= py_dims[i];
  }

  PyObject* array =
      PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(numpy_dtype),
                           tensor_dims.size(), py_dims, py_strides, nullptr,
                           NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, nullptr);

  if (self->eagertensor.is_cpu()) {
    auto dense_tensor =
        std::dynamic_pointer_cast<pt::DenseTensor>(self->eagertensor.impl());
    platform::CPUPlace place;
    // deep copy
    paddle::memory::Copy(
        place, reinterpret_cast<void*>(
                   (reinterpret_cast<PyArrayObject_fields*>(array))->data),
        place, dense_tensor->data(), sizeof_dtype * numel);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Tensor.numpy() only support cpu tensor."));
    Py_INCREF(Py_None);
    return Py_None;
  }

  return array;
}

static PyObject* eager_tensor_method_is_initialized(EagerTensorObject* self,
                                                    PyObject* args,
                                                    PyObject* kwargs) {
  return ToPyObject(self->eagertensor.initialized());
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
