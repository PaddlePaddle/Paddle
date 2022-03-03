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
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/api/lib/utils/tensor_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

extern PyTypeObject* p_tensor_type;
extern PyTypeObject* g_multidevicefeedreader_pytype;
extern PyTypeObject* g_orderedmultidevicefeedreader_pytype;

size_t PyArray_Size_(PyObject* numpy_data) {
  size_t res = 1;
  auto dims = pybind11::detail::array_proxy(numpy_data)->dimensions;
  auto nd = pybind11::detail::array_proxy(numpy_data)->nd;
  while (nd--) {
    res *= (*dims++);
  }
  return res;
}

class EagerNumpyAllocation : public phi::Allocation {
 public:
  explicit EagerNumpyAllocation(PyObject* numpy_data, phi::DataType dtype)
      : Allocation(
            static_cast<void*>(pybind11::detail::array_proxy(numpy_data)->data),
            framework::DataTypeSize(dtype) * PyArray_Size_(numpy_data),
            paddle::platform::CPUPlace()),
        arr_(numpy_data) {
    PADDLE_ENFORCE_NOT_NULL(arr_, platform::errors::InvalidArgument(
                                      "The underlying PyObject pointer of "
                                      "numpy array cannot be nullptr"));
    PADDLE_ENFORCE_NE(
        arr_, Py_None,
        platform::errors::PreconditionNotMet(
            "The underlying PyObject pointer of numpy array cannot be None"));
    Py_INCREF(arr_);
  }
  ~EagerNumpyAllocation() override {
    py::gil_scoped_acquire gil;
    Py_DECREF(arr_);
  }

 private:
  PyObject* arr_;
};

static PyObject* eager_api_set_expected_place(PyObject* self, PyObject* args,
                                              PyObject* kwargs) {
  EAGER_TRY
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 0), 0);
  egr::Controller::Instance().SetExpectedPlace(place);

  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_get_expected_place(PyObject* self, PyObject* args,
                                              PyObject* kwargs) {
  EAGER_TRY
  return ToPyObject(egr::Controller::Instance().GetExpectedPlace());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_scale(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
  EAGER_TRY
  // TODO(jiabin): Sync Tensor and Variable here when we support
  paddle::experimental::Tensor ret = egr::scale(
      reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor,
      CastPyArg2AttrFloat(PyTuple_GET_ITEM(args, 1), 1),
      CastPyArg2AttrFloat(PyTuple_GET_ITEM(args, 2), 2),
      CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3),
      CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4));
  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_run_backward(PyObject* self, PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  auto tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0);
  auto grad_tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 1), 1);
  egr::RunBackward(tensors, grad_tensors,
                   CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2));
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_tensor_copy(PyObject* self, PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor& src =
      reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor;
  paddle::experimental::Tensor& dst =
      reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 1))->tensor;
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 2), 2);
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);

  dst = src.copy_to(phi::TransToPhiBackend(place), blocking);
  egr::EagerUtils::autograd_meta(&dst)->SetStopGradient(
      egr::EagerUtils::autograd_meta(&(src))->StopGradient());
  egr::EagerUtils::autograd_meta(&dst)->SetPersistable(
      egr::EagerUtils::autograd_meta(&(src))->Persistable());
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_read_next_tensor_list(PyObject* self, PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  auto tensor_base_list =
      CastPyArg2VectorOfTensorBase(PyTuple_GET_ITEM(args, 0), 0);
  std::vector<paddle::experimental::Tensor> tensor_list;
  tensor_list.reserve(tensor_base_list.size());
  auto func = [](framework::Tensor& tensor_base) {
    paddle::experimental::Tensor tensor(
        egr::Controller::Instance().GenerateUniqueName());
    auto autograd_meta = egr::EagerUtils::autograd_meta(&tensor);
    autograd_meta->SetPersistable(false);
    autograd_meta->SetStopGradient(true);
    tensor.set_impl(std::make_shared<phi::DenseTensor>(tensor_base));
    return tensor;
  };
  for (auto& tensor_base : tensor_base_list) {
    tensor_list.emplace_back(func(tensor_base));
  }
  return ToPyObject(tensor_list);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef variable_functions[] = {
    {"scale", (PyCFunction)(void (*)(void))eager_api_scale,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_set_expected_place",
     (PyCFunction)(void (*)(void))eager_api_set_expected_place,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_get_expected_place",
     (PyCFunction)(void (*)(void))eager_api_get_expected_place,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"run_backward", (PyCFunction)(void (*)(void))eager_api_run_backward,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"tensor_copy", (PyCFunction)(void (*)(void))eager_api_tensor_copy,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"read_next_tensor_list",
     (PyCFunction)(void (*)(void))eager_api_read_next_tensor_list,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

void BindFunctions(PyObject* module) {
  if (PyModule_AddFunctions(module, variable_functions) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle erroe in BindFunctions(PyModule_AddFunctions)."));
    return;
  }
}

}  // namespace pybind
}  // namespace paddle
