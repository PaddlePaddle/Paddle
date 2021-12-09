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
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/include/core.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

extern PyTypeObject* p_eager_tensor_type;

size_t PyArray_Size_(PyObject* numpy_data) {
  size_t res = 1;
  auto dims = pybind11::detail::array_proxy(numpy_data)->dimensions;
  auto nd = pybind11::detail::array_proxy(numpy_data)->nd;
  while (nd--) {
    res *= (*dims++);
  }
  return res;
}

class EagerNumpyAllocation : public paddle::memory::allocation::Allocation {
 public:
  explicit EagerNumpyAllocation(PyObject* numpy_data, pten::DataType dtype)
      : Allocation(
            static_cast<void*>(pybind11::detail::array_proxy(numpy_data)->data),
            pten::DataTypeSize(dtype) * PyArray_Size_(numpy_data),
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

static PyObject* eager_api_scale(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
  EAGER_TRY
  // TODO(jiabin): Sync Tensor and Variable here when we support
  egr::EagerTensor ret =
      egr::scale(reinterpret_cast<EagerTensorObject*>(PyTuple_GET_ITEM(args, 0))
                     ->eagertensor,
                 CastPyArg2AttrFloat(PyTuple_GET_ITEM(args, 1), 1),
                 CastPyArg2AttrFloat(PyTuple_GET_ITEM(args, 2), 2),
                 CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3),
                 CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4));
  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_numpy_to_tensor(PyObject* numpy_data,
                                           pten::DataType dtype,
                                           const paddle::platform::Place& place,
                                           bool stop_gradient) {
  std::vector<int64_t> vec_dims;
  auto numpy_shape = pybind11::detail::array_proxy(numpy_data)->dimensions;
  int rank = pybind11::detail::array_proxy(numpy_data)->nd;
  for (int i = 0; i < rank; i++) {
    vec_dims.push_back(static_cast<int64_t>(numpy_shape[i]));
  }
  paddle::framework::DDim dims = paddle::framework::make_ddim(vec_dims);

  // TODO(jiabin): Support GPU later
  auto meta = pten::DenseTensorMeta(dtype, dims);
  auto holder = std::make_shared<EagerNumpyAllocation>(numpy_data, dtype);
  auto shared_storage =
      pten::make_intrusive<paddle::experimental::SharedStorage>(holder, 0);
  std::shared_ptr<pten::DenseTensor> densetensor(
      new pten::DenseTensor(std::move(shared_storage), std::move(meta)));

  PyObject* obj = p_eager_tensor_type->tp_alloc(p_eager_tensor_type, 0);
  if (obj) {
    auto v = reinterpret_cast<EagerTensorObject*>(obj);
    new (&(v->eagertensor)) egr::EagerTensor();
    v->eagertensor.set_impl(densetensor);
    v->eagertensor.set_name(egr::Controller::Instance().GenerateUniqueName());
    auto meta = egr::EagerUtils::autograd_meta(&(v->eagertensor));
    meta->SetStopGradient(stop_gradient);

    // Created tensor will be leaf tensor
    // So we append AccumulationNode to it.
    auto accumulation_node = std::make_shared<egr::GradNodeAccumulation>();
    meta->SetGradNode(accumulation_node);

    // TODO(jiabin): Shall we increase ref cnt here to make python ref cnt num
    // correctly?
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "tp_alloc return null, can not new a PyObject."));
  }

  return obj;
}

static PyObject* eager_api_to_tensor(PyObject* self, PyObject* args,
                                     PyObject* kwargs) {
  EAGER_TRY
  // TODO(jiabin): Support Kwargs here
  PyObject* data = PyTuple_GET_ITEM(args, 0);
  auto str_dtype = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 1), 1);
  pten::DataType dtype = pten::String2DataType(str_dtype);
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 2), 2);
  bool stop_gradient = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);
  // TODO(jiabin): Support this when python given name
  // auto str_name = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 4), 4);

  if (pybind11::detail::npy_api::get().PyArray_Check_(data)) {
    return eager_api_numpy_to_tensor(data, dtype, place, stop_gradient);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Eater to_tensor only support numpy to tensor."));
    Py_INCREF(Py_None);
    return Py_None;
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_retain_grad_for_tensor(PyObject* self,
                                                  PyObject* args,
                                                  PyObject* kwargs) {
  EAGER_TRY
  egr::egr_utils_api::RetainGradForTensor(
      CastPyArg2EagerTensor(PyTuple_GET_ITEM(args, 0), 0));
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_run_backward(PyObject* self, PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  auto tensors = CastPyArg2VectorOfEagerTensor(PyTuple_GET_ITEM(args, 0), 0);
  auto grad_tensors =
      CastPyArg2VectorOfEagerTensor(PyTuple_GET_ITEM(args, 1), 1);
  RunBackward(tensors, grad_tensors,
              CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2));
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef variable_functions[] = {
    {"to_tensor", (PyCFunction)(void (*)(void))eager_api_to_tensor,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"scale", (PyCFunction)(void (*)(void))eager_api_scale,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_set_expected_place",
     (PyCFunction)(void (*)(void))eager_api_set_expected_place,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"retain_grad_for_tensor",
     (PyCFunction)(void (*)(void))eager_api_retain_grad_for_tensor,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"run_backward", (PyCFunction)(void (*)(void))eager_api_run_backward,
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
