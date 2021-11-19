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
// #define PY_ARRAY_UNIQUE_SYMBOL Paddle_PyArray_API_F
#define INIT_NUMPY_ARRAY_CPP

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <Python.h>

#include <string>
#include <vector>

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/include/core.h"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wconversion-null"

namespace paddle {
namespace pybind {

int init_numpy_f() {
  import_array();
  return 0;
}
static const int numpy_initialized_f = init_numpy_f();

namespace py = ::pybind11;

// TODO(wanghuancoder) we must build paddle whl package with lower numpy version
bool check_numpy_available() {
  static bool ret = []() {
    if (_import_array() >= 0) {
      return true;
    }

    std::string message = "Failed to initialize NumPy";
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    if (value) {
      PyObject* err_msg = PyObject_Str(value);
      PyObject* err_msg2 =
          PyUnicode_AsEncodedString(err_msg, "utf-8", "strict");
      if (err_msg2) {
        LOG(WARNING) << "Numpy Error: '" << PyBytes_AS_STRING(err_msg2)
                     << "'. You can try upgrading numpy.";
        Py_XDECREF(err_msg2);
      }
      Py_XDECREF(err_msg);
    }
    PyErr_Clear();
    return false;
  }();
  return ret;
}

extern PyTypeObject* p_eager_tensor_type;

static PyObject* eager_api_set_expected_place(PyObject* self, PyObject* args,
                                              PyObject* kwargs) {
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 0), 0);
  egr::Controller::Instance().SetExpectedPlace(place);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject* eager_api_scale(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
  // TODO(jiabin): Sync Tensor and Variable here when we support
  egr::EagerTensor ret =
      egr::scale(reinterpret_cast<EagerTensorObject*>(PyTuple_GET_ITEM(args, 0))
                     ->eagertensor,
                 CastPyArg2AttrFloat(PyTuple_GET_ITEM(args, 1), 1),
                 CastPyArg2AttrFloat(PyTuple_GET_ITEM(args, 2), 2),
                 CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3),
                 CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4));
  return ToPyObject(ret);
}

class EagerNumpyAllocation : public paddle::memory::allocation::Allocation {
 public:
  explicit EagerNumpyAllocation(PyObject* numpy_data, pten::DataType dtype)
      : Allocation(
            static_cast<void*>(
                (reinterpret_cast<PyArrayObject_fields*>(numpy_data))->data),
            pten::DataTypeSize(dtype) * PyArray_Size(numpy_data),
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

static inline PyObject* eager_api_numpy_to_tensor(
    PyObject* numpy_data, pten::DataType dtype,
    const paddle::platform::Place& place, bool stop_gradient) {
  std::vector<int64_t> vec_dims;
  auto numpy_shape = PyArray_DIMS(reinterpret_cast<PyArrayObject*>(numpy_data));
  int rank = PyArray_NDIM(reinterpret_cast<PyArrayObject*>(numpy_data));
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
  // TODO(jiabin): Support Kwargs here
  PyObject* data = PyTuple_GET_ITEM(args, 0);
  auto str_dtype = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 1), 1);
  pten::DataType dtype = pten::String2DataType(str_dtype);
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 2), 2);
  bool stop_gradient = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);
  // TODO(jiabin): Support this when python given name
  // auto str_name = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 4), 4);

  if (check_numpy_available() && PyArray_Check(data)) {
    return eager_api_numpy_to_tensor(data, dtype, place, stop_gradient);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Eater to_tensor only support numpy to tensor."));
    Py_INCREF(Py_None);
    return Py_None;
  }
}

static PyObject* eager_api_retain_grad_for_tensor(PyObject* self,
                                                  PyObject* args,
                                                  PyObject* kwargs) {
  RetainGradForTensor(CastPyArg2EagerTensor(PyTuple_GET_ITEM(args, 0), 0));
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject* eager_api_run_backward(PyObject* self, PyObject* args,
                                        PyObject* kwargs) {
  auto tensors = CastPyArg2VectorOfEagerTensor(PyTuple_GET_ITEM(args, 0), 0);
  auto grad_tensors =
      CastPyArg2VectorOfEagerTensor(PyTuple_GET_ITEM(args, 1), 1);
  RunBackward(tensors, grad_tensors,
              CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2));
  Py_INCREF(Py_None);
  return Py_None;
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
