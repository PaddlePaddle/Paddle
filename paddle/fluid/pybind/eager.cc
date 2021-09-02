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
#define PY_ARRAY_UNIQUE_SYMBOL Paddle_PyArray_API
#define INIT_NUMPY_ARRAY_CPP

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <Python.h>

#include <string>
#include <vector>

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/top/api/include/tensor.h"
#include "paddle/top/core/convert_utils.h"
#include "paddle/top/core/dense_tensor.h"
#include "paddle/top/core/dtype.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

PyTypeObject* pEagerTensorType;

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

PyObject* eagertensor_new(PyTypeObject* type, PyObject* args,
                          PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  return obj;
}

static void eagertensor_dealloc(EagerTensorObject* self) {
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static int eagertensor_init(EagerTensorObject* self, PyObject* args,
                            PyObject* kwargs) {
  return 0;
}

PyObject* eagertensor_str(PyObject* self) { return ToPyObject(""); }

PyObject* eager_tensor_properties_get_shape(EagerTensorObject* self,
                                            void* closure) {
  auto ddim = self->eagertensor.shape();
  std::vector<int64_t> value;
  size_t rank = static_cast<size_t>(ddim.size());
  value.resize(rank);
  for (size_t i = 0; i < rank; i++) {
    value[i] = ddim[i];
  }

  return ToPyObject(value);
}

PyObject* eager_tensor_properties_get_stop_gradient(EagerTensorObject* self,
                                                    void* closure) {
  auto meta = egr::EagerUtils::autograd_meta(self->eagertensor);
  return ToPyObject(meta->StopGradient());
}

int eager_tensor_properties_set_stop_gradient(EagerTensorObject* self,
                                              PyObject* value, void* closure) {
  auto meta = egr::EagerUtils::autograd_meta(self->eagertensor);
  meta->SetNumericStopGradient(CastPyArg2AttrBoolean(value, 0));
  return 0;
}

PyObject* eager_tensor_properties_get_dtype(EagerTensorObject* self,
                                            void* closure) {
  return ToPyObject(DataType2String(self->eagertensor.type()));
}

PyObject* eager_tensor_properties_get_place_str(EagerTensorObject* self,
                                                void* closure) {
  std::stringstream ostr;
  ostr << self->eagertensor.place();
  return ToPyObject(ostr.str());
}

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

static PyObject* eager_api_scale(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
  std::vector<pt::Tensor> ret =
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
  explicit EagerNumpyAllocation(PyObject* numpy_data, pt::DataType dtype)
      : Allocation(
            static_cast<void*>(
                (reinterpret_cast<PyArrayObject_fields*>(numpy_data))->data),
            pt::DataTypeSize(dtype) * PyArray_Size(numpy_data),
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

static inline PyObject* eager_api_numpy_to_tensor(PyObject* numpy_data,
                                                  pt::DataType dtype,
                                                  int place_id, int device_id,
                                                  bool stop_gradient) {
  std::vector<int64_t> vec_dims;
  auto numpy_shape = PyArray_DIMS(reinterpret_cast<PyArrayObject*>(numpy_data));
  int rank = PyArray_NDIM(reinterpret_cast<PyArrayObject*>(numpy_data));
  for (int i = 0; i < rank; i++) {
    vec_dims.push_back(static_cast<int64_t>(numpy_shape[i]));
  }
  paddle::framework::DDim dims = paddle::framework::make_ddim(vec_dims);

  std::unique_ptr<pt::TensorMeta> meta(
      new pt::TensorMeta(dims, static_cast<pt::Backend>(place_id), dtype));

  std::shared_ptr<pt::DenseTensor> densetensor(
      new pt::DenseTensor(std::move(meta)));

  auto holder = std::make_shared<EagerNumpyAllocation>(numpy_data, dtype);
  densetensor->ShareAllocation(holder);

  PyObject* obj = pEagerTensorType->tp_alloc(pEagerTensorType, 0);
  if (obj) {
    auto v = (EagerTensorObject*)obj;  // NOLINT
    v->eagertensor.SetImpl(densetensor);
    auto meta = egr::EagerUtils::autograd_meta(v->eagertensor);
    meta->SetNumericStopGradient(stop_gradient);
  }

  return obj;
}

static PyObject* eager_api_to_tensor(PyObject* self, PyObject* args,
                                     PyObject* kwargs) {
  PyObject* data = PyTuple_GET_ITEM(args, 0);
  auto str_dtype = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 1), 1);
  pt::DataType dtype = pt::String2DataTyep(str_dtype);
  int place_id = CastPyArg2AttrInt(PyTuple_GET_ITEM(args, 2), 2);
  int device_id = CastPyArg2AttrInt(PyTuple_GET_ITEM(args, 3), 3);
  bool stop_gradient = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4);

  if (check_numpy_available() && PyArray_Check(data)) {
    return eager_api_numpy_to_tensor(data, dtype, place_id, device_id,
                                     stop_gradient);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Eater to_tensor only support numpy to tensor."));
    Py_INCREF(Py_None);
    return Py_None;
  }
}

static struct PyGetSetDef variable_properties[] = {
    {"shape", (getter)eager_tensor_properties_get_shape, nullptr, nullptr,
     nullptr},
    {"stop_gradient", (getter)eager_tensor_properties_get_stop_gradient,
     (setter)eager_tensor_properties_set_stop_gradient, nullptr, nullptr},
    {"dtype", (getter)eager_tensor_properties_get_dtype, nullptr, nullptr,
     nullptr},
    {"_place_str", (getter)eager_tensor_properties_get_place_str, nullptr,
     nullptr, nullptr},
    {nullptr}};

PyMethodDef variable_methods[] = {
    {"numpy", (PyCFunction)(void (*)(void))eager_tensor_method_numpy,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_is_initialized",
     (PyCFunction)(void (*)(void))eager_tensor_method_is_initialized,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

PyMethodDef variable_functions[] = {
    {"to_tensor", (PyCFunction)(void (*)(void))eager_api_to_tensor,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"scale", (PyCFunction)(void (*)(void))eager_api_scale,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

static PyTypeObject EagerTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0) "core_avx.eager.EagerTensor", /* tp_name */
    sizeof(EagerTensorObject),       /* tp_basicsize */
    0,                               /* tp_itemsize */
    (destructor)eagertensor_dealloc, /* tp_dealloc */
    0,                               /* tp_vectorcall_offset */
    0,                               /* tp_getattr */
    0,                               /* tp_setattr */
    0,                               /* tp_reserved */
    0,                               /* tp_repr */
    0,                               /* tp_as_number */
    0,                               /* tp_as_sequence */
    0,                               /* tp_as_mapping */
    0,                               /* tp_hash  */
    0,                               /* tp_call */
    eagertensor_str,                 /* tp_str */
    0,                               /* tp_getattro */
    0,                               /* tp_setattro */
    0,                               /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HEAPTYPE,    /* tp_flags */
    0,                          /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    variable_methods,           /* tp_methods */
    0,                          /* tp_members */
    variable_properties,        /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)eagertensor_init, /* tp_init */
    0,                          /* tp_alloc */
    eagertensor_new,            /* tp_new */
};

void BindEager(pybind11::module* module) {
  auto m = module->def_submodule("eager");

  pEagerTensorType = &EagerTensorType;
  if (PyType_Ready(&EagerTensorType) < 0) {
    return;
  }

  Py_INCREF(&EagerTensorType);
  if (PyModule_AddObject(m.ptr(), "EagerTensor",
                         reinterpret_cast<PyObject*>(&EagerTensorType)) < 0) {
    Py_DECREF(&EagerTensorType);
    Py_DECREF(m.ptr());
    return;
  }

  if (PyModule_AddFunctions(m.ptr(), variable_functions) < 0) {
    return;
  }
}

}  // namespace pybind
}  // namespace paddle
