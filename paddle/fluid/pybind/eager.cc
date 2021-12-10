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

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/include/core.h"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "paddle/fluid/pybind/eager_op_function_impl.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

PyTypeObject* p_eager_tensor_type;

PyObject* EagerTensorNew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = reinterpret_cast<EagerTensorObject*>(obj);
    new (&(v->eager_tensor)) egr::EagerTensor();
  }
  return obj;
}

// TODO(jiabin): Overload this once we need more constructor in Python
void EagerTensorInitializer(EagerTensorObject* self,
                            framework::proto::VarType::Type dtype,
                            const std::vector<int>& dims,
                            const std::string& name,
                            framework::proto::VarType::Type var_type,
                            bool persistable) {
  self->eager_tensor.set_name(name);
  egr::EagerUtils::autograd_meta(&(self->eager_tensor))
      ->SetPersistable(persistable);
  if (var_type == paddle::framework::proto::VarType::LOD_TENSOR) {
    // TODO(jiabin): Maybe support LOD later
    std::shared_ptr<pten::DenseTensor> dense_tensor =
        std::make_shared<pten::DenseTensor>();
    dense_tensor->set_meta(pten::DenseTensorMeta(
        pten::TransToPtenDataType(dtype), paddle::framework::make_ddim(dims)));
    self->eager_tensor.set_impl(dense_tensor);
  }
}

// We have to do some ugly work, since python c api doesn't support
int EagerTensorInit(PyObject* self, PyObject* args, PyObject* kwds) {
  /** We should have init function with signature:
   * 1.
   * def __init__ ()
   * 2.
   * def __init__ (
   * ** dtype: paddle::framework::proto::VarType::Type,
   * ** dims: vector<int>,
   * ** name: std::string,
   * ** type: paddle::framework::proto::VarType::Type,
   * ** persistable: bool)
   * 3. (multi-place)
   * def __init__ (
   * ** value: ndarray,
   * ** place: paddle::platform::Place,
   * ** persistable: bool,
   * ** zero_copy: bool,
   * ** name: std::string,
   * ** stop_gradient: bool)
   * 4.
   * def __init__ (
   * ** value: ndarray)
   * 5.
   * def __init__ (
   * ** tensor: EagerTensor)
   * 6. (multi-place)
   * def __init__ (
   * ** tensor: EagerTensor,
   * ** place: paddle::platform::Place)
   *  **/
  PADDLE_ENFORCE_NOT_NULL(
      self, paddle::platform::errors::Fatal(
                "Calling __init__ of Eager Tensor without __new__ is "
                "forbidden. Please check your code and make sure you new a "
                "eager tensor before init it."));

  auto py_tensor_ptr = reinterpret_cast<EagerTensorObject*>(self);

  // TODO(jiabin): Only support case 2 for now
  if (PyTuple_Size(args) == (Py_ssize_t)5) {
    paddle::framework::proto::VarType::Type dtype =
        CastPyArg2ProtoType(PyTuple_GET_ITEM(args, 0), 0);
    std::vector<int> dims = CastPyArg2VectorOfInt(PyTuple_GET_ITEM(args, 1), 1);
    std::string act_name = "";
    PyObject* name_obj = PyTuple_GET_ITEM(args, 2);
    if (name_obj == Py_None) {
      act_name =
          egr::Controller::Instance().GenerateUniqueName("generated_tensor");
    } else {
      act_name = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 2), 2);
    }
    paddle::framework::proto::VarType::Type var_type =
        CastPyArg2ProtoType(PyTuple_GET_ITEM(args, 3), 3);
    bool persistable = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4);
    EagerTensorInitializer(py_tensor_ptr, dtype, dims, act_name, var_type,
                           persistable);
    return 0;
  } else {
    return 1;
  }
}

static void eagertensor_dealloc(EagerTensorObject* self) {
  self->eager_tensor.~EagerTensor();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

extern struct PyGetSetDef variable_properties[];

extern PyMethodDef variable_methods[];

PyTypeObject eager_tensor_type = {
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
    0,                               /* tp_str */
    0,                               /* tp_getattro */
    0,                               /* tp_setattro */
    0,                               /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HEAPTYPE, /* tp_flags */
    0,                       /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    0,                       /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    variable_methods,        /* tp_methods */
    0,                       /* tp_members */
    variable_properties,     /* tp_getset */
    0,                       /* tp_base */
    0,                       /* tp_dict */
    0,                       /* tp_descr_get */
    0,                       /* tp_descr_set */
    0,                       /* tp_dictoffset */
    EagerTensorInit,         /* tp_init */
    0,                       /* tp_alloc */
    EagerTensorNew,          /* tp_new */
    0,                       /* tp_free */
    0,                       /* tp_is_gc */
    0,                       /* tp_bases */
    0,                       /* tp_mro */
    0,                       /* tp_cache */
    0,                       /* tp_subclasses */
    0,                       /* tp_weaklist */
    0,                       /* tp_del */
    0,                       /* tp_version_tag */
    0                        /* tp_finalize */
};

void BindEager(pybind11::module* module) {
  auto m = module->def_submodule("eager");

  p_eager_tensor_type = &eager_tensor_type;
  if (PyType_Ready(&eager_tensor_type) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle erroe in BindEager(PyType_Ready)."));
    return;
  }

  Py_INCREF(&eager_tensor_type);
  if (PyModule_AddObject(m.ptr(), "EagerTensor",
                         reinterpret_cast<PyObject*>(&eager_tensor_type)) < 0) {
    Py_DECREF(&eager_tensor_type);
    Py_DECREF(m.ptr());
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle erroe in BindEager(PyModule_AddObject)."));
    return;
  }

  BindFunctions(m.ptr());
  BindEagerOpFunctions(&m);
}

}  // namespace pybind
}  // namespace paddle
