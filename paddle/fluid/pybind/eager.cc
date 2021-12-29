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

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
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
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/pybind/eager_op_function_impl.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/api/lib/utils/tensor_utils.h"
namespace paddle {
namespace pybind {

namespace py = ::pybind11;

PyTypeObject* p_eager_tensor_type;
extern PyTypeObject* g_vartype_pytype;
extern PyTypeObject* g_framework_tensor_pytype;

PyObject* EagerTensorNew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = reinterpret_cast<EagerTensorObject*>(obj);
    new (&(v->eager_tensor)) egr::EagerTensor();
  }
  return obj;
}

// TODO(jiabin): Overload this once we need more constructor in Python
void EmptyEagerTensorInitializer(
    EagerTensorObject* self, const std::string& name,
    const paddle::platform::Place& place, bool persistable = false,
    bool stop_gradient = true, framework::proto::VarType::Type dtype =
                                   paddle::framework::proto::VarType::FP32,
    const std::vector<int>& dims = {},
    framework::proto::VarType::Type var_type =
        paddle::framework::proto::VarType::LOD_TENSOR) {
  self->eager_tensor.set_name(name);
  auto autograd_meta = egr::EagerUtils::autograd_meta(&(self->eager_tensor));
  autograd_meta->SetPersistable(persistable);
  autograd_meta->SetStopGradient(stop_gradient);
  if (var_type == paddle::framework::proto::VarType::LOD_TENSOR) {
    // TODO(jiabin): Maybe support LOD later
    std::shared_ptr<pten::DenseTensor> dense_tensor =
        std::make_shared<pten::DenseTensor>(
            pten::make_intrusive<paddle::experimental::SharedStorage>(place),
            pten::DenseTensorMeta(pten::TransToPtenDataType(dtype),
                                  paddle::framework::make_ddim(dims)));
    self->eager_tensor.set_impl(dense_tensor);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "We only support LoDTensor to be constructed by this initializer, "
        "please check your var type first and make sure you are going to "
        "construct LoDTensor."));
  }

  if (!autograd_meta->GetMutableGradNode()) {
    VLOG(3) << "Tensor(" << name
            << ") have not GradNode, add GradNodeAccumulation for it.";
    autograd_meta->SetGradNode(std::make_shared<egr::GradNodeAccumulation>());
  }
}

void InitEagerTensorWithNumpyValue(EagerTensorObject* self,
                                   const py::object& array,
                                   bool zero_copy = false) {
  PADDLE_ENFORCE_EQ(
      self->eager_tensor.defined(), true,
      paddle::platform::errors::Fatal(
          "Calling InitEagerTensorWithNumpyValue of Eager Tensor without "
          "EmptyEagerTensorInitializer is "
          "forbidden. Please check your code and make sure you new a "
          "eager tensor before init it with NumPy."));
  pten::DenseTensor* impl_ptr =
      static_cast<pten::DenseTensor*>(self->eager_tensor.impl().get());
  paddle::platform::Place place = impl_ptr->place();
  paddle::framework::LoDTensor temp_tensor = paddle::framework::LoDTensor();
  if (platform::is_cpu_place(place)) {
    SetTensorFromPyArray<platform::CPUPlace>(
        &temp_tensor, array, BOOST_GET_CONST(platform::CPUPlace, place),
        zero_copy);
  } else if (platform::is_xpu_place(place)) {
    SetTensorFromPyArray<platform::XPUPlace>(
        &temp_tensor, array, BOOST_GET_CONST(platform::XPUPlace, place),
        zero_copy);
  } else if (platform::is_gpu_place(place)) {
    SetTensorFromPyArray<platform::CUDAPlace>(
        &temp_tensor, array, BOOST_GET_CONST(platform::CUDAPlace, place),
        zero_copy);
  } else if (platform::is_cuda_pinned_place(place)) {
    SetTensorFromPyArray<platform::CUDAPinnedPlace>(
        &temp_tensor, array, BOOST_GET_CONST(platform::CUDAPinnedPlace, place),
        zero_copy);
  } else if (platform::is_npu_place(place)) {
    SetTensorFromPyArray<platform::NPUPlace>(
        &temp_tensor, array, BOOST_GET_CONST(platform::NPUPlace, place),
        zero_copy);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place should be one of "
        "CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace/NPUPlace"));
  }
  paddle::experimental::ReMakePtenDenseTensor(temp_tensor, impl_ptr);
}

void InitEagerTensorWithEagerTensor(EagerTensorObject* self,
                                    const egr::EagerTensor& src,
                                    const paddle::platform::Place& place,
                                    const std::string& name) {
  self->eager_tensor.set_name(name);
  if (place == src.place()) {
    auto impl = std::static_pointer_cast<pten::DenseTensor>(src.impl());
    self->eager_tensor.set_impl(impl);
    VLOG(4) << "Same place, do ShareDataWith";
  } else {
    self->eager_tensor.set_impl(
        src.copy_to(pten::TransToPtenBackend(place), true).impl());
    VLOG(4) << "Different place, do TensorCopy";
  }
  egr::EagerUtils::autograd_meta(&(self->eager_tensor))->SetStopGradient(true);
  if (src.get_autograd_meta()) {
    egr::EagerUtils::unsafe_autograd_meta(self->eager_tensor)
        ->SetPersistable(
            egr::EagerUtils::unsafe_autograd_meta(src)->Persistable());
  } else {
    egr::EagerUtils::unsafe_autograd_meta(self->eager_tensor)
        ->SetPersistable(false);
  }
}

void InitEagerTensorWithFrameworkTensor(EagerTensorObject* self,
                                        const framework::Tensor& src,
                                        const paddle::platform::Place& place,
                                        const std::string& name) {
  self->eager_tensor.set_name(name);
  if (place == src.place()) {
    std::shared_ptr<pten::DenseTensor> dense_tensor =
        std::make_shared<pten::DenseTensor>(
            pten::make_intrusive<paddle::experimental::SharedStorage>(place),
            pten::DenseTensorMeta(pten::TransToPtenDataType(src.type()),
                                  src.dims()));
    paddle::experimental::ReMakePtenDenseTensor(src, dense_tensor.get());
    self->eager_tensor.set_impl(dense_tensor);
    VLOG(4) << "Same place, do ShareDataWith";
  } else {
    std::shared_ptr<pten::DenseTensor> dense_tensor =
        std::make_shared<pten::DenseTensor>(
            pten::make_intrusive<paddle::experimental::SharedStorage>(
                src.place()),
            pten::DenseTensorMeta(pten::TransToPtenDataType(src.type()),
                                  src.dims()));
    paddle::experimental::ReMakePtenDenseTensor(src, dense_tensor.get());
    auto temp = egr::EagerTensor(dense_tensor);
    self->eager_tensor.set_impl(
        temp.copy_to(pten::TransToPtenBackend(place), true).impl());
    VLOG(4) << "Different place, do TensorCopy";
  }
  egr::EagerUtils::autograd_meta(&(self->eager_tensor))->SetStopGradient(true);
  egr::EagerUtils::unsafe_autograd_meta(self->eager_tensor)
      ->SetPersistable(false);
}
// TODO(jiabin): We have to do some ugly work, refactor this method using
// PyArg_ParseTuple()，PyArg_ParseTupleAndKeywords() and PyArg_Parse() later to
// support kwargs.
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
   * 3. (multi-place) (must have first 2 parameter)
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
   * 6. (multi-place) (must have first 2 parameter)
   * def __init__ (
   * ** tensor: EagerTensor,
   * ** place: paddle::platform::Place,
   * ** name: std::string)
   * 7. (multi-place) (must have first 2 parameter)
   * def __init__ (
   * ** tensor: FrameworkTensor,
   * ** place: paddle::platform::Place,
   * ** name: std::string)
   *  **/
  PADDLE_ENFORCE_NOT_NULL(
      self, paddle::platform::errors::Fatal(
                "Calling __init__ of Eager Tensor without __new__ is "
                "forbidden. Please check your code and make sure you new a "
                "eager tensor before init it."));

  auto py_tensor_ptr = reinterpret_cast<EagerTensorObject*>(self);

  // TODO(jiabin): Only support case 2 for now
  Py_ssize_t args_num = PyTuple_Size(args);
  switch (args_num) {
    case (Py_ssize_t)0: {
      // case 1
      VLOG(6) << "Calling case1's initializer.";
      EmptyEagerTensorInitializer(
          py_tensor_ptr,
          egr::Controller::Instance().GenerateUniqueName("generated_tensor"),
          egr::Controller::Instance().GetExpectedPlace());
      return 0;
    }
    case (Py_ssize_t)1: {
      // case 4, 5
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
        VLOG(6) << "Calling case4's initializer.";
        PADDLE_ENFORCE_EQ(
            pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr), true,
            paddle::platform::errors::Fatal(
                "We expected initial parametes list like: \n **value: ndarray. "
                "But got value with wrong type: %s",
                reinterpret_cast<PyTypeObject*>(arg0_ptr->ob_type)->tp_name));
        py::object numpy_value = py::object(py::handle(arg0_ptr), true);
        EmptyEagerTensorInitializer(
            py_tensor_ptr,
            egr::Controller::Instance().GenerateUniqueName("generated_tensor"),
            egr::Controller::Instance().GetExpectedPlace());
        InitEagerTensorWithNumpyValue(py_tensor_ptr, numpy_value,
                                      /** zero copy **/ false);
        return 0;
      } else if (PyObject_IsInstance(arg0_ptr, reinterpret_cast<PyObject*>(
                                                   p_eager_tensor_type))) {
        VLOG(6) << "Calling case5's initializer.";
        auto src_tensor = CastPyArg2EagerTensor(arg0_ptr, 0);
        InitEagerTensorWithEagerTensor(
            py_tensor_ptr, src_tensor,
            egr::Controller::Instance().GetExpectedPlace(),
            egr::Controller::Instance().GenerateUniqueName("generated_tensor"));
        return 0;
      } else if (PyObject_IsInstance(
                     arg0_ptr,
                     reinterpret_cast<PyObject*>(g_framework_tensor_pytype))) {
        VLOG(6) << "Calling case7's initializer.";
        auto src_tensor = CastPyArg2FrameworkTensor(arg0_ptr, 0);
        InitEagerTensorWithFrameworkTensor(
            py_tensor_ptr, src_tensor, src_tensor.place(),
            egr::Controller::Instance().GenerateUniqueName("generated_tensor"));
        return 0;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "We only support construct tensor from numpy value or tensor with "
            "python args by this initializer, "
            "please check your input first and make sure you are on the right "
            "way."));
      }
      return 0;
    }
    case (Py_ssize_t)2: {
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
        VLOG(6) << "Calling case3's initializer.";
        PADDLE_ENFORCE_EQ(
            pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr), true,
            paddle::platform::errors::Fatal(
                "We expected initial parametes list like: \n **value: ndarray. "
                "But got value with wrong type: %s",
                reinterpret_cast<PyTypeObject*>(arg0_ptr->ob_type)->tp_name));
        py::object numpy_value = py::object(py::handle(arg0_ptr), true);
        paddle::platform::Place place =
            CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
        EmptyEagerTensorInitializer(
            py_tensor_ptr,
            egr::Controller::Instance().GenerateUniqueName("generated_tensor"),
            place);
        InitEagerTensorWithNumpyValue(py_tensor_ptr, numpy_value,
                                      /** zero copy **/ false);
        return 0;
      } else if (PyObject_IsInstance(arg0_ptr, reinterpret_cast<PyObject*>(
                                                   p_eager_tensor_type))) {
        VLOG(6) << "Calling case6's initializer.";
        auto src_tensor = CastPyArg2EagerTensor(arg0_ptr, 0);
        paddle::platform::Place place =
            CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
        InitEagerTensorWithEagerTensor(
            py_tensor_ptr, src_tensor, place,
            egr::Controller::Instance().GenerateUniqueName("generated_tensor"));
        return 0;
      } else if (PyObject_IsInstance(
                     arg0_ptr,
                     reinterpret_cast<PyObject*>(g_framework_tensor_pytype))) {
        VLOG(6) << "Calling case7's initializer.";
        auto src_tensor = CastPyArg2FrameworkTensor(arg0_ptr, 0);
        paddle::platform::Place place =
            CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
        InitEagerTensorWithFrameworkTensor(
            py_tensor_ptr, src_tensor, place,
            egr::Controller::Instance().GenerateUniqueName("generated_tensor"));
        return 0;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "We only support construct tensor from numpy value or tensor with "
            "python args by this initializer, "
            "please check your input first and make sure you are on the right "
            "way."));
      }
      return 0;
    }
    case (Py_ssize_t)3: {
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
        VLOG(6) << "Calling case3's initializer.";
        PADDLE_ENFORCE_EQ(
            pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr), true,
            paddle::platform::errors::Fatal(
                "We expected initial parametes list like: \n **value: ndarray. "
                "But got value with wrong type: %s",
                reinterpret_cast<PyTypeObject*>(arg0_ptr->ob_type)->tp_name));
        py::object numpy_value = py::object(py::handle(arg0_ptr), true);
        paddle::platform::Place place =
            CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
        bool persistable = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2);
        EmptyEagerTensorInitializer(
            py_tensor_ptr,
            egr::Controller::Instance().GenerateUniqueName("generated_tensor"),
            place, persistable);
        InitEagerTensorWithNumpyValue(py_tensor_ptr, numpy_value,
                                      /** zero copy **/ false);
        return 0;
      } else if (PyObject_IsInstance(arg0_ptr, reinterpret_cast<PyObject*>(
                                                   p_eager_tensor_type))) {
        VLOG(6) << "Calling case6's initializer.";
        auto src_tensor = CastPyArg2EagerTensor(arg0_ptr, 0);
        paddle::platform::Place place =
            CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
        std::string act_name = "";
        PyObject* name_obj = PyTuple_GET_ITEM(args, 2);
        if (name_obj == Py_None) {
          act_name = egr::Controller::Instance().GenerateUniqueName(
              "generated_tensor");
        } else {
          act_name = CastPyArg2AttrString(name_obj, 2);
        }
        InitEagerTensorWithEagerTensor(py_tensor_ptr, src_tensor, place,
                                       act_name);
        return 0;
      } else if (PyObject_IsInstance(
                     arg0_ptr,
                     reinterpret_cast<PyObject*>(g_framework_tensor_pytype))) {
        VLOG(6) << "Calling case7's initializer.";
        auto src_tensor = CastPyArg2FrameworkTensor(arg0_ptr, 0);
        paddle::platform::Place place =
            CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
        std::string act_name = "";
        PyObject* name_obj = PyTuple_GET_ITEM(args, 2);
        if (name_obj == Py_None) {
          act_name = egr::Controller::Instance().GenerateUniqueName(
              "generated_tensor");
        } else {
          act_name = CastPyArg2AttrString(name_obj, 2);
        }
        InitEagerTensorWithFrameworkTensor(py_tensor_ptr, src_tensor, place,
                                           act_name);
        return 0;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "We only support construct tensor from numpy value or tensor with "
            "python args by this initializer, "
            "please check your input first and make sure you are on the right "
            "way."));
      }
      return 0;
    }
    case (Py_ssize_t)4: {
      VLOG(6) << "Calling case3's initializer.";
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      PADDLE_ENFORCE_EQ(
          pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr), true,
          paddle::platform::errors::Fatal(
              "We expected initial parametes list like: \n **value: ndarray, "
              "\n ** place: paddle::platform::Place, \n ** persistable: bool, "
              "\n ** zero_copy: bool, \n ** name: std::string, \n ** "
              "stop_gradient: bool. But got value with wrong type: %s",
              reinterpret_cast<PyTypeObject*>(arg0_ptr->ob_type)->tp_name));
      py::object numpy_value =
          py::object(py::handle(PyTuple_GET_ITEM(args, 0)), true);
      paddle::platform::Place place =
          CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
      bool persistable = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2);
      bool zero_copy = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);
      EmptyEagerTensorInitializer(
          py_tensor_ptr,
          egr::Controller::Instance().GenerateUniqueName("generated_tensor"),
          place, persistable);
      InitEagerTensorWithNumpyValue(py_tensor_ptr, numpy_value, zero_copy);
      return 0;
    }
    case (Py_ssize_t)5: {
      // case 2
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      if (PyObject_IsInstance(arg0_ptr,
                              reinterpret_cast<PyObject*>(g_vartype_pytype))) {
        VLOG(6) << "Calling case2's initializer.";
        paddle::framework::proto::VarType::Type dtype =
            CastPyArg2ProtoType(PyTuple_GET_ITEM(args, 0), 0);
        std::vector<int> dims =

            CastPyArg2VectorOfInt(PyTuple_GET_ITEM(args, 1), 1);
        std::string act_name = "";
        PyObject* name_obj = PyTuple_GET_ITEM(args, 2);
        if (name_obj == Py_None) {
          act_name = egr::Controller::Instance().GenerateUniqueName(
              "generated_tensor");
        } else {
          act_name = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 2), 2);
        }
        paddle::framework::proto::VarType::Type var_type =
            CastPyArg2ProtoType(PyTuple_GET_ITEM(args, 3), 3);
        bool persistable = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4);
        EmptyEagerTensorInitializer(
            py_tensor_ptr, act_name,
            egr::Controller::Instance().GetExpectedPlace(), persistable, true,
            dtype, dims, var_type);
        return 0;
      } else if (PyObject_IsInstance(arg0_ptr, reinterpret_cast<PyObject*>(
                                                   p_eager_tensor_type))) {
        PADDLE_ENFORCE_EQ(
            pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr), true,
            paddle::platform::errors::Fatal(
                "We expected initial parametes list like: \n **value: ndarray, "
                "\n ** place: paddle::platform::Place, \n ** persistable: "
                "bool, \n ** zero_copy: bool, \n ** name: std::string, \n ** "
                "stop_gradient: bool. But got value with wrong type: %s",
                reinterpret_cast<PyTypeObject*>(arg0_ptr->ob_type)->tp_name));
        py::object numpy_value =
            py::object(py::handle(PyTuple_GET_ITEM(args, 0)), true);
        paddle::platform::Place place =
            CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
        bool persistable = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2);
        bool zero_copy = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);
        std::string act_name = "";
        PyObject* name_obj = PyTuple_GET_ITEM(args, 4);
        if (name_obj == Py_None) {
          act_name = egr::Controller::Instance().GenerateUniqueName(
              "generated_tensor");
        } else {
          act_name = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 4), 4);
        }
        EmptyEagerTensorInitializer(py_tensor_ptr, act_name, place,
                                    persistable);
        InitEagerTensorWithNumpyValue(py_tensor_ptr, numpy_value, zero_copy);
        return 0;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "We only support construct tensor from numpy value or dtype with "
            "python args by this initializer, "
            "please check your input first and make sure you are on the right "
            "way."));
      }
      return 0;
    }
    case (Py_ssize_t)6: {
      // case 3
      VLOG(6) << "Calling case3's initializer.";
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      PADDLE_ENFORCE_EQ(
          pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr), true,
          paddle::platform::errors::Fatal(
              "We expected initial parametes list like: \n **value: ndarray, "
              "\n ** place: paddle::platform::Place, \n ** persistable: bool, "
              "\n ** zero_copy: bool, \n ** name: std::string, \n ** "
              "stop_gradient: bool. But got value with wrong type: %s",
              reinterpret_cast<PyTypeObject*>(arg0_ptr->ob_type)->tp_name));
      py::object numpy_value =
          py::object(py::handle(PyTuple_GET_ITEM(args, 0)), true);
      paddle::platform::Place place =
          CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
      bool persistable = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2);
      bool zero_copy = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);
      std::string act_name = "";
      PyObject* name_obj = PyTuple_GET_ITEM(args, 4);
      if (name_obj == Py_None) {
        act_name =
            egr::Controller::Instance().GenerateUniqueName("generated_tensor");
      } else {
        act_name = CastPyArg2AttrString(name_obj, 4);
      }
      bool stop_gradient = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 5), 5);
      EmptyEagerTensorInitializer(py_tensor_ptr, act_name, place, persistable,
                                  stop_gradient);
      InitEagerTensorWithNumpyValue(py_tensor_ptr, numpy_value, zero_copy);
      return 0;
    }
    default: {
      PADDLE_THROW(platform::errors::Fatal(
          "Can't not find expected num of args, please check your call, and "
          "make sure u call the existed constructor."));
      return 1;
    }
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
