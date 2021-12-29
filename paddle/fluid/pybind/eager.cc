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

// initialize EagerTensor by EagerTensor or framework::Tensor (mix args and
// kwargs) automatically
void AutoInitEagerTensorByTensor(
    EagerTensorObject* py_tensor_ptr,
    std::unordered_map<std::string, PyObject*> kws_map, PyObject* args,
    bool flag_kwargs, Py_ssize_t args_num, bool init_by_egr_tensor = true) {
  // key_word lists: [value, place, name]
  std::unordered_map<std::string, Py_ssize_t> kw_order_map{
      {"value", 1}, {"place", 2}, {"name", 3}};

  paddle::platform::Place place =
      egr::Controller::Instance().GetExpectedPlace();
  std::string act_name = "";

  if (kw_order_map["place"] <= args_num) {
    place = CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
  } else {
    if (flag_kwargs && kws_map["place"] != NULL) {
      place = CastPyArg2Place(kws_map["place"], 0);
    }
  }

  if (kw_order_map["name"] <= args_num) {
    act_name = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 2), 2);
  } else {
    if (flag_kwargs) {
      if (kws_map["name"] == NULL) {
        act_name =
            egr::Controller::Instance().GenerateUniqueName("generated_tensor");
      } else {
        act_name = CastPyArg2AttrString(kws_map["name"], 0);
      }
    } else {
      act_name =
          egr::Controller::Instance().GenerateUniqueName("generated_tensor");
    }
  }

  if (init_by_egr_tensor) {
    egr::EagerTensor src_tensor;
    if (kw_order_map["value"] <= args_num) {
      src_tensor = CastPyArg2EagerTensor(PyTuple_GET_ITEM(args, 0), 0);
    } else {
      if (flag_kwargs && kws_map["value"] != NULL) {
        src_tensor = CastPyArg2EagerTensor(kws_map["value"], 0);
      }
    }
    InitEagerTensorWithEagerTensor(py_tensor_ptr, src_tensor, place, act_name);
  } else {
    // init by framework tensor
    framework::Tensor src_tensor;
    if (kw_order_map["value"] <= args_num) {
      src_tensor = CastPyArg2FrameworkTensor(PyTuple_GET_ITEM(args, 0), 0);
    } else {
      if (flag_kwargs && kws_map["value"] != NULL) {
        src_tensor = CastPyArg2FrameworkTensor(kws_map["value"], 0);
      }
    }
    InitEagerTensorWithFrameworkTensor(py_tensor_ptr, src_tensor, place,
                                       act_name);
  }
}

// initialize EagerTensor by PyArray (mix args and kwargs) automatically
void AutoInitEagerTensorByPyArray(
    EagerTensorObject* py_tensor_ptr,
    std::unordered_map<std::string, PyObject*> kws_map, PyObject* args,
    bool flag_kwargs, Py_ssize_t args_num) {
  // key_word lists: [value, place, persistable, zero_copy, name, stop_gradient]
  std::unordered_map<std::string, Py_ssize_t> kw_order_map{
      {"value", 1},     {"place", 2}, {"persistable", 3},
      {"zero_copy", 4}, {"name", 5},  {"stop_gradient", 6}};

  py::object numpy_value = py::object();
  paddle::platform::Place place =
      egr::Controller::Instance().GetExpectedPlace();
  bool persistable = false;
  bool zero_copy = false;
  std::string act_name = "";
  bool stop_gradient = true;

  if (kw_order_map["value"] <= args_num) {
    numpy_value = py::object(py::handle(PyTuple_GET_ITEM(args, 0)), true);
  } else {
    if (flag_kwargs && kws_map["value"] != NULL) {
      numpy_value = py::object(py::handle(kws_map["value"]), true);
    }
  }

  if (kw_order_map["place"] <= args_num) {
    place = CastPyArg2Place(PyTuple_GET_ITEM(args, 1), 1);
  } else {
    if (flag_kwargs && kws_map["place"] != NULL) {
      place = CastPyArg2Place(kws_map["place"], 0);
    }
  }

  if (kw_order_map["persistable"] <= args_num) {
    persistable = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2);
  } else {
    if (flag_kwargs && kws_map["persistable"] != NULL) {
      persistable = CastPyArg2AttrBoolean(kws_map["persistable"], 0);
    }
  }

  if (kw_order_map["zero_copy"] <= args_num) {
    zero_copy = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);
  } else {
    if (flag_kwargs && kws_map["zero_copy"] != NULL) {
      zero_copy = CastPyArg2AttrBoolean(kws_map["zero_copy"], 0);
    }
  }

  if (kw_order_map["name"] <= args_num) {
    act_name = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 4), 4);
  } else {
    if (flag_kwargs) {
      if (kws_map["name"] == NULL) {
        act_name =
            egr::Controller::Instance().GenerateUniqueName("generated_tensor");
      } else {
        act_name = CastPyArg2AttrString(kws_map["name"], 0);
      }
    } else {
      act_name =
          egr::Controller::Instance().GenerateUniqueName("generated_tensor");
    }
  }

  if (kw_order_map["stop_gradient"] <= args_num) {
    stop_gradient = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 5), 5);

  } else {
    if (flag_kwargs) {
      if (kws_map["stop_gradient"] != NULL) {
        stop_gradient = CastPyArg2AttrBoolean(kws_map["stop_gradient"], 0);
      }
    }
  }
  EmptyEagerTensorInitializer(py_tensor_ptr, act_name, place, persistable,
                              stop_gradient);
  InitEagerTensorWithNumpyValue(py_tensor_ptr, numpy_value, zero_copy);
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

/** We should have init function with signature:
   * 1.
   * def __init__ ()
   * 2.
   * def __init__ (
   * ** dtype: paddle::framework::proto::VarType::Type,
   * ** dims: vector<int>,
   * ** name: std::string,
   * ** type: paddle::framework::proto::VarType::LodTensor,
   * ** persistable: bool)
   * 3. (multi-place)
   * (should have at least one parameter, one parameter equals to case 4, zero
   * parameter equals to case 1)
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
   * (should have at least one parameter, one parameter equals to case 5, zero
   * parameter equals to case 1.)
   * def __init__ (
   * ** tensor: EagerTensor,
   * ** place: paddle::platform::Place,
   * ** name: std::string)
   * 7. (multi-place) (should have at least one parameter, one parameter similar
   * to case 5, zero parameter equals to case 1.)
   * def __init__ (
   * ** tensor: FrameworkTensor,
   * ** place: paddle::platform::Place,
   * ** name: std::string)
   *  **/
int EagerTensorInit(PyObject* self, PyObject* args, PyObject* kwargs) {
  // set a flag to record use kwargs or not
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;

  // all kwargs
  PyObject* kw_zero_copy = NULL;
  PyObject* kw_persistable = NULL;
  PyObject* kw_stop_gradient = NULL;

  PyObject* kw_value = NULL;  // receive PyArray or EagerTensor
  PyObject* kw_place = NULL;
  PyObject* kw_name = NULL;
  PyObject* kw_dims = NULL;
  PyObject* kw_dtype = NULL;
  PyObject* kw_type = NULL;

  // the keywords argument
  static char* kwlist[] = {
      const_cast<char*>("value"),       const_cast<char*>("place"),
      const_cast<char*>("persistable"), const_cast<char*>("zero_copy"),
      const_cast<char*>("name"),        const_cast<char*>("stop_gradient"),
      const_cast<char*>("dims"),        const_cast<char*>("dtype"),
      const_cast<char*>("vtype"),       NULL};

  // 'O' Store a Python object (without any conversion) in a C object pointer,
  // '|' Indicates that the remaining arguments in the Python argument list are
  // optional.
  // PyArg_ParseTupleAndKeywords can Parse the parameters of a function that
  // takes both positional and keyword parameters into local variables,
  // which enhance case2, case3, case4, case5, case6.
  bool flag_ = PyArg_ParseTupleAndKeywords(
      args, kwargs, "|OOOOOOOOO", kwlist, &kw_value, &kw_place, &kw_persistable,
      &kw_zero_copy, &kw_name, &kw_stop_gradient, &kw_dims, &kw_dtype,
      &kw_type);

  // helper map
  std::unordered_map<std::string, PyObject*> kws_map{
      {"value", kw_value},
      {"place", kw_place},
      {"persistable", kw_persistable},
      {"zero_copy", kw_zero_copy},
      {"name", kw_name},
      {"stop_gradient", kw_stop_gradient},
      {"dims", kw_dims},
      {"dtype", kw_dtype},
      {"type", kw_type}};

  PADDLE_ENFORCE_EQ(flag_, true, paddle::platform::errors::PreconditionNotMet(
                                     "Please check your input first and make "
                                     "sure you are on the right way."));

  PADDLE_ENFORCE_NOT_NULL(
      self, paddle::platform::errors::Fatal(
                "Calling __init__ of Eager Tensor without __new__ is "
                "forbidden. Please check your code and make sure you new a "
                "eager tensor before init it."));

  auto py_tensor_ptr = reinterpret_cast<EagerTensorObject*>(self);

  Py_ssize_t args_num = PyTuple_Size(args);
  VLOG(6) << " args_num: " << args_num;
  switch (args_num) {
    case (Py_ssize_t)0: {
      if (!flag_kwargs) {
        // case 1
        VLOG(6) << "Calling case1's initializer.";
        EmptyEagerTensorInitializer(
            py_tensor_ptr,
            egr::Controller::Instance().GenerateUniqueName("generated_tensor"),
            egr::Controller::Instance().GetExpectedPlace());
        return 0;
      } else {  // no position args, all arguments are kwargs
        if (kw_value != NULL) {
          if (pybind11::detail::npy_api::get().PyArray_Check_(kw_value)) {
            VLOG(6) << "Calling case3's or case4's initializer";
            AutoInitEagerTensorByPyArray(py_tensor_ptr, kws_map, args,
                                         flag_kwargs, args_num);
            return 0;
          } else if (PyObject_IsInstance(kw_value, reinterpret_cast<PyObject*>(
                                                       p_eager_tensor_type))) {
            VLOG(6) << "Calling case5's or case6's initializer";
            AutoInitEagerTensorByTensor(py_tensor_ptr, kws_map, args,
                                        flag_kwargs, args_num);
            return 0;
          } else if (PyObject_IsInstance(kw_value,
                                         reinterpret_cast<PyObject*>(
                                             g_framework_tensor_pytype))) {
            VLOG(6) << "Calling case7's initializer.";
            AutoInitEagerTensorByTensor(
                py_tensor_ptr, kws_map, args, flag_kwargs, args_num,
                /* false means not init by egr tensor*/ false);
            return 0;
          }
        } else if (kw_dtype != NULL &&
                   PyObject_IsInstance(kw_dtype, reinterpret_cast<PyObject*>(
                                                     g_vartype_pytype))) {
          VLOG(6) << "Calling case2's initializer";

          PADDLE_ENFORCE_NOT_NULL(
              kw_dims,
              paddle::platform::errors::Fatal(
                  "Calling __init__ of Eager Tensor with NULL dims is "
                  "forbidden. Please check your code and make sure you new a "
                  "dims before calling this constructor."));

          PADDLE_ENFORCE_NOT_NULL(
              kw_name,
              paddle::platform::errors::Fatal(
                  "Calling __init__ of Eager Tensor with NULL name is "
                  "forbidden. Please check your code and make sure you new a "
                  "name before calling this constructor."));

          PADDLE_ENFORCE_NOT_NULL(
              kw_dtype,
              paddle::platform::errors::Fatal(
                  "Calling __init__ of Eager Tensor with NULL dtype is "
                  "forbidden. Please check your code and make sure you new a "
                  "dtype before calling this constructor."));

          PADDLE_ENFORCE_NOT_NULL(
              kw_persistable,
              paddle::platform::errors::Fatal(
                  "Calling __init__ of Eager Tensor with NULL persistable is "
                  "forbidden. Please check your code and make sure you new a "
                  "persistable before calling this constructor."));

          paddle::framework::proto::VarType::Type dtype =
              CastPyArg2ProtoType(kw_dtype, 0);
          std::vector<int> dims = CastPyArg2VectorOfInt(kw_dims, 0);

          std::string act_name = "";
          if (kw_name == Py_None) {
            act_name = egr::Controller::Instance().GenerateUniqueName(
                "generated_tensor");
          } else {
            act_name = CastPyArg2AttrString(kw_name, 0);
          }

          paddle::framework::proto::VarType::Type var_type =
              CastPyArg2ProtoType(kw_type, 0);
          bool persistable = CastPyArg2AttrBoolean(kw_persistable, 0);

          EmptyEagerTensorInitializer(
              py_tensor_ptr, act_name,
              egr::Controller::Instance().GetExpectedPlace(), persistable,
              /* stop_gradient */ true, dtype, dims, var_type);

          return 0;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "We support construct tensor from numpy value or tensor "
              "with python args and kwargs by this initializer, "
              "please check your input first and make sure you are on the "
              "right way."));
        }
      }
    }
    case (Py_ssize_t)1:
    case (Py_ssize_t)2:
    case (Py_ssize_t)3: {
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
        VLOG(6) << "Calling case3's or case4's initializer.";
        AutoInitEagerTensorByPyArray(py_tensor_ptr, kws_map, args, flag_kwargs,
                                     args_num);
        return 0;
      } else if (PyObject_IsInstance(arg0_ptr, reinterpret_cast<PyObject*>(
                                                   p_eager_tensor_type))) {
        VLOG(6) << "Calling case5's or case6's initializer.";
        AutoInitEagerTensorByTensor(py_tensor_ptr, kws_map, args, flag_kwargs,
                                    args_num);
        return 0;
      } else if (PyObject_IsInstance(
                     arg0_ptr,
                     reinterpret_cast<PyObject*>(g_framework_tensor_pytype))) {
        VLOG(6) << "Calling case7's initializer.";
        AutoInitEagerTensorByTensor(
            py_tensor_ptr, kws_map, args, flag_kwargs, args_num,
            /* false means not init by egr tensor*/ false);
        return 0;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "We support construct tensor from numpy value or tensor "
            "with python args and kwargs by this initializer, "
            "please check your input first and make sure you are on the "
            "right way."));
      }
      return 0;
    }
    case (Py_ssize_t)4: {
      if (!flag_kwargs) {
        PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
        if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
          VLOG(6) << "Calling case3's initializer.";
          AutoInitEagerTensorByPyArray(py_tensor_ptr, kws_map, args,
                                       flag_kwargs, args_num);
          return 0;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "We support construct tensor from numpy value or tensor with "
              "python args and kwargs by this initializer. "
              "please check your input first and make sure you are on the "
              "right way."));
        }
      } else {  // four position args, remainting arguments are kwargs
        PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
        if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
          VLOG(6) << "Calling case3's or case4's initializer";
          AutoInitEagerTensorByPyArray(py_tensor_ptr, kws_map, args,
                                       flag_kwargs, args_num);
          return 0;
        } else if (PyObject_IsInstance(arg0_ptr, reinterpret_cast<PyObject*>(
                                                     p_eager_tensor_type))) {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "We support construct tensor from numpy value or tensor with "
              "python args and kwargs by this initializer. "
              "please check your input first and make sure you are on the "
              "right way."));
        }
      }
      return 0;
    }
    case (Py_ssize_t)5: {
      if (!flag_kwargs) {
        PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
        if (PyObject_IsInstance(
                arg0_ptr, reinterpret_cast<PyObject*>(g_vartype_pytype))) {
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
          bool persistable =
              CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4);
          EmptyEagerTensorInitializer(
              py_tensor_ptr, act_name,
              egr::Controller::Instance().GetExpectedPlace(), persistable, true,
              dtype, dims, var_type);
          return 0;
        } else if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
          AutoInitEagerTensorByPyArray(py_tensor_ptr, kws_map, args,
                                       flag_kwargs, args_num);
          return 0;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "We support construct tensor from numpy value or tensor "
              "with python args and kwargs by this initializer, "
              "please check your input first and make sure you are on the "
              "right way."));
        }
      } else {  // five position args, remainting arguments are kwargs
        PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
        if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
          VLOG(6) << "Calling case3's or case4's initializer";
          AutoInitEagerTensorByPyArray(py_tensor_ptr, kws_map, args,
                                       flag_kwargs, args_num);
          return 0;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "We support construct tensor from numpy value or tensor with "
              "python args and kwargs by this initializer. "
              "please check your input first and make sure you are on the "
              "right way."));
        }
      }
    }
    case (Py_ssize_t)6: {
      if (!flag_kwargs) {
        // case 3
        VLOG(6) << "Calling case3's initializer.";
        AutoInitEagerTensorByPyArray(py_tensor_ptr, kws_map, args, flag_kwargs,
                                     args_num);
        return 0;
      } else {  // six position args, remainting arguments are kwargs, but this
                // is not a right way
        PADDLE_THROW(platform::errors::InvalidArgument(
            "There are six position args and the remainting arguments are "
            "kwargs,"
            "please check your input first and make sure you are on the right "
            "way."));
      }
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
        "Init Paddle error in BindEager(PyType_Ready)."));
    return;
  }

  Py_INCREF(&eager_tensor_type);
  if (PyModule_AddObject(m.ptr(), "EagerTensor",
                         reinterpret_cast<PyObject*>(&eager_tensor_type)) < 0) {
    Py_DECREF(&eager_tensor_type);
    Py_DECREF(m.ptr());
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEager(PyModule_AddObject)."));
    return;
  }

  BindFunctions(m.ptr());
  BindEagerOpFunctions(&m);
}

}  // namespace pybind
}  // namespace paddle
