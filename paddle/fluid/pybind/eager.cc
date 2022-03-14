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
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "pybind11/detail/internals.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/pybind/eager_op_function_impl.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/api/lib/utils/tensor_utils.h"
namespace paddle {
namespace pybind {

namespace py = ::pybind11;

PyTypeObject* p_tensor_type;
extern PyTypeObject* g_vartype_pytype;
extern PyTypeObject* g_framework_tensor_pytype;

PyObject* TensorNew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = reinterpret_cast<TensorObject*>(obj);
    new (&(v->tensor)) paddle::experimental::Tensor();
  }
  return obj;
}

// TODO(jiabin): Overload this once we need more constructor in Python
void EmptyTensorInitializer(TensorObject* self, const std::string& name,
                            const paddle::platform::Place& place,
                            bool persistable = false, int stop_gradient = -1,
                            framework::proto::VarType::Type dtype =
                                paddle::framework::proto::VarType::FP32,
                            const std::vector<int>& dims = {},
                            framework::proto::VarType::Type var_type =
                                paddle::framework::proto::VarType::LOD_TENSOR) {
  auto ddims = phi::make_ddim(dims);
  self->tensor.set_name(name);
  auto autograd_meta = egr::EagerUtils::autograd_meta(&(self->tensor));
  autograd_meta->SetPersistable(persistable);
  if (stop_gradient != -1) {
    autograd_meta->SetStopGradient(static_cast<bool>(stop_gradient));
  }
  if (var_type == paddle::framework::proto::VarType::LOD_TENSOR) {
    // TODO(jiabin): Maybe support LOD later
    std::shared_ptr<phi::DenseTensor> dense_tensor =
        std::make_shared<phi::DenseTensor>(
            phi::make_intrusive<paddle::experimental::SharedStorage>(place),
            phi::DenseTensorMeta(paddle::framework::TransToPhiDataType(dtype),
                                 ddims));
    if (phi::product(ddims) > 0) {
      dense_tensor->mutable_data(place);
    }
    self->tensor.set_impl(dense_tensor);
  }

  if (!autograd_meta->GetMutableGradNode()) {
    VLOG(3) << "Tensor(" << name
            << ") have not GradNode, add GradNodeAccumulation for it.";
    autograd_meta->SetGradNode(
        std::make_shared<egr::GradNodeAccumulation>(autograd_meta));
  }
}

void InitTensorWithNumpyValue(TensorObject* self, const py::object& array,
                              bool zero_copy = false) {
  PADDLE_ENFORCE_EQ(
      self->tensor.defined(), true,
      paddle::platform::errors::Fatal(
          "Calling InitTensorWithNumpyValue of Eager Tensor without "
          "EmptyTensorInitializer is "
          "forbidden. Please check your code and make sure you new a "
          "eager tensor before init it with NumPy."));
  phi::DenseTensor* impl_ptr =
      static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  paddle::platform::Place place = impl_ptr->place();
  if (platform::is_cpu_place(place)) {
    SetTensorFromPyArray<platform::CPUPlace>(impl_ptr, array, place, zero_copy);
  } else if (platform::is_xpu_place(place)) {
    SetTensorFromPyArray<platform::XPUPlace>(impl_ptr, array, place, zero_copy);
  } else if (platform::is_gpu_place(place)) {
    SetTensorFromPyArray<platform::CUDAPlace>(impl_ptr, array, place,
                                              zero_copy);
  } else if (platform::is_cuda_pinned_place(place)) {
    SetTensorFromPyArray<platform::CUDAPinnedPlace>(impl_ptr, array, place,
                                                    zero_copy);
  } else if (platform::is_npu_place(place)) {
    SetTensorFromPyArray<platform::NPUPlace>(impl_ptr, array, place, zero_copy);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place should be one of "
        "CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace/NPUPlace"));
  }
}

void InitTensorWithTensor(TensorObject* self,
                          const paddle::experimental::Tensor& src,
                          const paddle::platform::Place& place,
                          const std::string& name) {
  self->tensor.set_name(name);
  if (place == src.inner_place()) {
    auto impl = std::static_pointer_cast<phi::DenseTensor>(src.impl());
    self->tensor.set_impl(impl);
    VLOG(4) << "Same place, do ShareDataWith";
  } else {
    self->tensor.set_impl(
        src.copy_to(phi::TransToPhiBackend(place), true).impl());
    VLOG(4) << "Different place, do TensorCopy";
  }
  if (src.get_autograd_meta()) {
    egr::EagerUtils::autograd_meta(&(self->tensor))
        ->SetPersistable(
            egr::EagerUtils::unsafe_autograd_meta(src)->Persistable());
  } else {
    egr::EagerUtils::autograd_meta(&(self->tensor))->SetPersistable(false);
  }
}

void InitTensorWithFrameworkTensor(TensorObject* self,
                                   const framework::Tensor& src,
                                   const paddle::platform::Place& place,
                                   const std::string& name) {
  self->tensor.set_name(name);
  if (place == src.place()) {
    self->tensor.set_impl(std::make_shared<phi::DenseTensor>(src));
    VLOG(4) << "Same place, do ShareDataWith";
  } else {
    auto temp =
        paddle::experimental::Tensor(std::make_shared<phi::DenseTensor>(src));
    self->tensor.set_impl(
        temp.copy_to(phi::TransToPhiBackend(place), true).impl());
    VLOG(4) << "Different place, do TensorCopy";
  }
  egr::EagerUtils::autograd_meta(&(self->tensor))->SetPersistable(false);
}

py::object ParsePyArray(
    std::unordered_map<std::string, PyObject*> kws_map,
    std::unordered_map<std::string, Py_ssize_t> kw_order_map, PyObject* args,
    bool flag_kwargs, Py_ssize_t args_num) {
  py::object numpy_value = py::object();

  if (kw_order_map["value"] <= args_num) {
    numpy_value = py::object(
        py::handle(PyTuple_GET_ITEM(args, kw_order_map["value"] - 1)), true);
  } else {
    if (flag_kwargs && kws_map["value"] != NULL) {
      numpy_value = py::object(py::handle(kws_map["value"]), true);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The first expected arguments is {value: PyArray}, "
          "but could not parse the first argument {value: PyArray} "
          "successfully. "
          "Please check your input first and make sure you are on the right "
          "way."));
    }
  }
  return numpy_value;
}

paddle::platform::Place ParsePlace(
    std::unordered_map<std::string, PyObject*> kws_map,
    std::unordered_map<std::string, Py_ssize_t> kw_order_map, PyObject* args,
    bool flag_kwargs, Py_ssize_t args_num) {
  paddle::platform::Place place =
      egr::Controller::Instance().GetExpectedPlace();

  if (kw_order_map["place"] <= args_num) {
    place = CastPyArg2Place(PyTuple_GET_ITEM(args, kw_order_map["place"] - 1),
                            kw_order_map["place"] - 1);
  } else {
    if (flag_kwargs && kws_map["place"] != NULL) {
      place = CastPyArg2Place(kws_map["place"], 0);
    } else {
      // default
      return place;
    }
  }
  return place;
}

// boolean arguments: zero_copy, stop_gradient, persistable
int ParseBooleanArgs(std::string key,
                     std::unordered_map<std::string, PyObject*> kws_map,
                     std::unordered_map<std::string, Py_ssize_t> kw_order_map,
                     PyObject* args, bool flag_kwargs, Py_ssize_t args_num) {
  int res = -1;

  if (kw_order_map[key] <= args_num) {
    res = static_cast<int>(CastPyArg2AttrBoolean(
        PyTuple_GET_ITEM(args, kw_order_map[key] - 1), kw_order_map[key] - 1));
  } else {
    if (flag_kwargs && kws_map[key] != NULL) {
      res = static_cast<int>(CastPyArg2AttrBoolean(kws_map[key], 0));
    }
  }
  return res;
}

std::string ParseName(std::unordered_map<std::string, PyObject*> kws_map,
                      std::unordered_map<std::string, Py_ssize_t> kw_order_map,
                      PyObject* args, bool flag_kwargs, Py_ssize_t args_num) {
  std::string act_name = "";
  if (kw_order_map["name"] <= args_num) {
    PyObject* name_obj = PyTuple_GET_ITEM(args, kw_order_map["name"] - 1);
    if (name_obj == Py_None) {
      act_name =
          egr::Controller::Instance().GenerateUniqueName("generated_tensor");
    } else {
      act_name = CastPyArg2AttrString(name_obj, kw_order_map["name"] - 1);
    }
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
  return act_name;
}

// initialize Tensor by PyArray(first argument is PyArray,
// mix args and kwargs) automatically.
void AutoInitTensorByPyArray(TensorObject* py_tensor_ptr,
                             std::unordered_map<std::string, PyObject*> kws_map,
                             PyObject* args, bool flag_kwargs,
                             Py_ssize_t args_num) {
  // The first argument of the Tensor constructor is PyArray,
  // there are 6 arguments to construct the new Tensor,
  // kw_order_map's key is every arguments of the constructor,
  // kw_order_map's value is the position of the arguments respectively.
  // If u want to update this constructor with new arguments,
  // need to update this map and to add or change related code.
  std::unordered_map<std::string, Py_ssize_t> kw_order_map{
      {"value", 1},     {"place", 2}, {"persistable", 3},
      {"zero_copy", 4}, {"name", 5},  {"stop_gradient", 6}};

  py::object numpy_value = py::object();
  paddle::platform::Place place =
      egr::Controller::Instance().GetExpectedPlace();
  bool persistable = false;
  bool zero_copy = false;
  std::string act_name = "";
  int stop_gradient = -1;

  numpy_value =
      ParsePyArray(kws_map, kw_order_map, args, flag_kwargs, args_num);
  place = ParsePlace(kws_map, kw_order_map, args, flag_kwargs, args_num);
  persistable = (1 == ParseBooleanArgs("persistable", kws_map, kw_order_map,
                                       args, flag_kwargs, args_num));
  zero_copy = (1 == ParseBooleanArgs("zero_copy", kws_map, kw_order_map, args,
                                     flag_kwargs, args_num));
  act_name = ParseName(kws_map, kw_order_map, args, flag_kwargs, args_num);
  stop_gradient = ParseBooleanArgs("stop_gradient", kws_map, kw_order_map, args,
                                   flag_kwargs, args_num);

  EmptyTensorInitializer(py_tensor_ptr, act_name, place, persistable,
                         stop_gradient);
  InitTensorWithNumpyValue(py_tensor_ptr, numpy_value, zero_copy);
}

// initialize Tensor by Tensor or framework::Tensor (mix args and
// kwargs) automatically.
void AutoInitTensorByTensor(TensorObject* py_tensor_ptr,
                            std::unordered_map<std::string, PyObject*> kws_map,
                            PyObject* args, bool flag_kwargs,
                            Py_ssize_t args_num,
                            bool init_by_egr_tensor = true) {
  // The first argument of the Tensor constructor is Tensor or
  // framework Tensor,
  // there are 3 arguments to construct the new Tensor,
  // kw_order_map's key is every arguments of the constructor,
  // kw_order_map's value is the position of the arguments respectively.
  // If u want to update this constructor with new arguments,
  // need to update this map and to add or change related code.
  std::unordered_map<std::string, Py_ssize_t> kw_order_map{
      {"value", 1}, {"place", 2}, {"name", 3}};

  paddle::platform::Place place =
      egr::Controller::Instance().GetExpectedPlace();
  std::string act_name = "";

  place = ParsePlace(kws_map, kw_order_map, args, flag_kwargs, args_num);
  act_name = ParseName(kws_map, kw_order_map, args, flag_kwargs, args_num);

  if (init_by_egr_tensor) {
    paddle::experimental::Tensor src_tensor;
    if (kw_order_map["value"] <= args_num) {
      src_tensor =
          CastPyArg2Tensor(PyTuple_GET_ITEM(args, kw_order_map["value"] - 1),
                           kw_order_map["value"] - 1);
    } else {
      if (flag_kwargs && kws_map["value"] != NULL) {
        src_tensor = CastPyArg2Tensor(kws_map["value"], 0);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The first expected kwargs is {value: Tensor}, "
            "but could not parse the first argument {value: Tensor} "
            "successfully. "
            "Please check your input first and make sure you are on the right "
            "way."));
      }
    }
    InitTensorWithTensor(py_tensor_ptr, src_tensor, place, act_name);
  } else {
    // init by framework tensor
    framework::Tensor src_tensor;
    if (kw_order_map["value"] <= args_num) {
      src_tensor = CastPyArg2FrameworkTensor(
          PyTuple_GET_ITEM(args, kw_order_map["value"] - 1),
          kw_order_map["value"] - 1);
    } else {
      if (flag_kwargs && kws_map["value"] != NULL) {
        src_tensor = CastPyArg2FrameworkTensor(kws_map["value"], 0);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The first expected arguments is {value: framework::Tensor}, "
            "but could not parse the first argument {value: framework::Tensor} "
            "successfully. "
            "Please check your input first and make sure you are on the right "
            "way."));
      }
    }
    InitTensorWithFrameworkTensor(py_tensor_ptr, src_tensor, place, act_name);
  }
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
   * ** tensor: Tensor)
   * 6. (multi-place)
   * (should have at least one parameter, one parameter equals to case 5, zero
   * parameter equals to case 1.)
   * def __init__ (
   * ** tensor: Tensor,
   * ** place: paddle::platform::Place,
   * ** name: std::string)
   * 7. (multi-place) (should have at least one parameter, one parameter similar
   * to case 5, zero parameter equals to case 1.)
   * def __init__ (
   * ** tensor: FrameworkTensor,
   * ** place: paddle::platform::Place,
   * ** name: std::string)
   *  **/
int TensorInit(PyObject* self, PyObject* args, PyObject* kwargs) {
  // set a flag to record use kwargs or not
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;

  // all kwargs
  PyObject* kw_zero_copy = NULL;
  PyObject* kw_persistable = NULL;
  PyObject* kw_stop_gradient = NULL;

  PyObject* kw_value = NULL;  // receive PyArray or Tensor
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
      const_cast<char*>("type"),        NULL};

  // 'O' Store a Python object (without any conversion) in a C object pointer,
  // '|' Indicates that the remaining arguments in the Python argument list are
  // optional.
  // PyArg_ParseTupleAndKeywords can Parse the parameters of a function that
  // takes both positional and keyword parameters into local variables,
  // which enhance case2, case3, case4, case5, case6, case7.
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

  PADDLE_ENFORCE_EQ(flag_, true,
                    paddle::platform::errors::PreconditionNotMet(
                        "Could not parse args and kwargs successfully, "
                        "please check your input first and make"
                        "sure you are on the right way. "
                        "The expected arguments as follow: ("
                        "value, place, persistable, zero_copy, "
                        "name, stop_gradient, dims, dtype, type)"));

  PADDLE_ENFORCE_NOT_NULL(
      self, paddle::platform::errors::Fatal(
                "Calling __init__ of Eager Tensor without __new__ is "
                "forbidden. Please check your code and make sure you new a "
                "eager tensor before init it."));

  auto py_tensor_ptr = reinterpret_cast<TensorObject*>(self);

  Py_ssize_t args_num = PyTuple_Size(args);
  VLOG(6) << " args_num: " << args_num;

  // args_num = 0, means that there is no position arguments.
  if (args_num == (Py_ssize_t)0) {
    if (!flag_kwargs) {
      // case 1
      VLOG(6) << "Calling case1's initializer.";
      EmptyTensorInitializer(
          py_tensor_ptr,
          egr::Controller::Instance().GenerateUniqueName("generated_tensor"),
          egr::Controller::Instance().GetExpectedPlace());
      return 0;
    } else {  // no position args, all arguments are kwargs
      if (kw_value != NULL) {
        if (pybind11::detail::npy_api::get().PyArray_Check_(kw_value)) {
          VLOG(6) << "Calling case3's or case4's initializer";
          AutoInitTensorByPyArray(py_tensor_ptr, kws_map, args, flag_kwargs,
                                  args_num);
          return 0;
        } else if (PyObject_IsInstance(
                       kw_value, reinterpret_cast<PyObject*>(p_tensor_type))) {
          VLOG(6) << "Calling case5's or case6's initializer";
          AutoInitTensorByTensor(py_tensor_ptr, kws_map, args, flag_kwargs,
                                 args_num);
          return 0;
        } else if (PyObject_IsInstance(kw_value,
                                       reinterpret_cast<PyObject*>(
                                           g_framework_tensor_pytype))) {
          VLOG(6) << "Calling case7's initializer.";
          AutoInitTensorByTensor(py_tensor_ptr, kws_map, args, flag_kwargs,
                                 args_num,
                                 /* false means not init by egr tensor*/ false);
          return 0;
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Could not parse the first keyword argument successfully, "
              "the first keyword argument is value, but it should be PyArray "
              "or Tensor or framework::Tensor. "
              "Please check your input first and make sure you are on the "
              "right way."));
        }
      } else if (kw_dtype != NULL &&
                 PyObject_IsInstance(
                     kw_dtype, reinterpret_cast<PyObject*>(g_vartype_pytype))) {
        VLOG(6) << "Calling case2's initializer";

        PADDLE_ENFORCE_NOT_NULL(
            kw_dims,
            paddle::platform::errors::InvalidArgument(
                "Calling __init__ of Eager Tensor with NULL dims is "
                "forbidden. Please check your code and make sure you new a "
                "dims before calling this constructor."));

        PADDLE_ENFORCE_NOT_NULL(
            kw_name,
            paddle::platform::errors::InvalidArgument(
                "Calling __init__ of Eager Tensor with NULL name is "
                "forbidden. Please check your code and make sure you new a "
                "name before calling this constructor."));

        PADDLE_ENFORCE_NOT_NULL(
            kw_dtype,
            paddle::platform::errors::InvalidArgument(
                "Calling __init__ of Eager Tensor with NULL dtype is "
                "forbidden. Please check your code and make sure you new a "
                "dtype before calling this constructor."));

        PADDLE_ENFORCE_NOT_NULL(
            kw_persistable,
            paddle::platform::errors::InvalidArgument(
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

        EmptyTensorInitializer(py_tensor_ptr, act_name,
                               egr::Controller::Instance().GetExpectedPlace(),
                               persistable,
                               /* stop_gradient */ -1, dtype, dims, var_type);

        return 0;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "We not only support construct Tensor from numpy value "
            "or tensor(Tensor or framework::Tensor) "
            "with python kwargs by this initializer, "
            "but also even support dtype to init a empty Tensor. "
            "Please check your input first and make sure you call the existed "
            "constructor."));
      }
    }
  } else if (args_num == (Py_ssize_t)1 || args_num == (Py_ssize_t)2 ||
             args_num == (Py_ssize_t)3) {
    // 1 to 3 position args, remainting arguments are kwargs
    PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
    if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
      VLOG(6) << "Calling case3's or case4's initializer.";
      AutoInitTensorByPyArray(py_tensor_ptr, kws_map, args, flag_kwargs,
                              args_num);
      return 0;
    } else if (PyObject_IsInstance(
                   arg0_ptr, reinterpret_cast<PyObject*>(p_tensor_type))) {
      VLOG(6) << "Calling case5's or case6's initializer.";
      AutoInitTensorByTensor(py_tensor_ptr, kws_map, args, flag_kwargs,
                             args_num);
      return 0;
    } else if (PyObject_IsInstance(arg0_ptr, reinterpret_cast<PyObject*>(
                                                 g_framework_tensor_pytype))) {
      VLOG(6) << "Calling case7's initializer.";
      AutoInitTensorByTensor(py_tensor_ptr, kws_map, args, flag_kwargs,
                             args_num,
                             /* false means not init by egr tensor*/ false);
      return 0;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "We support construct Tensor from numpy value "
          "or tensor(Tensor or framework::Tensor) "
          "with python args and kwargs by this initializer, "
          "but the first argument should be PyArray or Tensor or "
          "framework::Tensor. "
          "Please check your input first and make sure you call the existed "
          "constructor."));
    }
  } else if (args_num == (Py_ssize_t)4) {
    // 4 position args, remainting arguments are kwargs
    PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
    if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
      VLOG(6) << "Calling case3's or case4's initializer.";
      AutoInitTensorByPyArray(py_tensor_ptr, kws_map, args, flag_kwargs,
                              args_num);
      return 0;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Incompatible constructor arguments, "
          "there are 4 position args and remainting arguments arg kwargs,"
          "but the first position args should be PyArray. "
          "Please check your code and make sure the first position args is "
          "PyArray."));
    }
  } else if (args_num == (Py_ssize_t)5) {
    if (!flag_kwargs) {
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
        EmptyTensorInitializer(py_tensor_ptr, act_name,
                               egr::Controller::Instance().GetExpectedPlace(),
                               persistable, -1, dtype, dims, var_type);
        return 0;
      } else if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
        VLOG(6) << "Calling case3's initializer.";
        AutoInitTensorByPyArray(py_tensor_ptr, kws_map, args, flag_kwargs,
                                args_num);
        return 0;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Incompatible constructor arguments, "
            "there are only 5 position args,"
            "but the first position args should be PyArray or dtype. "
            "Please check your code and make sure you call the existed "
            "constructor."));
      }
    } else {  // five position args, remainting arguments are kwargs
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
        VLOG(6) << "Calling case3's or case4's initializer";
        AutoInitTensorByPyArray(py_tensor_ptr, kws_map, args, flag_kwargs,
                                args_num);
        return 0;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Incompatible constructor arguments, "
            "there are 5 position args and remainting arguments are kwargs,"
            "but the first position args should be PyArray. "
            "Please check your code and make sure the first position args is "
            "PyArray."));
      }
    }
  } else if (args_num == (Py_ssize_t)6) {
    if (!flag_kwargs) {
      // case 3
      VLOG(6) << "Calling case3's initializer.";
      AutoInitTensorByPyArray(py_tensor_ptr, kws_map, args, flag_kwargs,
                              args_num);
      return 0;
    } else {  // six position args, remainting arguments are kwargs, but this
              // is not a right way
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Incompatible constructor arguments, "
          "there are 6 position args and the remainting arguments are kwargs. "
          "Please check your code and make sure the first position args is "
          "PyArray."));
    }
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "Can't not find expected num of args, please check your call, and "
        "make sure u call the existed constructor."));
  }

  return 1;
}

static void TensorDealloc(TensorObject* self) {
  self->tensor.~Tensor();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

extern struct PyGetSetDef variable_properties[];

extern PyMethodDef variable_methods[];

PyNumberMethods number_methods;
PySequenceMethods sequence_methods;
PyMappingMethods mapping_methods;

void BindEager(pybind11::module* module) {
  auto m = module->def_submodule("eager");

  auto heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  heap_type->ht_name = ToPyObject("Tensor");
  heap_type->ht_qualname = ToPyObject("Tensor");
  auto type = &heap_type->ht_type;
  type->tp_name = "Tensor";
  type->tp_basicsize = sizeof(TensorObject);
  type->tp_dealloc = (destructor)TensorDealloc;
  type->tp_as_number = &number_methods;
  type->tp_as_sequence = &sequence_methods;
  type->tp_as_mapping = &mapping_methods;
  type->tp_methods = variable_methods;
  type->tp_getset = variable_properties;
  type->tp_init = TensorInit;
  type->tp_new = TensorNew;
  Py_INCREF(&PyBaseObject_Type);
  type->tp_base = reinterpret_cast<PyTypeObject*>(&PyBaseObject_Type);
  type->tp_flags |=
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
#if PY_VERSION_HEX >= 0x03050000
  type->tp_as_async = &heap_type->as_async;
#endif
  p_tensor_type = type;

  if (PyType_Ready(type) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEager(PyType_Ready)."));
    return;
  }

  Py_INCREF(type);
  if (PyModule_AddObject(m.ptr(), "Tensor", reinterpret_cast<PyObject*>(type)) <
      0) {
    Py_DECREF(type);
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
