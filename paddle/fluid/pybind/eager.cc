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
#include "paddle/fluid/pybind/eager.h"

#include <Python.h>
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include <string>
#include <vector>

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "pybind11/detail/internals.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/placement_types.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/string_tensor.h"

using phi::distributed::DistTensor;
using phi::distributed::DistTensorMeta;
using phi::distributed::Placement;
using phi::distributed::Placements;
using phi::distributed::ProcessMesh;
using phi::distributed::TensorDistAttr;
using phi::distributed::auto_parallel::str_join;

namespace paddle::pybind {

namespace py = ::pybind11;

extern PyTypeObject* p_tensor_type;
extern PyTypeObject* p_string_tensor_type;  // For StringTensor
extern PyTypeObject* g_vartype_pytype;
extern PyTypeObject* g_data_type_pytype;
extern PyTypeObject* g_framework_tensor_pytype;

PyObject* TensorNew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = reinterpret_cast<TensorObject*>(obj);
    new (&(v->tensor)) paddle::Tensor();
  }
  return obj;
}

// TODO(jiabin): Overload this once we need more constructor in Python
void EmptyTensorInitializer(TensorObject* self,
                            const std::string& name,
                            const phi::Place& place,
                            bool persistable = false,
                            int stop_gradient = -1,
                            paddle::DataType dtype = paddle::DataType::FLOAT32,
                            const std::vector<int>& dims = {0},
                            framework::proto::VarType::Type var_type =
                                paddle::framework::proto::VarType::DENSE_TENSOR,
                            ProcessMesh* process_mesh = nullptr,
                            Placements* placements = nullptr) {
  auto ddims = common::make_ddim(dims);
  self->tensor.set_name(name);
  auto autograd_meta = egr::EagerUtils::autograd_meta(&(self->tensor));
  autograd_meta->SetPersistable(persistable);
  if (stop_gradient != -1) {
    autograd_meta->SetStopGradient(static_cast<bool>(stop_gradient));
  }
  if (process_mesh != nullptr) {
#ifdef PADDLE_WITH_DISTRIBUTE
    VLOG(6) << "in EmptyTensorInitializer, create DistTensor";
    self->tensor.set_impl(std::make_shared<DistTensor>());
#else
    PADDLE_THROW(common::errors::Unavailable(
        "The tensor-based initialization of (Dist)Tensor is not supported "
        "in the current PaddlePaddle, please recompile and install "
        "PaddlePaddle "
        "with the option of `WITH_DISTRIBUTE=ON`."));
#endif
  } else {
    VLOG(6) << "in EmptyTensorInitializer, create DenseTensor";
    if (var_type == paddle::framework::proto::VarType::DENSE_TENSOR) {
      // TODO(jiabin): Maybe support LegacyLoD later
      std::shared_ptr<phi::DenseTensor> dense_tensor = nullptr;
      if (dims.size() == 1 && dims[0] == 0) {
        std::shared_ptr<phi::Allocation> allocation_ptr = nullptr;
        dense_tensor = std::make_shared<phi::DenseTensor>(
            allocation_ptr, phi::DenseTensorMeta(dtype, ddims));
      } else {
        // TODO(dev): we need enhance check for ddims.
        dense_tensor = std::make_shared<phi::DenseTensor>(
            std::make_shared<phi::Allocation>(),
            phi::DenseTensorMeta(dtype, ddims));
      }
      self->tensor.set_impl(dense_tensor);
    } else if (var_type == paddle::framework::proto::VarType::SELECTED_ROWS) {
      std::shared_ptr<phi::SelectedRows> tensor =
          std::make_shared<phi::SelectedRows>();
      self->tensor.set_impl(tensor);
    }
  }

  if (!autograd_meta->GetMutableGradNode()) {
    autograd_meta->SetGradNode(
        std::make_shared<egr::GradNodeAccumulation>(autograd_meta));
    VLOG(3) << "Tensor(" << name
            << ") have not GradNode, add GradNodeAccumulation"
            << autograd_meta->GradNode() << " for it.";
  }
}

void EmptyStringTensorInitializer(TensorObject* self,
                                  const std::string& name,
                                  const phi::Place& place,
                                  const std::vector<int>& dims = {}) {
  auto ddims = common::make_ddim(dims);
  self->tensor.set_name(name);
  // Note(zhoushunjie): Only support CPUPlace when create StringTensor
  auto actual_place = phi::CPUPlace();
  // Allocate memory
  paddle::experimental::DefaultAllocator string_allocator(actual_place);
  std::shared_ptr<phi::StringTensor> string_tensor =
      std::make_shared<phi::StringTensor>(&string_allocator,
                                          phi::StringTensorMeta{ddims});
  if (common::product(ddims) > 0) {
    string_tensor->mutable_data(actual_place);
  }
  self->tensor.set_impl(string_tensor);
}

void InitTensorWithNumpyValue(TensorObject* self,
                              const py::object& array,
                              const phi::Place& place,
                              bool zero_copy = false) {
  PADDLE_ENFORCE_EQ(
      self->tensor.defined(),
      true,
      common::errors::Unavailable(
          "Calling InitTensorWithNumpyValue of Eager Tensor without "
          "EmptyTensorInitializer is "
          "forbidden. Please check your code and make sure you new a "
          "eager tensor before init it with NumPy."));

  phi::DenseTensor* impl_ptr =
      static_cast<phi::DenseTensor*>(self->tensor.impl().get());
  if (phi::is_cpu_place(place)) {
    SetTensorFromPyArray<phi::CPUPlace>(impl_ptr, array, place, zero_copy);
  } else if (phi::is_xpu_place(place)) {
#if defined(PADDLE_WITH_XPU)
    phi::backends::xpu::SetXPUDeviceId(place.device);
    VLOG(4) << "CurrentDeviceId: "
            << phi::backends::xpu::GetXPUCurrentDeviceId() << " from "
            << static_cast<int>(place.device);
#else
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with XPU if use XPUPlace."));
#endif
    SetTensorFromPyArray<phi::XPUPlace>(impl_ptr, array, place, zero_copy);
  } else if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::backends::gpu::SetDeviceId(place.device);
    VLOG(4) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
            << " from " << static_cast<int>(place.device);
#else
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    SetTensorFromPyArray<phi::GPUPlace>(impl_ptr, array, place, zero_copy);
  } else if (phi::is_cuda_pinned_place(place)) {
    SetTensorFromPyArray<phi::GPUPinnedPlace>(
        impl_ptr, array, place, zero_copy);
  } else if (phi::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
    phi::DeviceManager::SetDevice(place);
    VLOG(4) << "CurrentDeviceId: "
            << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from "
            << static_cast<int>(place.device);
#else
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    SetTensorFromPyArray<phi::CustomPlace>(impl_ptr, array, place, zero_copy);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Place should be one of "
        "CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace/CustomPlace"));
  }
}

void InitStringTensorWithNumpyValue(TensorObject* self, const py::object& obj) {
  PADDLE_ENFORCE_EQ(
      self->tensor.defined(),
      true,
      common::errors::Fatal(
          "Calling InitStringTensorWithNumpyValue of Eager StringTensor "
          "without "
          "EmptyStringTensorInitializer is "
          "forbidden. Please check your code and make sure you new a "
          "eager tensor before init it with NumPy."));
  phi::StringTensor* impl_ptr =
      static_cast<phi::StringTensor*>(self->tensor.impl().get());
  phi::Place place = impl_ptr->place();
  auto array = obj.cast<py::array>();
  if (phi::is_cpu_place(place)) {
    SetStringTensorFromPyArray<phi::CPUPlace>(impl_ptr, array, place);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "StringTensor only support CPUPlace now, but receive %s",
        place.DebugString()));
  }
}

void InitDistTensorWithTensor(TensorObject* self,
                              const paddle::Tensor& src,
                              const phi::Place& place,
                              const std::string& name,
                              const ProcessMesh& process_mesh,
                              const Placements& placements) {
#ifdef PADDLE_WITH_DISTRIBUTE
  PADDLE_ENFORCE_EQ(src.is_dense_tensor(),
                    true,
                    common::errors::InvalidArgument(
                        "DistTensor can only initialize by DenseTensor"));
  self->tensor.set_name(name);
  VLOG(4) << "Do TensorCopy from DenseTensor to DistTensor.";
  if (place == src.place()) {
    std::shared_ptr<phi::DenseTensor> tensor =
        std::static_pointer_cast<phi::DenseTensor>(src.impl());
    self->tensor.set_impl(
        std::make_shared<DistTensor>(tensor, process_mesh, placements));
    VLOG(4) << "Same place, do ShareDataWith for DistTensor.";
  } else {
    std::shared_ptr<phi::DenseTensor> tensor;
    if (src.initialized()) {
      tensor = std::static_pointer_cast<phi::DenseTensor>(
          src.copy_to(place, true).impl());
    } else {
      // lazy init branch. The src tensor is on undefined place.
      PADDLE_ENFORCE(
          src.place().GetType() == phi::AllocationType::UNDEFINED,
          common::errors::InvalidArgument("Only undefined place is support for "
                                          "uninitialized input tensor."));
      tensor = std::static_pointer_cast<phi::DenseTensor>(src.impl());
    }
    self->tensor.set_impl(
        std::make_shared<DistTensor>(tensor, process_mesh, placements));
    VLOG(4) << "Different place, do TensorCopy for DistTensor.";
  }
  if (src.get_autograd_meta()) {
    egr::EagerUtils::autograd_meta(&(self->tensor))
        ->SetPersistable(
            egr::EagerUtils::unsafe_autograd_meta(src)->Persistable());
  } else {
    egr::EagerUtils::autograd_meta(&(self->tensor))->SetPersistable(false);
  }
#else
  PADDLE_THROW(common::errors::Unavailable(
      "The tensor-based initialization of (Dist)Tensor is not supported "
      "in the current PaddlePaddle, please recompile and install PaddlePaddle "
      "with the option of `WITH_DISTRIBUTE=ON`."));
#endif
}

void InitDistTensorWithTensor(TensorObject* self,
                              const paddle::Tensor& local_tensor,
                              const std::vector<int>& global_dims,
                              const phi::Place& place,
                              const std::string& name,
                              const ProcessMesh& process_mesh,
                              const Placements& placements) {
#ifdef PADDLE_WITH_DISTRIBUTE
  PADDLE_ENFORCE_EQ(local_tensor.is_dense_tensor(),
                    true,
                    common::errors::InvalidArgument(
                        "DistTensor can only initialize by DenseTensor"));
  self->tensor.set_name(name);
  auto global_ddims = common::make_ddim(global_dims);
  VLOG(4) << "Do TensorCopy from DenseTensor to DistTensor.";
  if (place == local_tensor.place()) {
    std::shared_ptr<phi::DenseTensor> tensor =
        std::static_pointer_cast<phi::DenseTensor>(local_tensor.impl());
    self->tensor.set_impl(std::make_shared<DistTensor>(
        tensor, global_ddims, process_mesh, placements));
    VLOG(4) << "Same place, do ShareDataWith for DistTensor.";
  } else {
    std::shared_ptr<phi::DenseTensor> tensor =
        std::static_pointer_cast<phi::DenseTensor>(
            local_tensor.copy_to(place, true).impl());
    self->tensor.set_impl(std::make_shared<DistTensor>(
        tensor, global_ddims, process_mesh, placements));
    VLOG(4) << "Different place, do TensorCopy for DistTensor.";
  }
  if (local_tensor.get_autograd_meta()) {
    egr::EagerUtils::autograd_meta(&(self->tensor))
        ->SetPersistable(
            egr::EagerUtils::unsafe_autograd_meta(local_tensor)->Persistable());
  } else {
    egr::EagerUtils::autograd_meta(&(self->tensor))->SetPersistable(false);
  }
#else
  PADDLE_THROW(common::errors::Unavailable(
      "The tensor-based initialization of (Dist)Tensor is not supported "
      "in the current PaddlePaddle, please recompile and install PaddlePaddle "
      "with the option of `WITH_DISTRIBUTE=ON`."));
#endif
}

void InitTensorWithTensor(TensorObject* self,
                          const paddle::Tensor& src,
                          const phi::Place& place,
                          const std::string& name) {
  self->tensor.set_name(name);
  if (place == src.place()) {
    self->tensor.set_impl(src.impl());
    VLOG(4) << "Same place, do ShareDataWith";
  } else {
    self->tensor.set_impl(src.copy_to(place, true).impl());
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
                                   const phi::DenseTensor& src,
                                   const phi::Place& place,
                                   const std::string& name) {
  self->tensor.set_name(name);
  if (place == src.place()) {
    self->tensor.set_impl(std::make_shared<phi::DenseTensor>(src));
    VLOG(4) << "Same place, do ShareDataWith";
  } else {
    auto temp = paddle::Tensor(std::make_shared<phi::DenseTensor>(src));
    self->tensor.set_impl(temp.copy_to(place, true).impl());
    VLOG(4) << "Different place, do TensorCopy";
  }
  egr::EagerUtils::autograd_meta(&(self->tensor))->SetPersistable(false);
}

void InitStringTensorWithStringTensor(TensorObject* self,
                                      const paddle::Tensor& src,
                                      const phi::Place& place,
                                      const std::string& name) {
  self->tensor.set_name(name);
  auto impl = std::static_pointer_cast<phi::StringTensor>(src.impl());
  self->tensor.set_impl(impl);
  VLOG(4)
      << "Do ShareDataWith when using StringTensor to initialize StringTensor";
}

py::object ParsePyArray(
    std::unordered_map<std::string, PyObject*> kws_map,
    std::unordered_map<std::string, Py_ssize_t> kw_order_map,
    PyObject* args,
    bool flag_kwargs,
    Py_ssize_t args_num) {
  py::object numpy_value = py::object();

  if (kw_order_map["value"] <= args_num) {
    numpy_value = py::reinterpret_borrow<py::object>(
        py::handle(PyTuple_GET_ITEM(args, kw_order_map["value"] - 1)));
  } else {
    if (flag_kwargs && kws_map["value"] != nullptr) {
      numpy_value =
          py::reinterpret_borrow<py::object>(py::handle(kws_map["value"]));

    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The first expected arguments is {value: PyArray}, "
          "but could not parse the first argument {value: PyArray} "
          "successfully. "
          "Please check your input first and make sure you are on the right "
          "way."));
    }
  }
  return numpy_value;
}

phi::Place ParsePlace(std::unordered_map<std::string, PyObject*> kws_map,
                      std::unordered_map<std::string, Py_ssize_t> kw_order_map,
                      PyObject* args,
                      bool flag_kwargs,
                      Py_ssize_t args_num) {
  phi::Place place = egr::Controller::Instance().GetExpectedPlace();

  if (kw_order_map["place"] <= args_num) {
    place = CastPyArg2Place(PyTuple_GET_ITEM(args, kw_order_map["place"] - 1),
                            kw_order_map["place"] - 1);
  } else {
    if (flag_kwargs && kws_map["place"] != nullptr) {
      place = CastPyArg2Place(kws_map["place"], 0);
    } else {
      // default
      return place;
    }
  }
  return place;
}

ProcessMesh ParseProcessMeshArgs(
    std::unordered_map<std::string, PyObject*> kws_map,
    std::unordered_map<std::string, Py_ssize_t> kw_order_map,
    PyObject* args,
    bool flag_kwargs,
    Py_ssize_t args_num) {
  ProcessMesh process_mesh;
  if (kw_order_map["process_mesh"] <= args_num) {
    process_mesh = CastPyArg2ProcessMesh(
        PyTuple_GET_ITEM(args, kw_order_map["process_mesh"] - 1),
        kw_order_map["process_mesh"] - 1);
  } else if (flag_kwargs && kws_map["process_mesh"] != nullptr) {
    process_mesh = CastPyArg2ProcessMesh(kws_map["process_mesh"], 0);
  }
  return process_mesh;
}

Placements ParsePlacementsArgs(
    std::unordered_map<std::string, PyObject*> kws_map,
    std::unordered_map<std::string, Py_ssize_t> kw_order_map,
    PyObject* args,
    bool flag_kwargs,
    Py_ssize_t args_num) {
  Placements placements;
  const std::string& placements_key = "placements";

  if (kw_order_map[placements_key] <= args_num) {  // NOLINT
    placements = CastPyArg2VectorOfPlacement(
        PyTuple_GET_ITEM(args, kw_order_map[placements_key] - 1),
        kw_order_map[placements_key] - 1);
  } else if (flag_kwargs && kws_map[placements_key] != nullptr) {
    placements = CastPyArg2VectorOfPlacement(kws_map[placements_key], 0);
  }
  return placements;
}

std::vector<int> ParseDimsArgs(
    std::unordered_map<std::string, PyObject*> kws_map,
    std::unordered_map<std::string, Py_ssize_t> kw_order_map,
    PyObject* args,
    bool flag_kwargs,
    Py_ssize_t args_num) {
  std::vector<int> dims;
  const std::string& dims_key = "dims";

  if (kw_order_map[dims_key] <= args_num) {
    dims = CastPyArg2VectorOfInt(
        PyTuple_GET_ITEM(args, kw_order_map[dims_key] - 1),
        kw_order_map[dims_key] - 1);
  } else if (flag_kwargs && kws_map[dims_key] != nullptr) {
    dims = CastPyArg2VectorOfInt(kws_map[dims_key], 0);
  }

  return dims;
}

// boolean arguments: zero_copy, stop_gradient, persistable
int ParseBooleanArgs(std::string key,
                     std::unordered_map<std::string, PyObject*> kws_map,
                     std::unordered_map<std::string, Py_ssize_t> kw_order_map,
                     PyObject* args,
                     bool flag_kwargs,
                     Py_ssize_t args_num) {
  int res = -1;

  if (kw_order_map[key] <= args_num) {
    res = static_cast<int>(CastPyArg2AttrBoolean(
        PyTuple_GET_ITEM(args, kw_order_map[key] - 1), kw_order_map[key] - 1));
  } else {
    if (flag_kwargs && kws_map[key] != nullptr) {
      res = static_cast<int>(CastPyArg2AttrBoolean(kws_map[key], 0));
    }
  }
  return res;
}

std::string ParseName(std::unordered_map<std::string, PyObject*> kws_map,
                      std::unordered_map<std::string, Py_ssize_t> kw_order_map,
                      PyObject* args,
                      bool flag_kwargs,
                      Py_ssize_t args_num,
                      std::string unique_name_prefix = "generated_tensor") {
  std::string act_name = "";
  if (kw_order_map["name"] <= args_num) {
    PyObject* name_obj = PyTuple_GET_ITEM(args, kw_order_map["name"] - 1);
    if (name_obj == Py_None) {
      act_name =
          egr::Controller::Instance().GenerateUniqueName(unique_name_prefix);
    } else {
      act_name = CastPyArg2AttrString(name_obj, kw_order_map["name"] - 1);
    }
  } else {
    if (flag_kwargs) {
      if ((kws_map["name"] == NULL) || (kws_map["name"] == Py_None)) {
        act_name =
            egr::Controller::Instance().GenerateUniqueName(unique_name_prefix);
      } else {
        act_name = CastPyArg2AttrString(kws_map["name"], 0);
      }
    } else {
      act_name =
          egr::Controller::Instance().GenerateUniqueName(unique_name_prefix);
    }
  }
  return act_name;
}

// initialize Tensor by PyArray(first argument is PyArray,
// mix args and kwargs) automatically.
void AutoInitTensorByPyArray(TensorObject* py_tensor_ptr,
                             std::unordered_map<std::string, PyObject*> kws_map,
                             PyObject* args,
                             bool flag_kwargs,
                             Py_ssize_t args_num) {
  // The first argument of the Tensor constructor is PyArray,
  // there are 6 arguments to construct the new Tensor,
  // kw_order_map's key is every arguments of the constructor,
  // kw_order_map's value is the position of the arguments respectively.
  // If u want to update this constructor with new arguments,
  // need to update this map and to add or change related code.
  std::unordered_map<std::string, Py_ssize_t> kw_order_map{
      {"value", 1},
      {"place", 2},
      {"persistable", 3},
      {"zero_copy", 4},
      {"name", 5},
      {"stop_gradient", 6}};

  py::object numpy_value = py::object();
  phi::Place place = egr::Controller::Instance().GetExpectedPlace();
  bool persistable = false;
  bool zero_copy = false;
  std::string act_name = "";
  int stop_gradient = -1;

  numpy_value =
      ParsePyArray(kws_map, kw_order_map, args, flag_kwargs, args_num);
  place = ParsePlace(kws_map, kw_order_map, args, flag_kwargs, args_num);
  persistable =
      (1 ==
       ParseBooleanArgs(
           "persistable", kws_map, kw_order_map, args, flag_kwargs, args_num));
  zero_copy =
      (1 ==
       ParseBooleanArgs(
           "zero_copy", kws_map, kw_order_map, args, flag_kwargs, args_num));
  act_name = ParseName(kws_map, kw_order_map, args, flag_kwargs, args_num);
  stop_gradient = ParseBooleanArgs(
      "stop_gradient", kws_map, kw_order_map, args, flag_kwargs, args_num);

  EmptyTensorInitializer(
      py_tensor_ptr, act_name, place, persistable, stop_gradient);
  InitTensorWithNumpyValue(py_tensor_ptr, numpy_value, place, zero_copy);
}

// initialize Tensor by Tensor or phi::DenseTensor (mix args and
// kwargs) automatically.
void AutoInitTensorByTensor(TensorObject* py_tensor_ptr,
                            std::unordered_map<std::string, PyObject*> kws_map,
                            PyObject* args,
                            bool flag_kwargs,
                            Py_ssize_t args_num,
                            bool init_by_egr_tensor = true) {
  // The first argument of the Tensor constructor is Tensor or
  // framework Tensor,
  // there are 6 arguments to construct the new Tensor,
  // kw_order_map's key is every arguments of the constructor,
  // kw_order_map's value is the position of the arguments respectively.
  // If u want to update this constructor with new arguments,
  // need to update this map and to add or change related code.
  std::unordered_map<std::string, Py_ssize_t> kw_order_map{{"value", 1},
                                                           {"place", 2},
                                                           {"name", 3},
                                                           {"dims", 4},
                                                           {"process_mesh", 5},
                                                           {"placements", 6}};

  phi::Place place = egr::Controller::Instance().GetExpectedPlace();
  std::string act_name = "";

  place = ParsePlace(kws_map, kw_order_map, args, flag_kwargs, args_num);
  act_name = ParseName(kws_map, kw_order_map, args, flag_kwargs, args_num);

  if (init_by_egr_tensor) {
    paddle::Tensor src_tensor;
    if (kw_order_map["value"] <= args_num) {
      src_tensor =
          CastPyArg2Tensor(PyTuple_GET_ITEM(args, kw_order_map["value"] - 1),
                           kw_order_map["value"] - 1);
    } else {
      if (flag_kwargs && kws_map["value"] != nullptr) {
        src_tensor = CastPyArg2Tensor(kws_map["value"], 0);
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "The first expected kwargs is {value: Tensor}, "
            "but could not parse the first argument {value: Tensor} "
            "successfully. "
            "Please check your input first and make sure you are on the right "
            "way."));
      }
    }

    ProcessMesh process_mesh = ParseProcessMeshArgs(
        kws_map, kw_order_map, args, flag_kwargs, args_num);

    if (!process_mesh.empty()) {
      auto placements = ParsePlacementsArgs(
          kws_map, kw_order_map, args, flag_kwargs, args_num);

      auto global_dims =
          ParseDimsArgs(kws_map, kw_order_map, args, flag_kwargs, args_num);
      if (!global_dims.empty()) {
        InitDistTensorWithTensor(py_tensor_ptr,
                                 src_tensor,
                                 global_dims,
                                 place,
                                 act_name,
                                 process_mesh,
                                 placements);
      } else {
        InitDistTensorWithTensor(py_tensor_ptr,
                                 src_tensor,
                                 place,
                                 act_name,
                                 process_mesh,
                                 placements);
      }

    } else {
      InitTensorWithTensor(py_tensor_ptr, src_tensor, place, act_name);
    }
  } else {
    // init by framework tensor
    phi::DenseTensor src_tensor;
    if (kw_order_map["value"] <= args_num) {
      src_tensor = CastPyArg2FrameworkTensor(
          PyTuple_GET_ITEM(args, kw_order_map["value"] - 1),
          kw_order_map["value"] - 1);
    } else {
      if (flag_kwargs && kws_map["value"] != nullptr) {
        src_tensor = CastPyArg2FrameworkTensor(kws_map["value"], 0);
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "The first expected arguments is {value: phi::DenseTensor}, "
            "but could not parse the first argument {value: phi::DenseTensor} "
            "successfully. "
            "Please check your input first and make sure you are on the right "
            "way."));
      }
    }
    InitTensorWithFrameworkTensor(py_tensor_ptr, src_tensor, place, act_name);
  }
}

void AutoInitStringTensorByPyArray(
    TensorObject* py_tensor_ptr,
    std::unordered_map<std::string, PyObject*> kws_map,
    PyObject* args,
    bool flag_kwargs,
    Py_ssize_t args_num) {
  // The first argument of the StringTensor constructor is PyArray,
  // there are 4 arguments to construct the new StringTensor,
  // kw_order_map's key is every arguments of the constructor,
  // kw_order_map's value is the position of the arguments respectively.
  // If u want to update this constructor with new arguments,
  // need to update this map and to add or change related code.
  std::unordered_map<std::string, Py_ssize_t> kw_order_map{{"value", 1},
                                                           {"name", 2}};
  py::object numpy_value = py::object();
  phi::Place place = egr::Controller::Instance().GetExpectedPlace();
  std::string act_name = "";

  numpy_value =
      ParsePyArray(kws_map, kw_order_map, args, flag_kwargs, args_num);
  act_name = ParseName(kws_map,
                       kw_order_map,
                       args,
                       flag_kwargs,
                       args_num,
                       "generated_string_tensor");
  EmptyStringTensorInitializer(py_tensor_ptr, act_name, place);
  InitStringTensorWithNumpyValue(py_tensor_ptr, numpy_value);
}

void AutoInitStringTensorByStringTensor(
    TensorObject* py_tensor_ptr,
    std::unordered_map<std::string, PyObject*> kws_map,
    PyObject* args,
    bool flag_kwargs,
    Py_ssize_t args_num) {
  // The first argument of the Tensor constructor is StringTensor,
  // there are 3 arguments to construct the new StringTensor,
  // kw_order_map's key is every arguments of the constructor,
  // kw_order_map's value is the position of the arguments respectively.
  // If u want to update this constructor with new arguments,
  // need to update this map and to add or change related code.
  std::unordered_map<std::string, Py_ssize_t> kw_order_map{{"value", 1},
                                                           {"name", 2}};

  phi::Place place = egr::Controller::Instance().GetExpectedPlace();
  std::string act_name = "";

  act_name = ParseName(kws_map,
                       kw_order_map,
                       args,
                       flag_kwargs,
                       args_num,
                       "generated_string_tensor");
  paddle::Tensor src_tensor;
  if (kw_order_map["value"] <= args_num) {
    src_tensor =
        CastPyArg2Tensor(PyTuple_GET_ITEM(args, kw_order_map["value"] - 1),
                         kw_order_map["value"] - 1);
  } else {
    if (flag_kwargs && kws_map["value"] != nullptr) {
      src_tensor = CastPyArg2Tensor(kws_map["value"], 0);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The first expected kwargs is {value: Tensor}, "
          "but could not parse the first argument {value: Tensor} "
          "successfully. "
          "Please check your input first and make sure you are on the right "
          "way."));
    }
  }
  InitStringTensorWithStringTensor(py_tensor_ptr, src_tensor, place, act_name);
}

PyDoc_STRVAR(  // NOLINT
    TensorDoc,
    R"DOC(Tensor($self, /, value, place, persistable, zero_copy, name, stop_gradient, dims, dtype, type)
--

Tensor is the basic data structure in PaddlePaddle. There are some ways to create a Tensor:

- Use the existing ``data`` to create a Tensor, please refer to :ref:`api_paddle_to_tensor`.
- Create a Tensor with a specified ``shape``, please refer to :ref:`api_paddle_ones`,
  :ref:`api_paddle_zeros`, :ref:`api_paddle_full`.
- Create a Tensor with the same ``shape`` and ``dtype`` as other Tensor, please refer to
  :ref:`api_paddle_ones_like`, :ref:`api_paddle_zeros_like`, :ref:`api_paddle_full_like`.
)DOC");

/** We should have init function with signature:
 * 1.
 * def __init__ ()
 * 2.
 * (should have at least five parameter, five parameters create DenseTensor,
 * seven parameters create DistTensor)
 * def __init__ (
 * ** dtype: paddle::DataType,
 * ** dims: vector<int>,
 * ** name: std::string,
 * ** type: paddle::framework::proto::VarType::DENSE_TENSOR,
 * ** persistable: bool,
 * ** process_mesh: phi::distributed::ProcessMesh,
 * ** placements: std::vector<Placement>)
 * 3. (multi-place)
 * (should have at least one parameter, one parameter equals to case 4, zero
 * parameter equals to case 1)
 * def __init__ (
 * ** value: ndarray,
 * ** place: phi::Place,
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
 * ** global_tensor: Tensor,
 * ** place: phi::Place,
 * ** name: std::string,
 * ** process_mesh: phi::distributed::ProcessMesh,
 * ** placements: std::vector<Placement>)
 * 7. (multi-place)
 * (should have at least one parameter, one parameter equals to case 5, zero
 * parameter equals to case 1.)
 * def __init__ (
 * ** local_tensor: Tensor,
 * ** global_dims: vector<int>,
 * ** name: std::string,
 * ** process_mesh: phi::distributed::ProcessMesh,
 * ** placements: std::vector<Placement>)
 * 8. (multi-place) (should have at least one parameter, one parameter similar
 * to case 5, zero parameter equals to case 1.)
 * def __init__ (
 * ** tensor: FrameworkTensor,
 * ** place: phi::Place,
 * ** name: std::string)
 *  **/
int TensorInit(PyObject* self, PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  SetPythonStack();
  // set a flag to record use kwargs or not
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;

  // all kwargs
  PyObject* kw_zero_copy = nullptr;
  PyObject* kw_persistable = nullptr;
  PyObject* kw_stop_gradient = nullptr;

  PyObject* kw_value = nullptr;  // receive PyArray or Tensor
  PyObject* kw_place = nullptr;
  PyObject* kw_name = nullptr;
  PyObject* kw_dims = nullptr;
  PyObject* kw_dtype = nullptr;
  PyObject* kw_type = nullptr;
  PyObject* kw_process_mesh = nullptr;
  PyObject* kw_placements = nullptr;

  // the keywords argument
  static char* kwlist[] = {const_cast<char*>("value"),  // NOLINT
                           const_cast<char*>("place"),
                           const_cast<char*>("persistable"),
                           const_cast<char*>("zero_copy"),
                           const_cast<char*>("name"),
                           const_cast<char*>("stop_gradient"),
                           const_cast<char*>("dims"),
                           const_cast<char*>("dtype"),
                           const_cast<char*>("type"),
                           const_cast<char*>("process_mesh"),
                           const_cast<char*>("placements"),
                           nullptr};

  // 'O' Store a Python object (without any conversion) in a C object pointer,
  // '|' Indicates that the remaining arguments in the Python argument list are
  // optional.
  // PyArg_ParseTupleAndKeywords can Parse the parameters of a function that
  // takes both positional and keyword parameters into local variables,
  // which enhance case2, case3, case4, case5, case6, case7.
  bool flag_ = PyArg_ParseTupleAndKeywords(args,
                                           kwargs,
                                           "|OOOOOOOOOOO",
                                           kwlist,
                                           &kw_value,
                                           &kw_place,
                                           &kw_persistable,
                                           &kw_zero_copy,
                                           &kw_name,
                                           &kw_stop_gradient,
                                           &kw_dims,
                                           &kw_dtype,
                                           &kw_type,
                                           &kw_process_mesh,
                                           &kw_placements);

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
      {"type", kw_type},
      {"process_mesh", kw_process_mesh},
      {"placements", kw_placements}};

  PADDLE_ENFORCE_EQ(
      flag_,
      true,
      common::errors::PreconditionNotMet(
          "Could not parse args and kwargs successfully, "
          "please check your input first and make"
          "sure you are on the right way. "
          "The expected arguments as follow: ("
          "value, place, persistable, zero_copy, "
          "name, stop_gradient, dims, dtype, type, process_mesh, placements)"));

  PADDLE_ENFORCE_NOT_NULL(
      self,
      common::errors::Fatal(
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
      if (kw_value != nullptr) {
        if (pybind11::detail::npy_api::get().PyArray_Check_(kw_value)) {
          VLOG(6) << "Calling case3's or case4's initializer";
          AutoInitTensorByPyArray(
              py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
          return 0;
        } else if (PyObject_TypeCheck(kw_value, p_tensor_type)) {
          VLOG(6) << "Calling case5's or case6's or case7's initializer";
          AutoInitTensorByTensor(
              py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
          return 0;
        } else if (PyObject_TypeCheck(kw_value, g_framework_tensor_pytype)) {
          VLOG(6) << "Calling case8's initializer.";
          AutoInitTensorByTensor(py_tensor_ptr,
                                 kws_map,
                                 args,
                                 flag_kwargs,
                                 args_num,
                                 /* false means not init by egr tensor*/ false);
          return 0;
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "Could not parse the first keyword argument successfully, "
              "the first keyword argument is value, but it should be PyArray "
              "or Tensor or phi::DenseTensor. "
              "Please check your input first and make sure you are on the "
              "right way."));
        }
      } else if (kw_dtype != nullptr &&
                 (PyObject_TypeCheck(kw_dtype, g_data_type_pytype) ||
                  PyObject_TypeCheck(kw_dtype, g_vartype_pytype))) {
        // TODO(jeff41404): until the default value of FLAGS_deable_ir_appi is
        // True, can delete `PyObject_TypeCheck(kw_dtype, g_vartype_pytype)`
        // Retain it during the transitional period.
        VLOG(6) << "Calling case2's initializer";

        PADDLE_ENFORCE_NOT_NULL(
            kw_dims,
            common::errors::InvalidArgument(
                "Calling __init__ of Eager Tensor with NULL dims is "
                "forbidden. Please check your code and make sure you new a "
                "dims before calling this constructor."));

        PADDLE_ENFORCE_NOT_NULL(
            kw_name,
            common::errors::InvalidArgument(
                "Calling __init__ of Eager Tensor with NULL name is "
                "forbidden. Please check your code and make sure you new a "
                "name before calling this constructor."));

        PADDLE_ENFORCE_NOT_NULL(
            kw_dtype,
            common::errors::InvalidArgument(
                "Calling __init__ of Eager Tensor with NULL dtype is "
                "forbidden. Please check your code and make sure you new a "
                "dtype before calling this constructor."));

        PADDLE_ENFORCE_NOT_NULL(
            kw_persistable,
            common::errors::InvalidArgument(
                "Calling __init__ of Eager Tensor with NULL persistable is "
                "forbidden. Please check your code and make sure you new a "
                "persistable before calling this constructor."));

        paddle::DataType dtype = CastPyArg2DataType(kw_dtype, "TensorInit", 0);
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

        ProcessMesh* process_mesh_ptr = nullptr;
        if (kw_process_mesh != nullptr) {
          ProcessMesh process_mesh = CastPyArg2ProcessMesh(kw_process_mesh, 0);
          process_mesh_ptr = &process_mesh;
        }

        Placements* placements_ptr = nullptr;
        if (kw_placements != nullptr) {
          Placements placements = CastPyArg2VectorOfPlacement(kw_placements, 0);
          placements_ptr = &placements;
        }

        EmptyTensorInitializer(py_tensor_ptr,
                               act_name,
                               egr::Controller::Instance().GetExpectedPlace(),
                               persistable,
                               /* stop_gradient */ -1,
                               dtype,
                               dims,
                               var_type,
                               process_mesh_ptr,
                               placements_ptr);

        return 0;
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "We not only support construct Tensor from numpy value "
            "or tensor(Tensor or phi::DenseTensor) "
            "with python kwargs by this initializer, "
            "but also even support dtype to init a empty Tensor. "
            "Please check your input first and make sure you call the existed "
            "constructor."));
      }
    }
  } else if (args_num == (Py_ssize_t)1 || args_num == (Py_ssize_t)2 ||
             args_num == (Py_ssize_t)3) {
    // 1 to 3 position args, remaining arguments are kwargs
    PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
    if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
      VLOG(6) << "Calling case3's or case4's initializer.";
      AutoInitTensorByPyArray(
          py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
      return 0;
    } else if (PyObject_TypeCheck(arg0_ptr, p_tensor_type)) {
      VLOG(6) << "Calling case5's or case6's or case7's initializer.";
      AutoInitTensorByTensor(
          py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
      return 0;
    } else if (PyObject_TypeCheck(arg0_ptr, g_framework_tensor_pytype)) {
      VLOG(6) << "Calling case8's initializer.";
      AutoInitTensorByTensor(py_tensor_ptr,
                             kws_map,
                             args,
                             flag_kwargs,
                             args_num,
                             /* false means not init by egr tensor*/ false);
      return 0;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "We support construct Tensor from numpy value "
          "or tensor(Tensor or phi::DenseTensor) "
          "with python args and kwargs by this initializer, "
          "but the first argument should be PyArray or Tensor or "
          "phi::DenseTensor. "
          "Please check your input first and make sure you call the existed "
          "constructor."));
    }
  } else if (args_num == (Py_ssize_t)4) {
    // 4 position args, remaining arguments are kwargs
    PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
    if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
      VLOG(6) << "Calling case3's or case4's initializer.";
      AutoInitTensorByPyArray(
          py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
      return 0;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Incompatible constructor arguments, "
          "there are 4 position args and remaining arguments arg kwargs,"
          "but the first position args should be PyArray. "
          "Please check your code and make sure the first position args is "
          "PyArray."));
    }
  } else if (args_num == (Py_ssize_t)5) {
    if (!flag_kwargs) {
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      // TODO(jeff41404): until the default value of FLAGS_deable_ir_appi is
      // True, can delete `PyObject_TypeCheck(arg0_ptr, g_vartype_pytype)`
      // Retain it during the transitional period.
      if (PyObject_TypeCheck(arg0_ptr, g_data_type_pytype) ||
          PyObject_TypeCheck(arg0_ptr, g_vartype_pytype)) {
        VLOG(6) << "Calling case2's initializer.";
        paddle::DataType dtype = CastPyArg2DataType(arg0_ptr, "TensorInit", 0);
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
        EmptyTensorInitializer(py_tensor_ptr,
                               act_name,
                               egr::Controller::Instance().GetExpectedPlace(),
                               persistable,
                               -1,
                               dtype,
                               dims,
                               var_type);
        return 0;
      } else if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
        VLOG(6) << "Calling case3's initializer.";
        AutoInitTensorByPyArray(
            py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
        return 0;
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Incompatible constructor arguments, "
            "there are only 5 position args,"
            "but the first position args should be PyArray or dtype. "
            "Please check your code and make sure you call the existed "
            "constructor."));
      }
    } else {  // five position args, remaining arguments are kwargs
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
        VLOG(6) << "Calling case3's or case4's initializer";
        AutoInitTensorByPyArray(
            py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
        return 0;
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Incompatible constructor arguments, "
            "there are 5 position args and remaining arguments are kwargs,"
            "but the first position args should be PyArray. "
            "Please check your code and make sure the first position args is "
            "PyArray."));
      }
    }
  } else if (args_num == (Py_ssize_t)6) {
    if (!flag_kwargs) {
      // case 3
      VLOG(6) << "Calling case3's initializer.";
      AutoInitTensorByPyArray(
          py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
      return 0;
    } else {  // six position args, remaining arguments are kwargs, but this
              // is not a right way
      PADDLE_THROW(common::errors::InvalidArgument(
          "Incompatible constructor arguments, "
          "there are 6 position args and the remaining arguments are kwargs. "
          "Please check your code and make sure the first position args is "
          "PyArray."));
    }
  } else if (args_num == (Py_ssize_t)7) {
    if (!flag_kwargs) {
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      // TODO(jeff41404): until the default value of FLAGS_deable_ir_appi is
      // True, can delete `PyObject_TypeCheck(arg0_ptr, g_vartype_pytype)`
      // Retain it during the transitional period.
      if (PyObject_TypeCheck(arg0_ptr, g_data_type_pytype) ||
          PyObject_TypeCheck(arg0_ptr, g_vartype_pytype)) {
        VLOG(6) << "Calling case2's initializer.";
        paddle::DataType dtype = CastPyArg2DataType(arg0_ptr, "TensorInit", 0);
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
        ProcessMesh process_mesh =
            CastPyArg2ProcessMesh(PyTuple_GET_ITEM(args, 5), 5);
        Placements placements =
            CastPyArg2VectorOfPlacement(PyTuple_GET_ITEM(args, 6), 6);
        EmptyTensorInitializer(py_tensor_ptr,
                               act_name,
                               egr::Controller::Instance().GetExpectedPlace(),
                               persistable,
                               -1,
                               dtype,
                               dims,
                               var_type,
                               &process_mesh,
                               &placements);
        return 0;
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Incompatible constructor arguments, "
            "there are only 7 position args,"
            "but the first position args should be dtype. "
            "Please check your code and make sure you call the existed "
            "constructor."));
      }
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Incompatible constructor arguments, "
          "there are 7 position args and remaining arguments are kwargs,"
          "Please check your code and make sure you call the existed "
          "constructor."));
    }
  } else {
    PADDLE_THROW(common::errors::Fatal(
        "Can't not find expected num of args, please check your call, and "
        "make sure u call the existed constructor."));
  }

  return -1;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

/** We should have init function with signature:
 * 1.
 * def __init__ ()
 *
 * 2.
 * def __init__ (
 * ** dims: vector<int>,
 * ** name: std::string)
 *
 * 3.
 * (should have at least one parameter, one parameter equals to case 4, zero
 * parameter equals to case 1)
 * def __init__ (
 * ** value: ndarray,
 * ** zero_copy: bool,
 * ** name: std::string)
 *
 * 4.
 * def __init__ (
 * ** value: ndarray)
 *
 * 5.
 * def __init__ (
 * ** tensor: Tensor)
 *
 * 6.
 * (should have at least one parameter, one parameter equals to case 5, zero
 * parameter equals to case 1.)
 * def __init__ (
 * ** tensor: Tensor,
 * ** name: std::string)
 * **/
int StringTensorInit(PyObject* self, PyObject* args, PyObject* kwargs) {
  // set a flag to record use kwargs or not
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;

  // all kwargs
  PyObject* kw_zero_copy = nullptr;

  PyObject* kw_value = nullptr;  // receive PyArray or Tensor
  PyObject* kw_name = nullptr;
  PyObject* kw_dims = nullptr;

  // the keywords argument
  static char* kwlist[] = {const_cast<char*>("value"),  // NOLINT
                           const_cast<char*>("zero_copy"),
                           const_cast<char*>("name"),
                           const_cast<char*>("dims"),
                           nullptr};
  // 'O' Store a Python object (without any conversion) in a C object pointer,
  // '|' Indicates that the remaining arguments in the Python argument list are
  // optional.
  // PyArg_ParseTupleAndKeywords can Parse the parameters of a function that
  // takes both positional and keyword parameters into local variables,
  // which enhance case1, case2, case3, case4, case 5, case 6.
  bool flag_ = PyArg_ParseTupleAndKeywords(args,
                                           kwargs,
                                           "|OOOO",
                                           kwlist,
                                           &kw_value,
                                           &kw_zero_copy,
                                           &kw_name,
                                           &kw_dims);

  // helper map
  std::unordered_map<std::string, PyObject*> kws_map{
      {"value", kw_value},
      {"zero_copy", kw_zero_copy},
      {"name", kw_name},
      {"dims", kw_dims}};

  PADDLE_ENFORCE_EQ(flag_,
                    true,
                    common::errors::PreconditionNotMet(
                        "Could not parse args and kwargs successfully, "
                        "please check your input first and make"
                        "sure you are on the right way. "
                        "The expected arguments as follow: ("
                        "value, zero_copy, name, dims)"));

  PADDLE_ENFORCE_NOT_NULL(
      self,
      common::errors::Fatal(
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
      VLOG(6) << "Calling case1's string initializer.";
      EmptyStringTensorInitializer(
          py_tensor_ptr,
          egr::Controller::Instance().GenerateUniqueName(
              "generated_string_tensor"),
          egr::Controller::Instance().GetExpectedPlace());
      return 0;
    } else {
      if (kw_value != nullptr) {
        if (pybind11::detail::npy_api::get().PyArray_Check_(kw_value)) {
          VLOG(6) << "Calling case3's or case4's string initializer";
          AutoInitStringTensorByPyArray(
              py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
          return 0;
        } else if (PyObject_TypeCheck(kw_value, p_string_tensor_type)) {
          VLOG(6) << "Calling case5's or case6's string initializer";
          AutoInitStringTensorByStringTensor(
              py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
          return 0;
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "Could not parse the first keyword argument successfully, "
              "the first keyword argument is value, but it should be PyArray "
              "or StringTensor."
              "Please check your input first and make sure you are on the "
              "right way."));
        }
      } else if (kw_dims != nullptr) {
        VLOG(6) << "Calling case2's string initializer.";
        std::unordered_map<std::string, Py_ssize_t> kw_order_map{{"dims", 1},
                                                                 {"name", 2}};

        std::vector<int> dims = CastPyArg2VectorOfInt(kw_dims, 0);
        std::string act_name = ParseName(kws_map,
                                         kw_order_map,
                                         args,
                                         flag_kwargs,
                                         args_num,
                                         "generated_string_tensor");
        EmptyStringTensorInitializer(
            py_tensor_ptr,
            act_name,
            egr::Controller::Instance().GetExpectedPlace(),
            dims);
        return 0;
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "We not only support construct Tensor from numpy value "
            "or StringTensor with python kwargs by this initializer, "
            "but also even support dtype to init a empty StringTensor. "
            "Please check your input first and make sure you call the existed "
            "constructor."));
      }
    }
  } else if (args_num == (Py_ssize_t)1) {  // case 3 ~ 6
    // 1 position args, remaining arguments are kwargs
    PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
    if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
      VLOG(6) << "Calling case3's or case4's string initializer.";
      AutoInitStringTensorByPyArray(
          py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
      return 0;
    } else if (PyObject_TypeCheck(arg0_ptr, p_string_tensor_type)) {
      VLOG(6) << "Calling case5's or case6's string initializer.";
      AutoInitStringTensorByStringTensor(
          py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
      return 0;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Could not parse the first keyword argument successfully, "
          "the first keyword argument is value, but it should be PyArray "
          "or StringTensor."
          "Please check your input first and make sure you are on the "
          "right way."));
    }
  } else if (args_num == (Py_ssize_t)2) {  // case 2
    // 2 position args
    if (!flag_kwargs) {
      PyObject* arg0_ptr = PyTuple_GET_ITEM(args, 0);
      if (PyObject_TypeCheck(arg0_ptr, p_string_tensor_type)) {
        VLOG(6) << "Calling case6's string initializer.";
        AutoInitStringTensorByStringTensor(
            py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
        return 0;
      } else if (pybind11::detail::npy_api::get().PyArray_Check_(arg0_ptr)) {
        VLOG(6) << "Calling case3's string initializer.";
        AutoInitStringTensorByPyArray(
            py_tensor_ptr, kws_map, args, flag_kwargs, args_num);
        return 0;
      } else {
        VLOG(6) << "Calling case2's string initializer.";
        std::vector<int> dims = CastPyArg2VectorOfInt(arg0_ptr, 0);
        std::string act_name = "";
        PyObject* name_obj = PyTuple_GET_ITEM(args, 1);
        if (name_obj == Py_None) {
          act_name = egr::Controller::Instance().GenerateUniqueName(
              "generated_string_tensor");
        } else {
          act_name = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 1), 1);
        }
        EmptyStringTensorInitializer(
            py_tensor_ptr,
            act_name,
            egr::Controller::Instance().GetExpectedPlace(),
            dims);
        return 0;
      }
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "Can't not find expected num of args, please check your call, and "
          "make sure u call the existed constructor."));
    }
  }
  return 1;
}

void AddPyMethodDefs(std::vector<PyMethodDef>* vector, PyMethodDef* methods) {
  if (!vector->empty()) {
    // remove nullptr terminator
    vector->pop_back();
  }
  while (true) {
    vector->push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

static void TensorDealloc(TensorObject* self) {
  if (self->weakrefs != nullptr)
    PyObject_ClearWeakRefs(reinterpret_cast<PyObject*>(self));
  self->tensor.~Tensor();
  Py_XDECREF(self->dict);
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

extern struct PyGetSetDef variable_properties[];                // NOLINT
extern struct PyGetSetDef string_tensor_variable_properties[];  // NOLINT

extern PyMethodDef variable_methods[];                // NOLINT
extern PyMethodDef math_op_patch_methods[];           // NOLINT
extern PyMethodDef string_tensor_variable_methods[];  // NOLINT

PyNumberMethods number_methods;
PySequenceMethods sequence_methods;
PyMappingMethods mapping_methods;

void BindEager(pybind11::module* module) {
  auto m = module->def_submodule("eager");

  static std::vector<PyMethodDef> methods;
  AddPyMethodDefs(&methods, variable_methods);
  AddPyMethodDefs(&methods, math_op_patch_methods);

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
  type->tp_methods = methods.data();
  type->tp_getset = variable_properties;
  type->tp_init = TensorInit;
  type->tp_new = TensorNew;
  type->tp_doc = TensorDoc;
  type->tp_weaklistoffset = offsetof(TensorObject, weakrefs);
  Py_INCREF(&PyBaseObject_Type);
  type->tp_base = reinterpret_cast<PyTypeObject*>(&PyBaseObject_Type);
  type->tp_flags |=
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;  // NOLINT
  type->tp_dictoffset = offsetof(TensorObject, dict);
#if PY_VERSION_HEX >= 0x03050000
  type->tp_as_async = &heap_type->as_async;
#endif
  p_tensor_type = type;

  if (PyType_Ready(type) < 0) {
    PADDLE_THROW(
        common::errors::Fatal("Init Paddle error in BindEager(PyType_Ready)."));
    return;
  }

  Py_INCREF(type);
  if (PyModule_AddObject(m.ptr(), "Tensor", reinterpret_cast<PyObject*>(type)) <
      0) {
    Py_DECREF(type);
    Py_DECREF(m.ptr());
    PADDLE_THROW(common::errors::Fatal(
        "Init Paddle error in BindEager(PyModule_AddObject)."));
    return;
  }

  BindFunctions(m.ptr());
  BindEagerPyLayer(m.ptr());
  BindEagerOpFunctions(&m);
}

void BindEagerStringTensor(pybind11::module* module) {
  auto m = module->def_submodule("eager");

  auto heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  heap_type->ht_name = ToPyObject("StringTensor");
  heap_type->ht_qualname = ToPyObject("StringTensor");
  auto type = &heap_type->ht_type;
  type->tp_name = "StringTensor";
  type->tp_basicsize = sizeof(TensorObject);
  type->tp_dealloc = (destructor)TensorDealloc;
  type->tp_as_number = &number_methods;
  type->tp_as_sequence = &sequence_methods;
  type->tp_as_mapping = &mapping_methods;
  type->tp_methods = string_tensor_variable_methods;
  type->tp_getset = string_tensor_variable_properties;
  type->tp_init = StringTensorInit;
  type->tp_new = TensorNew;
  Py_INCREF(&PyBaseObject_Type);
  type->tp_base = reinterpret_cast<PyTypeObject*>(&PyBaseObject_Type);
  type->tp_flags |=
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;  // NOLINT
  type->tp_dictoffset = offsetof(TensorObject, dict);
#if PY_VERSION_HEX >= 0x03050000
  type->tp_as_async = &heap_type->as_async;
#endif
  p_string_tensor_type = type;

  if (PyType_Ready(type) < 0) {
    PADDLE_THROW(
        common::errors::Fatal("Init Paddle error in BindEager(PyType_Ready)."));
    return;
  }

  Py_INCREF(type);
  if (PyModule_AddObject(
          m.ptr(), "StringTensor", reinterpret_cast<PyObject*>(type)) < 0) {
    Py_DECREF(type);
    Py_DECREF(m.ptr());
    PADDLE_THROW(common::errors::Fatal(
        "Init Paddle error in BindEagerStringTensor(PyModule_AddObject)."));
    return;
  }
}

}  // namespace paddle::pybind
