/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/imperative.h"

#include <Python.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/scope_guard.h"
#include "paddle/fluid/imperative/all_reduce.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/bkcl_context.h"
#include "paddle/fluid/imperative/cncl_context.h"
#include "paddle/fluid/imperative/data_loader.h"
#include "paddle/fluid/imperative/gloo_context.h"
#include "paddle/fluid/imperative/hccl_context.h"
#include "paddle/fluid/imperative/heter_ccl_context.h"
#include "paddle/fluid/imperative/hooks.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/nccl_context.h"
#include "paddle/fluid/imperative/partial_grad_engine.h"
#include "paddle/fluid/imperative/profiler.h"
#include "paddle/fluid/imperative/py_layer_fwd.h"
#include "paddle/fluid/imperative/reducer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/pybind/cuda_streams_py.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/op_function.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/fluid/pybind/slice_utils.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/fluid/pybind/uva_utils.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/phi/core/type_defs.h"

namespace paddle {
namespace pybind {

std::atomic<int> VarBaseUniqueNameID{0};
PyTypeObject *g_varbase_pytype = nullptr;

namespace py = ::pybind11;

template <typename T>
static T PyObjectCast(PyObject *obj) {
  try {
    return py::cast<T>(py::handle(obj));
  } catch (py::cast_error &) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Python object is not type of %s", typeid(T).name()));
  }
}

class PyVariableWrapperHook : public imperative::VariableWrapperHook {
 public:
  explicit PyVariableWrapperHook(PyObject *func) : py_func_(func) {
    Py_INCREF(py_func_);
  }

  ~PyVariableWrapperHook() {
    py::gil_scoped_acquire gil;
    Py_DECREF(py_func_);
  }

  std::shared_ptr<imperative::VariableWrapper> operator()(
      const std::shared_ptr<imperative::VariableWrapper> &var) override {
    py::gil_scoped_acquire gil;
    VLOG(3) << "Call PyVariableWrapperHook for var " << var->Name();

    // 1. unpack temp VarBase from VariableWrapper
    std::shared_ptr<imperative::VarBase> tmp_varbase =
        std::make_shared<imperative::VarBase>(var);

    // 2. call hook and return
    PyObject *res = nullptr;
    try {
      res = PyObject_CallFunctionObjArgs(
          py_func_, py::cast(tmp_varbase).ptr(), nullptr);
    } catch (platform::EnforceNotMet &e) {
      throw std::move(e);
    } catch (std::exception &e) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Hook function of Tensor raises an exception: %s.", e.what()));
    } catch (...) {
      PADDLE_THROW(platform::errors::Fatal(
          "Hook function of Tensor raises an unknown exception."));
    }

    PADDLE_ENFORCE_NOT_NULL(res,
                            platform::errors::Unavailable(
                                "Hook function of Tensor return a nullptr."));
    if (res == Py_None) {
      return var;
    }

    auto res_varbase = PyObjectCast<std::shared_ptr<imperative::VarBase>>(res);
    // Here the reference count of `res` is 2, so we decreases the reference
    // count manually to avoid memory leaks
    Py_DECREF(res);
    return res_varbase->SharedVar();
  }

 private:
  PyObject *py_func_;
};

static const platform::Place PyObjectToPlace(const py::object &place_obj) {
  if (py::isinstance<platform::CPUPlace>(place_obj)) {
    return place_obj.cast<platform::CPUPlace>();
  } else if (py::isinstance<platform::CUDAPlace>(place_obj)) {
    return place_obj.cast<platform::CUDAPlace>();
  } else if (py::isinstance<platform::XPUPlace>(place_obj)) {
    return place_obj.cast<platform::XPUPlace>();
  } else if (py::isinstance<platform::CUDAPinnedPlace>(place_obj)) {
    return place_obj.cast<platform::CUDAPinnedPlace>();
  } else if (py::isinstance<platform::NPUPlace>(place_obj)) {
    return place_obj.cast<platform::NPUPlace>();
  } else if (py::isinstance<platform::IPUPlace>(place_obj)) {
    return place_obj.cast<platform::IPUPlace>();
  } else if (py::isinstance<platform::Place>(place_obj)) {
    return place_obj.cast<platform::Place>();
  } else if (py::isinstance<platform::MLUPlace>(place_obj)) {
    return place_obj.cast<platform::MLUPlace>();
  } else if (py::isinstance<platform::CustomPlace>(place_obj)) {
    return place_obj.cast<platform::CustomPlace>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place should be one of "
        "Place/CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace/NPUPlace/IPUPlace/"
        "MLUPlace/CustomPlace"));
  }
}

// only initialize varbase, but not its tensor.
static void InitVarBaseOnly(imperative::VarBase *self,
                            const std::string &name,
                            bool persistable = false,
                            int stop_gradient = -1) {
  auto name_ = name == "" ? imperative::GetCurrentTracer()->GenerateUniqueName(
                                "generated_tensor")
                          : name;

  VLOG(5) << "Init Tensor as: / name: " << name_
          << " / persistable: " << persistable
          << " / stop_gradient: " << stop_gradient;
  new (self) imperative::VarBase(name_);
  if (stop_gradient != -1) {
    self->SetOverridedStopGradient(stop_gradient);
  }
  self->SetPersistable(persistable);
  self->SetType(framework::proto::VarType::LOD_TENSOR);
}

// initialize varbase and its tensor.
static void InitVarBaseAndTensor(imperative::VarBase *self,
                                 const py::array &array,
                                 const platform::Place &place,
                                 const std::string &name,
                                 bool persistable = false,
                                 bool zero_copy = false,
                                 int stop_gradient = -1) {
  InitVarBaseOnly(self, name, persistable, stop_gradient);
  auto *tensor = self->MutableVar()->GetMutable<framework::LoDTensor>();
  VLOG(4) << "zero_copy: " << zero_copy;
  if (platform::is_cpu_place(place)) {
    SetTensorFromPyArray<platform::CPUPlace>(tensor, array, place, zero_copy);
  } else if (platform::is_xpu_place(place)) {
    SetTensorFromPyArray<platform::XPUPlace>(tensor, array, place, zero_copy);
  } else if (platform::is_gpu_place(place)) {
    SetTensorFromPyArray<platform::CUDAPlace>(tensor, array, place, zero_copy);
  } else if (platform::is_cuda_pinned_place(place)) {
    SetTensorFromPyArray<platform::CUDAPinnedPlace>(
        tensor, array, place, zero_copy);
  } else if (platform::is_npu_place(place)) {
    SetTensorFromPyArray<platform::NPUPlace>(tensor, array, place, zero_copy);
  } else if (platform::is_ipu_place(place)) {
    SetTensorFromPyArray<platform::IPUPlace>(tensor, array, place, zero_copy);
  } else if (platform::is_mlu_place(place)) {
    SetTensorFromPyArray<platform::MLUPlace>(tensor, array, place, zero_copy);
  } else if (platform::is_custom_place(place)) {
    SetTensorFromPyArray<platform::CustomPlace>(
        tensor, array, place, zero_copy);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place should be one of "
        "CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace/NPUPlace/IPUPlace/"
        "MLUPlace"));
  }
  self->SetDataType(framework::TransToProtoVarType(tensor->dtype()));
}

static void InitVarBaseFromNumpyWithKwargs(imperative::VarBase *self,
                                           const py::kwargs &kwargs) {
  VLOG(4) << "Init VarBase from kwargs: ";
  auto persistable = kwargs.contains("persistable")
                         ? kwargs["persistable"].cast<bool>()
                         : false;
  auto zero_copy =
      kwargs.contains("zero_copy") ? kwargs["zero_copy"].cast<bool>() : false;
  auto name = kwargs.contains("name") ? kwargs["name"].cast<std::string>() : "";
  auto stop_gradient = kwargs.contains("stop_gradient")
                           ? kwargs["stop_gradient"].cast<int>()
                           : -1;
  auto default_place = imperative::GetCurrentTracer()->ExpectedPlace();

  if (kwargs.contains("value")) {
    auto array = kwargs["value"].cast<py::array>();
    // place is only used when array is given, otherwise, it is meaningless and
    // ignored
    auto place = kwargs.contains("place") ? PyObjectToPlace(kwargs["place"])
                                          : default_place;
    InitVarBaseAndTensor(
        self, array, place, name, persistable, zero_copy, stop_gradient);
  } else {
    InitVarBaseOnly(self, name, persistable, stop_gradient);
  }
}

template <typename P>
static void InitVarBaseFromNumpyWithArg(imperative::VarBase *self,
                                        const py::array &array,
                                        const P &place,
                                        bool persistable = false,
                                        bool zero_copy = false,
                                        std::string name = "",
                                        int stop_gradient = -1) {
  VLOG(4) << "Init VarBase from Arg: ";
  // 0: self, 1: value, 2: place, 3: persistable, 4: zero_copy, 5: name , 6:
  // stop_gradient
  if (name == "") {
    name =
        imperative::GetCurrentTracer()->GenerateUniqueName("generated_tensor");
  }
  VLOG(5) << "Init Tensor as: / name: " << name
          << " / persistable: " << persistable << " / zero_copy: " << zero_copy
          << " / stop_gradient: " << stop_gradient << " / at " << place;
  new (self) imperative::VarBase(name);
  self->SetPersistable(persistable);
  auto *tensor = self->MutableVar()->GetMutable<framework::LoDTensor>();
  if (stop_gradient != -1) {
    self->SetOverridedStopGradient(stop_gradient);
  }
  SetTensorFromPyArray<P>(tensor, array, place, zero_copy);
  self->SetType(framework::proto::VarType::LOD_TENSOR);
  self->SetDataType(framework::TransToProtoVarType(tensor->dtype()));
}

static void InitVarBaseFromNumpyWithArgDefault(imperative::VarBase *self,
                                               const py::array &array) {
  auto place = imperative::GetCurrentTracer()->ExpectedPlace();
  VLOG(4) << "Init VarBase from numpy at " << place;
  InitVarBaseAndTensor(self, array, place, "");
}

static void InitVarBaseFromTensorWithArgDefault(imperative::VarBase *self,
                                                const phi::DenseTensor &tensor,
                                                const std::string &name) {
  VLOG(4) << "Init VarBase";
  auto place = imperative::GetCurrentTracer()->ExpectedPlace();
  auto name_ = name == "" ? imperative::GetCurrentTracer()->GenerateUniqueName(
                                "generated_tensor")
                          : name;
  new (self) imperative::VarBase(name_);
  self->SetPersistable(false);
  self->SetType(framework::proto::VarType::LOD_TENSOR);
  self->SetDataType(framework::TransToProtoVarType(tensor.dtype()));
  auto *new_tensor = self->MutableVar()->GetMutable<framework::LoDTensor>();
  // Same place，share data directly
  if (place == tensor.place()) {
    new_tensor->ShareDataWith(tensor);
    VLOG(4) << "Same place, do ShareDataWith";
  } else {
    framework::TensorCopy(tensor, place, new_tensor);
    VLOG(4) << "Different place, do TensorCopy";
  }
}

template <typename P>
static void InitVarBaseFromTensorWithArg(imperative::VarBase *self,
                                         const phi::DenseTensor &tensor,
                                         const P &place,
                                         const std::string &name) {
  VLOG(4) << "Init VarBase";
  auto name_ = name == "" ? imperative::GetCurrentTracer()->GenerateUniqueName(
                                "generated_tensor")
                          : name;
  new (self) imperative::VarBase(name_);
  self->SetPersistable(false);
  self->SetType(framework::proto::VarType::LOD_TENSOR);
  self->SetDataType(framework::TransToProtoVarType(tensor.dtype()));
  auto *new_tensor = self->MutableVar()->GetMutable<framework::LoDTensor>();
  // Same place，share data directly
  if (platform::is_same_place(place, tensor.place())) {
    new_tensor->ShareDataWith(tensor);
    VLOG(4) << "Same place, do ShareDataWith";
  } else {
    framework::TensorCopy(tensor, place, new_tensor);
    VLOG(4) << "Different place, do TensorCopy";
  }
}

static std::string GetTypeName(const imperative::VarBase &var) {
  if (var.Type() == framework::proto::VarType::RAW) {
    return "RAW";
  } else if (!var.Var().IsInitialized()) {
    return "nullptr";
  } else {
    return framework::ToTypeName(var.Var().Type());
  }
}

Py_ssize_t GetSliceIndexFromPyObject(PyObject *obj) {
  if (py::isinstance<imperative::VarBase>(obj)) {
    VLOG(6) << "Call GetSliceIndexFromTensor in Imperative";
    return GetSliceIndexFromTensor(
        py::cast<std::shared_ptr<imperative::VarBase>>(obj)
            ->Var()
            .Get<framework::LoDTensor>());
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "We should only get paddle::experimental::Tensor or VarBase in this "
        "method, when you reach this means we got another type index."));
  }
}

bool PyCheckTensor(PyObject *obj) {
  return py::isinstance<imperative::VarBase>(obj);
}
using PyNameVarBaseMap = std::unordered_map<std::string, py::handle>;

// NOTE(zjl): py::handle is a very light wrapper of PyObject *.
// Unlike py::object, py::handle does not change reference count of PyObject *.
static std::vector<std::shared_ptr<imperative::VarBase>>
GetVarBaseListFromPyHandle(const py::handle &handle) {
  PyObject *py_obj = handle.ptr();  // get underlying PyObject
  // Python None is not nullptr in C++!
  if (!py_obj || py_obj == Py_None) {
    return {};
  }

  std::vector<std::shared_ptr<imperative::VarBase>> result;

  if (PyList_Check(py_obj)) {  // List of VarBase
    size_t len = PyList_GET_SIZE(py_obj);
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      PyObject *py_ivar = PyList_GET_ITEM(py_obj, i);
      PADDLE_ENFORCE_NOT_NULL(
          py_ivar, platform::errors::InvalidArgument("Python Object is NULL"));
      result.emplace_back(
          PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar));
    }
  } else if (PyTuple_Check(py_obj)) {  // Tuple of VarBase
    size_t len = PyTuple_GET_SIZE(py_obj);
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      PyObject *py_ivar = PyTuple_GET_ITEM(py_obj, i);
      PADDLE_ENFORCE_NOT_NULL(
          py_ivar, platform::errors::InvalidArgument("Python Object is NULL"));
      result.emplace_back(
          PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar));
    }
  } else {  // VarBase
    result.emplace_back(
        PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_obj));
  }

  return result;
}

static imperative::NameVarBaseMap ConvertToNameVarBaseMap(
    const PyNameVarBaseMap &map) {
  imperative::NameVarBaseMap result;
  for (auto &pair : map) {
    auto var_vec = GetVarBaseListFromPyHandle(pair.second);
    if (!var_vec.empty()) {
      result.emplace(pair.first, std::move(var_vec));
    }
  }

  PADDLE_ENFORCE_EQ(
      PyErr_Occurred(),
      nullptr,
      platform::errors::InvalidArgument(py::str(py::handle(PyErr_Occurred()))));
  return result;
}

paddle::imperative::NameTensorMap ConvertToNameTensorMap(
    const PyNameVarBaseMap &map) {
  paddle::imperative::NameTensorMap result;
  for (auto &pair : map) {
    auto var_vec = CastPyArg2VectorOfTensor(pair.second.ptr(), 0);
    if (!var_vec.empty()) {
      // change vector<Tensor> -> vector<shared_ptr<Tensor>>
      std::vector<std::shared_ptr<egr::EagerVariable>> dst_var_vec;
      for (auto &v : var_vec) {
        dst_var_vec.emplace_back(
            std::make_shared<egr::EagerVariable>(std::move(v)));
      }
      result.emplace(pair.first, std::move(dst_var_vec));
    }
  }

  PADDLE_ENFORCE_EQ(
      PyErr_Occurred(),
      nullptr,
      platform::errors::InvalidArgument(py::str(py::handle(PyErr_Occurred()))));
  return result;
}

template <typename P>
static void VarBaseCopy(std::shared_ptr<imperative::VarBase> &src,  // NOLINT
                        imperative::VarBase &dst,                   // NOLINT
                        const P &dst_device,
                        const bool blocking) {
  if (dst.SharedVar()->IsEmpty()) {
    VLOG(3) << "deep copy Variable from " << src->Name() << " to "
            << dst.Name();
    dst.SetPersistable(src->Persistable());
    dst.SetDataType(src->DataType());
    dst.SetType(src->Type());
    dst.SetOverridedStopGradient(src->OverridedStopGradient());
    if (!src->SharedVar()->IsEmpty()) {
      if (src->Var().IsType<framework::LoDTensor>()) {
        auto &src_tensor = src->Var().Get<framework::LoDTensor>();
        auto *dst_tensor = dst.MutableVar()->GetMutable<framework::LoDTensor>();
        dst_tensor->set_lod(src_tensor.lod());
        framework::TensorCopy(src_tensor, dst_device, dst_tensor);
        if (blocking) {
          platform::DeviceContextPool::Instance().Get(dst_device)->Wait();
          auto src_device = src_tensor.place();
          if (!(src_device == dst_device)) {
            platform::DeviceContextPool::Instance().Get(src_device)->Wait();
          }
        }
      } else if (src->Var().IsType<phi::SelectedRows>()) {
        auto &src_selected_rows = src->Var().Get<phi::SelectedRows>();
        auto *dst_selected_rows =
            dst.MutableVar()->GetMutable<phi::SelectedRows>();
        dst_selected_rows->set_height(src_selected_rows.height());
        dst_selected_rows->set_rows(src_selected_rows.rows());
        framework::TensorCopy(src_selected_rows.value(),
                              dst_device,
                              dst_selected_rows->mutable_value());
        if (blocking) {
          platform::DeviceContextPool::Instance().Get(dst_device)->Wait();
          auto src_device = src_selected_rows.value().place();
          if (!(src_device == dst_device)) {
            platform::DeviceContextPool::Instance().Get(src_device)->Wait();
          }
        }
      }

      if (!blocking) {
        IncreaseVarbaseReferenceCountUntilCopyComplete(src, dst_device);
      }

    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The source Tensor(%s) can not copy when it is empty.", src->Name()));
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The destion Tensor(%s) can not copy when it is not empty.",
        dst.Name()));
  }
}

// Bind Methods
void BindImperative(py::module *m_ptr) {
  auto &m = *m_ptr;

  BindOpFunctions1(&m);
  BindOpFunctions2(&m);
  BindOpFunctions3(&m);
  BindOpFunctions4(&m);
  BindOpFunctions5(&m);
  BindOpFunctions6(&m);
  BindOpFunctions7(&m);
  BindOpFunctions8(&m);

#ifndef _WIN32
  // Dygraph DataLoader signal handler
  m.def("_set_process_pids", [](int64_t key, py::object &obj) {
    PADDLE_ENFORCE_EQ(
        py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj),
        true,
        platform::errors::InvalidArgument(
            "The subprocess ids set in DataLoader is illegal."
            "Expected data type is tuple or list, but received %s",
            obj.get_type()));
    py::list pids = py::cast<py::list>(obj);
    std::set<pid_t> pids_set = {};
    for (size_t i = 0; i < pids.size(); i++) {
      pids_set.insert(pids[i].cast<pid_t>());
    }
    imperative::SetLoadProcessPIDs(key, pids_set);
  });
  m.def("_erase_process_pids",
        [](int64_t key) { imperative::EraseLoadProcessPIDs(key); });
  m.def("_set_process_signal_handler",
        []() { imperative::SetLoadProcessSignalHandler(); });
  m.def("_throw_error_if_process_failed",
        []() { imperative::ThrowErrorIfLoadProcessFailed(); });
  // Dygraph DataLoader reader process & thread related functions
  m.def(
      "_convert_to_tensor_list",
      [](py::object &obj) -> py::list {
        // 0. input data check
        PADDLE_ENFORCE(
            py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj),
            platform::errors::InvalidArgument(
                "The batch data read into DataLoader is illegal."
                "Expected data type is tuple or list, but received %s",
                obj.get_type()));
        py::list batch = py::cast<py::list>(obj);
        py::list tensors;
        for (size_t i = 0; i < batch.size(); ++i) {
          // 1. cast to python array
          auto array = batch[i].cast<py::array>();
          PADDLE_ENFORCE_NE(
              string::Sprintf("%s", array.dtype()).compare("object"),
              0,
              platform::errors::InvalidArgument(
                  "Faild to convert input data to a regular ndarray.\n  * "
                  "Usually this means the input data contains nested "
                  "lists with different lengths.\n  * Check the reader "
                  "function passed to 'set_(sample/sample_list/batch)"
                  "_generator' to locate the data causes this issue."));
          // 2. construcct LoDTensor
          framework::LoDTensor t;
          SetTensorFromPyArray<platform::CPUPlace>(
              &t, array, platform::CPUPlace(), true);
          // 3. allocate shared memory
          void *data_ptr = t.data();
          size_t data_size = t.numel() * framework::DataTypeSize(t.dtype());
          auto shared_writer_holder =
              memory::allocation::AllocateMemoryMapWriterAllocation(data_size);
          // 4. maintain mmap fd set & backup ipc_name
          const std::string &ipc_name = shared_writer_holder->ipc_name();
          memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);
          // 5. copy data & reset holder
          memory::Copy(platform::CPUPlace(),
                       shared_writer_holder->ptr(),
                       platform::CPUPlace(),
                       data_ptr,
                       data_size);
          t.ResetHolder(shared_writer_holder);
          // 6. append to result list
          tensors.append(t);
        }
        return tensors;
      },
      py::return_value_policy::take_ownership);

  m.def(
      "_array_to_share_memory_tensor",
      [](py::object &obj) {
        // 1. cast to python array
        auto array = obj.cast<py::array>();
        PADDLE_ENFORCE_NE(
            string::Sprintf("%s", array.dtype()).compare("object"),
            0,
            platform::errors::InvalidArgument(
                "Faild to convert input data to a regular ndarray.\n  * "
                "Usually this means the input data contains nested "
                "lists with different lengths.\n  * Check the reader "
                "function passed to 'set_(sample/sample_list/batch)"
                "_generator' to locate the data causes this issue."));
        // 2. construcct LoDTensor
        framework::LoDTensor t;
        SetTensorFromPyArray<platform::CPUPlace>(
            &t, array, platform::CPUPlace(), true);
        // 3. allocate shared memory
        void *data_ptr = t.data();
        size_t data_size = t.numel() * framework::DataTypeSize(t.dtype());
        auto shared_writer_holder =
            memory::allocation::AllocateMemoryMapWriterAllocation(data_size);
        // 4. maintain mmap fd set & backup ipc_name
        const std::string &ipc_name = shared_writer_holder->ipc_name();
        memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);
        // 5. copy data & reset holder
        memory::Copy(platform::CPUPlace(),
                     shared_writer_holder->ptr(),
                     platform::CPUPlace(),
                     data_ptr,
                     data_size);
        t.ResetHolder(shared_writer_holder);

        return t;
      },
      py::return_value_policy::take_ownership);

  m.def("_remove_tensor_list_mmap_fds", [](py::list &tensor_list) {
    for (size_t i = 0; i < tensor_list.size(); ++i) {
      auto t = tensor_list[i].cast<framework::LoDTensor>();
      auto *mmap_writer_allocation =
          dynamic_cast<memory::allocation::MemoryMapWriterAllocation *>(
              t.Holder().get());
      PADDLE_ENFORCE_NOT_NULL(
          mmap_writer_allocation,
          platform::errors::NotFound("The shared memory of LoDTensor in "
                                     "DataLoader's child process has been "
                                     "released."));
      memory::allocation::MemoryMapFdSet::Instance().Remove(
          mmap_writer_allocation->ipc_name());
    }
  });

  m.def("_cleanup_mmap_fds",
        []() { memory::allocation::MemoryMapFdSet::Instance().Clear(); });
#endif

  m.def("start_imperative_gperf_profiler",
        []() { imperative::StartProfile(); });
  m.def("_set_eager_tracer",
        [](const std::shared_ptr<imperative::Tracer> &tracer) {
          egr::Controller::Instance().SetCurrentTracer(tracer);
        });
  m.def("stop_imperative_gperf_profiler", []() { imperative::StopProfile(); });

  m.def("_is_dygraph_debug_enabled",
        []() { return imperative::IsDebugEnabled(); });
  m.def("_dygraph_debug_level", []() { return imperative::GetDebugLevel(); });
  m.def("_switch_tracer",
        [](const std::shared_ptr<imperative::Tracer> &tracer) {
          egr::Controller::Instance().SetCurrentTracer(tracer);
          imperative::SetCurrentTracer(tracer);
        });
  py::class_<imperative::VarBase, std::shared_ptr<imperative::VarBase>> varbase(
      m, "VarBase", R"DOC()DOC");
  g_varbase_pytype = (PyTypeObject *)varbase.ptr();  // NOLINT
  varbase.def_static("_alive_vars", &imperative::VarBase::AliveVarNames)
      .def("__init__",
           [](imperative::VarBase &self) {
             std::string name =
                 imperative::GetCurrentTracer()->GenerateUniqueName(
                     "generated_tensor");
             new (&self) imperative::VarBase(name);
           })
      .def("__init__",
           [](imperative::VarBase &self,
              framework::proto::VarType::Type dtype,
              const std::vector<int64_t> &dims,
              const py::handle &name,
              framework::proto::VarType::Type type,
              bool persistable) {
             VLOG(4) << "Init VarBase";
             std::string act_name = "";
             if (!name.ptr() || name.ptr() == Py_None) {
               act_name = imperative::GetCurrentTracer()->GenerateUniqueName(
                   "generated_tensor");
             } else {
               act_name = name.cast<std::string>();
             }
             new (&self) imperative::VarBase(act_name);
             self.SetPersistable(persistable);
             self.SetType(type);
             self.SetDataType(dtype);
             if (type == framework::proto::VarType::LOD_TENSOR) {
               auto *tensor =
                   self.MutableVar()->GetMutable<framework::LoDTensor>();
               tensor->Resize(phi::make_ddim(dims));
             }
           })
      .def("__init__",
           &InitVarBaseFromNumpyWithArg<platform::CPUPlace>,
           py::arg("value"),
           py::arg("place"),
           py::arg("persistable") = false,
           py::arg("zero_copy") = false,
           py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__",
           &InitVarBaseFromNumpyWithArg<platform::XPUPlace>,
           py::arg("value"),
           py::arg("place"),
           py::arg("persistable") = false,
           py::arg("zero_copy") = false,
           py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__",
           &InitVarBaseFromNumpyWithArg<platform::CUDAPlace>,
           py::arg("value"),
           py::arg("place"),
           py::arg("persistable") = false,
           py::arg("zero_copy") = false,
           py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__",
           &InitVarBaseFromNumpyWithArg<platform::CUDAPinnedPlace>,
           py::arg("value"),
           py::arg("place"),
           py::arg("persistable") = false,
           py::arg("zero_copy") = false,
           py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__",
           &InitVarBaseFromNumpyWithArg<platform::NPUPlace>,
           py::arg("value"),
           py::arg("place"),
           py::arg("persistable") = false,
           py::arg("zero_copy") = false,
           py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__",
           &InitVarBaseFromNumpyWithArg<platform::MLUPlace>,
           py::arg("value"),
           py::arg("place"),
           py::arg("persistable") = false,
           py::arg("zero_copy") = false,
           py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__",
           &InitVarBaseFromNumpyWithArg<platform::CustomPlace>,
           py::arg("value"),
           py::arg("place"),
           py::arg("persistable") = false,
           py::arg("zero_copy") = false,
           py::arg("name") = "",
           py::arg("stop_gradient") = -1)
      .def("__init__", &InitVarBaseFromNumpyWithArgDefault, py::arg("value"))
      .def("__init__",
           &InitVarBaseFromTensorWithArgDefault,
           py::arg("tensor"),
           py::arg("name") = "")
      .def("__init__",
           &InitVarBaseFromTensorWithArg<platform::CPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("name") = "")
      .def("__init__",
           &InitVarBaseFromTensorWithArg<platform::XPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("name") = "")
      .def("__init__",
           &InitVarBaseFromTensorWithArg<platform::CUDAPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("name") = "")
      .def("__init__",
           &InitVarBaseFromTensorWithArg<platform::CUDAPinnedPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("name") = "")
      .def("__init__",
           &InitVarBaseFromTensorWithArg<platform::NPUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("name") = "")
      .def("__init__",
           &InitVarBaseFromTensorWithArg<platform::MLUPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("name") = "")
      .def("__init__",
           &InitVarBaseFromTensorWithArg<platform::CustomPlace>,
           py::arg("tensor"),
           py::arg("place"),
           py::arg("name") = "")
      .def("__init__", &InitVarBaseFromNumpyWithKwargs)
      .def(
          "__setitem_varbase__",
          [](std::shared_ptr<imperative::VarBase> &self,
             py::handle _index,
             py::object &value_obj) {
            VLOG(4) << "Call __setitem_varbase__";

            auto self_tensor =
                self->MutableVar()->GetMutable<framework::LoDTensor>();
            // NOTE(zhiqiu): PyTuple_Pack increases refcount while PyTuple_New
            // https://github.com/python/cpython/blob/24b63c695ae0a95b06379eaadace66735abac1e2/Objects/tupleobject.c#L251
            PyObject *index_ptr = !PyTuple_Check(_index.ptr())
                                      ? PyTuple_Pack(1, _index.ptr())
                                      : _index.ptr();
            DEFINE_PADDLE_SCOPE_GUARD([index_ptr, &_index]() {
              if (!PyTuple_Check(_index.ptr())) {
                Py_DECREF(index_ptr);
                VLOG(4) << "Call Py_DECREF";
              }
            });

            auto is_tensor = [](py::handle var) {
              if (!var.ptr() || var.ptr() == Py_None) {
                return false;
              }
              try {
                py::cast<std::shared_ptr<imperative::VarBase>>(var);
                return true;
              } catch (py::cast_error &) {
                return false;
              }
            };

            // NOTE(liym27):
            // Increase the version of VarBase self because __setitem__ is an
            // inplace operator for the VarBase self.
            self->BumpInplaceVersion();

            // 1. Check argumnets
            bool parse_index = true;

            // Check whether _index can be parsed.
            const int size = PyTuple_GET_SIZE(index_ptr);
            for (int dim = 0; dim < size; ++dim) {
              PyObject *slice_item = PyTuple_GetItem(index_ptr, dim);
              if (!(PyCheckInteger(slice_item) || PySlice_Check(slice_item) ||
                    slice_item == Py_Ellipsis || slice_item == Py_None)) {
                parse_index = false;
                break;
              }
            }

            // 2. Call op set_value to speed up if the condition is met,
            // otherwise call TensorToPyArray.
            // TODO(liym27): Try not to call TensorToPyArray because it always
            // copys data to cpu place, which reduces performance.
            if (parse_index) {
              std::vector<int> axes, starts, ends, steps, decrease_axes,
                  none_axes, infer_flags, list_select_idxs;
              // if index is a list, list_select_flag will be true
              bool list_select_flag = false;
              ParseIndexingSlice(self_tensor,
                                 index_ptr,
                                 &axes,
                                 &starts,
                                 &ends,
                                 &steps,
                                 &decrease_axes,
                                 &none_axes,
                                 &infer_flags,
                                 &list_select_idxs,
                                 &list_select_flag);

              framework::AttributeMap attrs = {{"axes", axes},
                                               {"starts", starts},
                                               {"ends", ends},
                                               {"steps", steps},
                                               {"decrease_axes", decrease_axes},
                                               {"none_axes", none_axes}};

              imperative::NameVarBaseMap ins = {{"Input", {self}}};
              imperative::NameVarBaseMap outs = {{"Out", {self}}};

              const auto &tracer = imperative::GetCurrentTracer();

              if (tracer->HasGrad()) {
                PADDLE_ENFORCE_EQ(
                    self->IsLeaf() && !self->OverridedStopGradient(),
                    false,
                    platform::errors::InvalidArgument(
                        "Leaf Tensor (%s) that doesn't stop gradient can't use "
                        "inplace strategy.",
                        self->Name()));
              }

              if (PyCheckTensor(value_obj.ptr())) {
                auto value_tensor =
                    value_obj.cast<std::shared_ptr<imperative::VarBase>>();
                ins.insert({"ValueTensor", {value_tensor}});

                // pass the stop_gradient from value to tensor
                if (!value_tensor->OverridedStopGradient() &&
                    self->OverridedStopGradient()) {
                  self->SetOverridedStopGradient(false);
                }
              } else if (py::isinstance<py::array>(value_obj)) {
                auto value_tensor = std::shared_ptr<imperative::VarBase>(
                    new imperative::VarBase(false,
                                            tracer->GenerateUniqueName()));
                py::object value = value_obj;
                if (self->DataType() == framework::proto::VarType::FP32) {
                  if (!py::isinstance<py::array_t<float>>(value_obj)) {
                    value = pybind11::detail::CastNumpyArray<float>(value_obj);
                  }
                } else if (self->DataType() ==
                           framework::proto::VarType::FP64) {
                  if (!py::isinstance<py::array_t<double>>(value_obj)) {
                    value = pybind11::detail::CastNumpyArray<double>(value_obj);
                  }
                } else if (self->DataType() ==
                           framework::proto::VarType::INT32) {
                  if (!py::isinstance<py::array_t<int32_t>>(value_obj)) {
                    value =
                        pybind11::detail::CastNumpyArray<int32_t>(value_obj);
                  }
                } else if (self->DataType() ==
                           framework::proto::VarType::INT64) {
                  if (!py::isinstance<py::array_t<int64_t>>(value_obj)) {
                    value =
                        pybind11::detail::CastNumpyArray<int64_t>(value_obj);
                  }
                } else if (self->DataType() ==
                           framework::proto::VarType::BOOL) {
                  if (!py::isinstance<py::array_t<bool>>(value_obj)) {
                    value = pybind11::detail::CastNumpyArray<bool>(value_obj);
                  }
                } else {
                  PADDLE_THROW(platform::errors::InvalidArgument(
                      "When assign a numpy.np value to a paddle.Tensor, "
                      "the data type of the paddle.Tensor must be bool, "
                      "float32, int32 or int64, "
                      "please check the type of tensor."));
                }

                SetTensorFromPyArray(value_tensor->MutableVar()
                                         ->GetMutable<framework::LoDTensor>(),
                                     value,
                                     self->Place(),
                                     false);
                ins.insert({"ValueTensor", {value_tensor}});

              } else {
                // convert the value to self data type
                if (py::isinstance<py::float_>(value_obj) ||
                    py::isinstance<py::int_>(value_obj) ||
                    py::isinstance<py::bool_>(value_obj)) {
                  if (self->DataType() == framework::proto::VarType::FP32) {
                    attrs["fp32_values"] =
                        std::vector<float>{value_obj.cast<float>()};
                  } else if (self->DataType() ==
                             framework::proto::VarType::FP64) {
                    attrs["fp64_values"] =
                        std::vector<double>{value_obj.cast<double>()};
                  } else if (self->DataType() ==
                             framework::proto::VarType::INT32) {
                    attrs["int32_values"] =
                        std::vector<int32_t>{value_obj.cast<int32_t>()};
                  } else if (self->DataType() ==
                             framework::proto::VarType::INT64) {
                    attrs["int64_values"] =
                        std::vector<int64_t>{value_obj.cast<int64_t>()};
                  } else if (self->DataType() ==
                             framework::proto::VarType::BOOL) {
                    attrs["bool_values"] =
                        std::vector<int>{value_obj.cast<bool>()};
                  } else {
                    PADDLE_THROW(platform::errors::InvalidArgument(
                        "When assign a value to a paddle.Tensor, "
                        "the data type of the paddle.Tensor must be bool, "
                        "float32, int32 or int64, "
                        "please check the type of tensor."));
                  }
                  attrs["shape"] = std::vector<int64_t>{1};

                } else {
                  PADDLE_THROW(platform::errors::InvalidArgument(
                      "Value type error. The assign value allows "
                      "numpy.ndarray, integer, float or bool, "
                      "but received %s.",
                      Py_TYPE(value_obj.ptr())));
                }
              }

              {
                // Release gil and do tracing
                py::gil_scoped_release release;
                tracer->TraceOp("set_value",
                                ins,
                                outs,
                                std::move(attrs),
                                {{"Input", "Out"}});
              }
            } else {
              auto self_numpy = TensorToPyArray(*self_tensor);
              VLOG(4) << "parse_index is false";
              if (is_tensor(_index)) {
                VLOG(4) << "index is tensor";
                auto index_var =
                    py::cast<std::shared_ptr<imperative::VarBase>>(_index);
                auto index_tensor =
                    index_var->MutableVar()->GetMutable<framework::LoDTensor>();
                auto index_numpy = TensorToPyArray(*index_tensor);
                self_numpy[index_numpy] = value_obj;
              } else {
                VLOG(4) << "index is not tensor";
                self_numpy[_index] = value_obj;
              }
              SetTensorFromPyArray(
                  self_tensor, self_numpy, self_tensor->place(), false);
            }
          })
      .def("_getitem_index_not_tensor",
           [](std::shared_ptr<imperative::VarBase> &self, py::handle _index) {
             VLOG(4) << "Call _getitem_index_not_tensor";
             std::vector<int> slice_axes, slice_starts, slice_ends,
                 slice_strides, decrease_axis, none_axes, infer_flags,
                 list_select_idxs;
             // if index is a list, list_select_flag will be true
             bool list_select_flag = false;
             auto tensor =
                 self->MutableVar()->GetMutable<framework::LoDTensor>();
             ParseIndexingSlice(tensor,
                                _index.ptr(),
                                &slice_axes,
                                &slice_starts,
                                &slice_ends,
                                &slice_strides,
                                &decrease_axis,
                                &none_axes,
                                &infer_flags,
                                &list_select_idxs,
                                &list_select_flag);
             // release gil and do tracing
             py::gil_scoped_release release;
             const auto &tracer = imperative::GetCurrentTracer();

             auto out = slice_axes.empty() && !list_select_flag
                            ? self
                            : std::shared_ptr<imperative::VarBase>(
                                  new imperative::VarBase(
                                      tracer->GenerateUniqueName()));

             if (!slice_axes.empty()) {
               imperative::NameVarBaseMap ins = {{"Input", {self}}};
               framework::AttributeMap attrs = {
                   {"axes", slice_axes},
                   {"starts", slice_starts},
                   {"ends", slice_ends},
                   {"infer_flags", infer_flags},
                   {"decrease_axis", decrease_axis}};
               imperative::NameVarBaseMap outs = {{"Out", {out}}};
               std::string op_type = "slice";
               for (auto stride : slice_strides) {
                 if (stride != 1) {
                   op_type = "strided_slice";
                   attrs.insert({"strides", slice_strides});
                   attrs.erase("decrease_axis");
                   break;
                 }
               }
               tracer->TraceOp(op_type, ins, outs, std::move(attrs));
             }
             if (!none_axes.empty()) {
               // Deal with cases when all axes are decreased.
               // After slice, the shape of out is [1], which should have been
               // [], but Paddle doesn't support scalar.
               // In order to ensure the correctness of the final shape of out,
               // one dimension of out needs to be decreased.
               // For example:
               // # x.shape: (2,3,4)
               // out = x[0, 1, 1, None] # out.shape : (1)
               if (static_cast<int>(decrease_axis.size()) ==
                   tensor->dims().size()) {
                 none_axes.pop_back();
               }
               if (!none_axes.empty()) {
                 // Deal with cases that decrease_axes is not empty
                 // For example:
                 // # x.shape: (2,3,4)
                 // out = x[0, 0:2, None] # out.shape : (2, 1, 4)
                 for (auto &axis : none_axes) {
                   int len = 0;
                   for (int da : decrease_axis) {
                     if (da < axis) {
                       len++;
                     }
                   }
                   axis -= len;
                 }

                 imperative::NameVarBaseMap ins = {{"X", {out}}};
                 framework::AttributeMap attrs = {{"axes", none_axes}};
                 auto new_out = std::shared_ptr<imperative::VarBase>(
                     new imperative::VarBase(tracer->GenerateUniqueName()));
                 auto out_xshape = std::shared_ptr<imperative::VarBase>(
                     new imperative::VarBase(tracer->GenerateUniqueName()));
                 imperative::NameVarBaseMap outs = {{"Out", {new_out}},
                                                    {"XShape", {out_xshape}}};
                 tracer->TraceOp("unsqueeze2", ins, outs, std::move(attrs));

                 return new_out;
               }
             }

             // the index is a list
             if (list_select_flag) {
               auto select_index = std::shared_ptr<imperative::VarBase>(
                   new imperative::VarBase(tracer->GenerateUniqueName()));
               auto *idx_tensor = select_index->MutableVar()
                                      ->GetMutable<framework::LoDTensor>();
               auto *dev_ctx = platform::DeviceContextPool::Instance().Get(
                   tracer->ExpectedPlace());
               paddle::framework::TensorFromVector(
                   list_select_idxs, *dev_ctx, idx_tensor);

               imperative::NameVarBaseMap ins = {{"X", {self}},
                                                 {"Index", {select_index}}};
               imperative::NameVarBaseMap outs = {{"Out", {out}}};
               tracer->TraceOp("index_select", ins, outs, {{"dim", 0}});
             }

             return out;
           })
      .def(
          "_getitem_from_offset",
          [](std::shared_ptr<imperative::VarBase> &self, const py::args &args) {
            const auto &tensor = self->Var().Get<framework::LoDTensor>();
            PADDLE_ENFORCE_EQ(
                tensor.IsInitialized(),
                true,
                platform::errors::InvalidArgument(
                    "Tensor of %s is Empty, please check if it has no data.",
                    self->Name()));

            const auto &tensor_dims = tensor.dims();

            std::vector<size_t> dims(tensor_dims.size());
            std::vector<size_t> strides(tensor_dims.size());

            size_t numel = 1;
            for (int i = tensor_dims.size() - 1; i >= 0; --i) {
              strides[i] = numel;
              dims[i] = static_cast<size_t>(tensor_dims[i]);
              numel *= dims[i];
            }
            size_t offset = 0;
            if (args.empty()) {
              PADDLE_ENFORCE_EQ(
                  numel,
                  1,
                  platform::errors::InvalidArgument(
                      "only one element tensors can be converted to Python "
                      "scalars when no input coordinates"));
            } else if (args.size() == 1) {
              offset = args[0].cast<size_t>();
              PADDLE_ENFORCE_LT(
                  offset,
                  numel,
                  platform::errors::InvalidArgument(
                      "index %d is out of bounds for size %d", offset, numel));
            } else {
              PADDLE_ENFORCE_EQ(args.size(),
                                dims.size(),
                                platform::errors::InvalidArgument(
                                    "incorrect number of indices for Tensor"));

              for (size_t i = 0; i < args.size(); ++i) {
                size_t index = args[i].cast<size_t>();
                PADDLE_ENFORCE_LT(
                    index,
                    dims[i],
                    platform::errors::InvalidArgument(
                        "index %d is out fo bounds for axis %d with size %d",
                        index,
                        i,
                        dims[i]));
                offset += index * strides[i];
              }
            }
#define TENSOR_TO_PY_SCALAR(T, proto_type)                                   \
  if (framework::TransToProtoVarType(tensor.dtype()) == proto_type) {        \
    std::string py_dtype_str = details::TensorDTypeToPyDTypeStr(proto_type); \
    T b = TensorGetElement<T>(tensor, offset);                               \
    return py::array(                                                        \
        py::dtype(py_dtype_str.c_str()), {}, {}, static_cast<void *>(&b));   \
  }

            _ForEachDataType_(TENSOR_TO_PY_SCALAR);
#undef TENSOR_TO_PY_SCALAR
            PADDLE_THROW(platform::errors::Unimplemented(
                "Unsupported tensor data type: %s", tensor.dtype()));
          },
          py::return_value_policy::copy)
      .def("_inplace_version",
           [](imperative::VarBase &self) -> uint32_t {
             const auto &var = self.MutableVar();
             PADDLE_ENFORCE_EQ(
                 var->IsInitialized(),
                 true,
                 platform::errors::InvalidArgument(
                     "Tensor of %s is Empty, please check if it has no data.",
                     self.Name()));
             return var->CurrentInplaceVersion();
           })
      .def(
          "_bump_inplace_version",
          [](std::shared_ptr<imperative::VarBase> &self) {
            // NOTE(liym27): _bump_inplace_version is only used for inplace
            // operation
            self->BumpInplaceVersion();
          },
          R"DOC(
        **Notes**:
            **This API is ONLY available in Dygraph mode.**
            **This is a very low level API. Users should not use it directly. **
         Bump the version whenever the Tensor is modified through an inplace operation.
            )DOC")
      .def(
          "numpy",

          [](imperative::VarBase &self) -> py::array {
            const auto &tensor = self.MutableVar()->Get<framework::LoDTensor>();
            PADDLE_ENFORCE_EQ(
                tensor.IsInitialized(),
                true,
                platform::errors::InvalidArgument(
                    "Tensor of %s is Empty, please check if it has no data.",
                    self.Name()));
            return TensorToPyArray(tensor, true);
          },
          R"DOC(
        Returns a numpy array shows the value of current Tensor.

        Returns:
            ndarray: The numpy value of current Tensor.

        Returns type:
            ndarray: dtype is same as current Tensor

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np
                data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
                linear = paddle.nn.Linear(32, 64)
                data = paddle.to_tensor(data)
                x = linear(data)
                print(x.numpy())
       )DOC")
      .def(
          "detach",
          [](const imperative::VarBase &self)
              -> std::shared_ptr<imperative::VarBase> {
            PADDLE_ENFORCE_EQ(
                self.Var().IsInitialized(),
                true,
                platform::errors::InvalidArgument(
                    "Tensor %s has not been initialized!", self.Name()));

            PADDLE_ENFORCE_EQ(
                self.Var().IsType<framework::LoDTensor>() ||
                    self.Var().IsType<phi::SelectedRows>(),
                true,
                platform::errors::InvalidArgument(
                    "Type of Tensor[%s] must be LoDTensor or SelectedRows!",
                    self.Name()));

            auto detach_var = std::make_shared<imperative::VarBase>(
                true, "detach_" + self.Name());

            detach_var->SetPersistable(self.Persistable());
            detach_var->SetType(self.Type());
            detach_var->SetDataType(self.DataType());

            if (self.Var().IsType<framework::LoDTensor>()) {
              const auto &origin_tensor =
                  self.Var().Get<framework::LoDTensor>();
              PADDLE_ENFORCE_EQ(
                  origin_tensor.IsInitialized(),
                  true,
                  platform::errors::InvalidArgument(
                      "Tensor %s has not been initialized!", self.Name()));

              auto *detach_tensor =
                  detach_var->MutableVar()->GetMutable<framework::LoDTensor>();
              detach_tensor->ShareDataWith(origin_tensor);
              // NOTE(liym27): Call ShareInplaceVersionCounterWith to share the
              // same TensorInplaceVersion, which is used to check whether
              // inplace
              // operations are correct.
              detach_tensor->ShareInplaceVersionCounterWith(origin_tensor);
            } else {
              const auto &origin_selected_rows =
                  self.Var().Get<phi::SelectedRows>();
              PADDLE_ENFORCE_EQ(
                  origin_selected_rows.value().IsInitialized(),
                  true,
                  platform::errors::InvalidArgument(
                      "Tensor %s has not been initialized!", self.Name()));

              auto *detach_selected_rows =
                  detach_var->MutableVar()->GetMutable<phi::SelectedRows>();
              detach_selected_rows->set_height(origin_selected_rows.height());
              detach_selected_rows->set_rows(origin_selected_rows.rows());
              detach_selected_rows->mutable_value()->ShareDataWith(
                  origin_selected_rows.value());
              detach_selected_rows->mutable_value()
                  ->ShareInplaceVersionCounterWith(
                      origin_selected_rows.value());
            }
            VLOG(3) << "The detached Tensor(" << detach_var->Name()
                    << ") share data with " << self.Name();
            return detach_var;
          },
          py::return_value_policy::take_ownership,
          R"DOC(

        Returns a new Tensor, detached from the current graph.
        It will share data with origin Tensor and always doesn't have a Tensor copy.
        In addition, the detached Tensor doesn't provide gradient propagation.

        Returns: The detached Tensor.

        Examples:
            .. code-block:: python

                import paddle

                x = paddle.to_tensor(1.0, stop_gradient=False)
                detach_x = x.detach()
                detach_x[:] = 10.0
                print(x)  # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=False,
                          #        [10.])
                y = x**2
                y.backward()
                print(x.grad)         # [20.0]
                print(detach_x.grad)  # None, 'stop_gradient=True' by default

                detach_x.stop_gradient = False # Set stop_gradient to be False, supported auto-grad
                z = detach_x**3
                z.backward()

                print(x.grad)         # [20.0], detach_x is detached from x's graph, not affect each other
                print(detach_x.grad)  # [300.0], detach_x has its own graph

                # Due to sharing of data with origin Tensor, There are some unsafe operations:
                y = 2 * x
                detach_x[:] = 5.0
                y.backward()
                # It will raise Error:
                #   one of the variables needed for gradient computation has been modified by an inplace operation.

       )DOC")
      .def("clear_gradient",
           &imperative::VarBase::ClearGradient,
           py::arg("set_to_zero") = true,
           R"DOC(

        Only for Tensor that has gradient, normally we use this for Parameters since other temporary Tensor doesen't has gradient.

        The Gradient of current Tensor will be set to ``0`` .

        Returns:  None

        Examples:
             .. code-block:: python

                import paddle
                input = paddle.uniform([10, 2])
                linear = paddle.nn.Linear(2, 3)
                out = linear(input)
                out.backward()
                print("Before clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))
                linear.weight.clear_gradient()
                print("After clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))
      )DOC")
      .def("_gradient_set_empty",
           &imperative::VarBase::_GradientSetEmpty,
           py::arg("set_is_empty") = true)
      .def("_is_gradient_set_empty", &imperative::VarBase::_IsGradientSetEmpty)
      .def(
          "clone",
          [](std::shared_ptr<imperative::VarBase> &self) {
            const auto &tensor = self->Var().Get<framework::LoDTensor>();
            PADDLE_ENFORCE_EQ(tensor.IsInitialized(),
                              true,
                              platform::errors::InvalidArgument(
                                  "%s has not been initialized", self->Name()));
            auto tracer = imperative::GetCurrentTracer();
            auto new_var = std::make_shared<imperative::VarBase>(
                true, tracer->GenerateUniqueName(self->Name() + "_clone"));
            framework::AttributeMap attrs;
            imperative::NameVarBaseMap ins = {{"X", {self}}};
            imperative::NameVarBaseMap outs = {{"Out", {new_var}}};
            tracer->TraceOp("assign", ins, outs, attrs);
            return new_var;
          },
          py::return_value_policy::copy,
          R"DOC(

        Returns a new Tensor, which is clone of origin Tensor, and it remains in the current graph.
        It will always have a Tensor copy.
        Tn addition, the cloned Tensor provides gradient propagation.

        Returns: The cloned Tensor.

        Examples:
            .. code-block:: python

              import paddle

              x = paddle.to_tensor(1.0, stop_gradient=False)
              clone_x = x.clone()
              y = clone_x**2
              y.backward()
              print(clone_x.stop_gradient) # False
              print(clone_x.grad)          # [2.0], support gradient propagation
              print(x.stop_gradient)       # False
              print(x.grad)                # [2.0], clone_x support gradient propagation for x

              x = paddle.to_tensor(1.0)
              clone_x = x.clone()
              clone_x.stop_gradient = False
              z = clone_x**3
              z.backward()
              print(clone_x.stop_gradient) # False
              print(clone_x.grad)          # [3.0], support gradient propagation
              print(x.stop_gradient) # True
              print(x.grad)          # None
       )DOC")
      .def("_grad_name", &imperative::VarBase::GradVarName)
      .def(
          "_grad_value",
          [](imperative::VarBase &self) {
            return self.MutableGradVar()->Get<framework::LoDTensor>();
          },
          py::return_value_policy::reference)
      .def("_set_grad_type",
           [](imperative::VarBase &self, framework::proto::VarType::Type type) {
             self.MutableGradVarBase()->SetType(type);
           })
      .def("_reset_grad_inplace_version",
           [](imperative::VarBase &self, bool set_to_zero) {
             /*
             *** This interfaceis a complete hack ***
             reset_grad_inplace_version removes all inplace related records to
             Grad VarBase/VariableWrapper,
             the essential purpose of which is to let you use inplace operations
             as if using its non-inplaced version,
             which of course will cause unexpected consequences if not used with
             care.
             Make sure you fully understand what you're doing before make use of
             this interface, and prepare for the worst.
             */
             py::gil_scoped_release release;

             if (self.HasGradVar()) {
               auto grad_var = self.GradVarBase();
               auto var_wrapper = grad_var->SharedVar();
               if (var_wrapper) {
                 var_wrapper->ResetInplaceVersion(set_to_zero);
               }
             }
           })
      .def(
          "_grad_ivar",
          [](const imperative::VarBase &self) {
            auto &grad_var = self.GradVarBase();

            if (grad_var && grad_var->Var().IsInitialized()) {
              auto *tensor =
                  grad_var->MutableVar()->IsType<framework::LoDTensor>()
                      ? grad_var->MutableVar()
                            ->GetMutable<framework::LoDTensor>()
                      : grad_var->MutableVar()
                            ->GetMutable<phi::SelectedRows>()
                            ->mutable_value();

              if (tensor->IsInitialized()) {
                return grad_var;
              }
            }
            return std::shared_ptr<imperative::VarBase>(nullptr);
          },
          py::return_value_policy::copy)
      .def("_set_grad_ivar",
           [](imperative::VarBase &self, imperative::VarBase &grad) {
             self.SetGradVarBase(grad);
           })
      .def("_is_sparse",
           [](imperative::VarBase &self) {
             return self.Var().IsType<phi::SelectedRows>();
           })
      .def(
          "_allreduce",
          [](imperative::VarBase &self,
             const imperative::ParallelStrategy &strategy) {
            if (strategy.nranks_ > 1) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2212
              imperative::AllReduce(self.Var(), self.MutableVar(), strategy);
#else
               if (!self.Var().IsType<phi::SelectedRows>()) {
                 imperative::AllReduce(self.Var(), self.MutableVar(), strategy);
               } else {
                 PADDLE_THROW(platform::errors::Unimplemented(
                     "Imperative SelectedRows allreduce is not supported when "
                     "paddle is compiled with NCCL verison lower than v2.2.12. "
                     "You can set is_sparse=False for the Layer containing "
                     "this argument, such as Embedding(is_sparse=False)."));
               }
#endif  // NCCL_VERSION_CODE
#else
               PADDLE_THROW(platform::errors::Unimplemented(
                   "Imperative allreduce is not supported when paddle is "
                   "not compiled with NCCL."));
#endif  // PADDLE_WITH_NCCL or PADDLE_WITH_RCCL
            }
          },
          py::call_guard<py::gil_scoped_release>())
      .def("_register_grad_hook",
           [](imperative::VarBase &self, const py::handle &hook) {
             PADDLE_ENFORCE_EQ(
                 !self.OverridedStopGradient() && self.HasGradVar(),
                 true,
                 platform::errors::InvalidArgument(
                     "Cannot register gradient hook on a Tensor that stop "
                     "gradient or without gradient."));
             return self.GradVarBase()->AddVariableWrapperHook(
                 std::make_shared<PyVariableWrapperHook>(hook.ptr()));
           })
      .def("_remove_grad_hook",
           [](imperative::VarBase &self, int64_t hook_id) {
             PADDLE_ENFORCE_EQ(
                 !self.OverridedStopGradient() && self.HasGradVar(),
                 true,
                 platform::errors::InvalidArgument(
                     "Cannot remove gradient hook on a Tensor that stop "
                     "gradient or without gradient."));
             return self.GradVarBase()->RemoveVariableWrapperHook(hook_id);
           })
      .def("_register_void_function_post_hook",
           [](imperative::VarBase &self, const py::handle &hook) {
             PADDLE_ENFORCE_EQ(
                 !self.OverridedStopGradient() && self.HasGradVar(),
                 true,
                 platform::errors::InvalidArgument(
                     "Cannot register void function post hook on a Tensor that "
                     "stop "
                     "gradient or without gradient."));
             auto py_func = PyObjectCast<std::function<void()>>(hook.ptr());
             auto grad_node = self.MutableGradVarBase()->GradNode();
             for (auto &cur_op : *grad_node) {
               cur_op.AddVoidFunctionPostHook(
                   std::make_shared<std::function<void()>>(py_func));
             }
           })
      .def(
          "_register_backward_hook",
          [](imperative::VarBase &self, const py::handle &hook) {
            PADDLE_ENFORCE_EQ(
                self.IsLeaf(),
                true,
                platform::errors::InvalidArgument(
                    "Only can register backward hook for leaf Tensor."));
            PADDLE_ENFORCE_EQ(
                !self.OverridedStopGradient() && self.HasGradVar(),
                true,
                platform::errors::InvalidArgument(
                    "Cannot register backward hook on a Tensor that stop "
                    "gradient or without gradient."));
            auto py_func = PyObjectCast<std::function<void()>>(hook.ptr());
            self.GradVarBase()->AddVoidHook(
                std::make_shared<std::function<void()>>(py_func));
          },
          R"DOC(
             Registers a backward hook for current Tensor.

             This hook will be called every time the gradient of current Tensor has been fully calculated.

             There are two differences with `_register_grad_hook`:
             1. This backward hook will be executed after the gradient accumulation completed across batchs,
                but the hook registered by `_register_grad_hook` will be executed the gradient accumulation
                completed in current batch.
             2. This backward hook function should have the following signature:

                  hook() -> None

                It requires no input and no return value.

             Args:
                 hook(function): A backward hook to be registered for Tensor.gradient

             Returns:
                 None
           )DOC")
      .def(
          "cpu",
          [](const std::shared_ptr<imperative::VarBase> &self) {
            if (platform::is_cpu_place(self->Place())) {
              return self;
            } else {
              auto new_var = self->NewVarBase(platform::CPUPlace(), true);
              new_var->SetOverridedStopGradient(self->OverridedStopGradient());
              return new_var;
            }
          },
          R"DOC(
        Returns a copy of this Tensor in CPU memory.

        If this Tensor is already in CPU memory, then no copy is performed and the original Tensor is returned.

        Examples:
            .. code-block:: python

              import paddle
              x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
              print(x.place)    # CUDAPlace(0)

              y = x.cpu()
              print(y.place)    # CPUPlace

              )DOC")
      .def(
          "pin_memory",
          [](const std::shared_ptr<imperative::VarBase> &self) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
            PADDLE_THROW(platform::errors::PermissionDenied(
                "Cannot copy this Tensor to pinned memory in CPU version "
                "Paddle, "
                "Please recompile or reinstall Paddle with CUDA support."));
#endif
            if (platform::is_cuda_pinned_place(self->Place())) {
              return self;
            } else {
              auto new_var =
                  self->NewVarBase(platform::CUDAPinnedPlace(), true);
              new_var->SetOverridedStopGradient(self->OverridedStopGradient());
              return new_var;
            }
          },
          R"DOC(
        Returns a copy of this Tensor in pin memory.

        If this Tensor is already in pin memory, then no copy is performed and the original Tensor is returned.

        Examples:
            .. code-block:: python

              import paddle
              x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
              print(x.place)      # CUDAPlace(0)

              y = x.pin_memory()
              print(y.place)      # CUDAPinnedPlace

      )DOC")
      .def(
          "cuda",
          [](const std::shared_ptr<imperative::VarBase> &self,
             py::handle &handle,
             bool blocking) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
            PADDLE_THROW(platform::errors::PermissionDenied(
                "Cannot copy this Tensor to GPU in CPU version Paddle, "
                "Please recompile or reinstall Paddle with CUDA support."));
#else
            int device_count = platform::GetGPUDeviceCount();
            int device_id = 0;
            if (handle == py::none()) {
              auto default_place =
                  imperative::GetCurrentTracer()->ExpectedPlace();
              device_id = default_place.GetDeviceId();
            } else {
              PyObject *py_obj = handle.ptr();
              PADDLE_ENFORCE_EQ(
                  PyCheckInteger(py_obj), true,
                  platform::errors::InvalidArgument(
                      " 'device_id' must be a positive integer"));
              device_id = py::cast<int>(handle);
            }
            PADDLE_ENFORCE_GE(
                device_id, 0,
                platform::errors::InvalidArgument(
                    "Can not copy Tensor to Invalid CUDAPlace(%d), device id "
                    "must inside [0, %d)",
                    device_id, device_count));
            PADDLE_ENFORCE_LT(
                device_id, device_count,
                platform::errors::InvalidArgument(
                    "Can not copy Tensor to Invalid CUDAPlace(%d), device id "
                    "must inside [0, %d)",
                    device_id, device_count));
            platform::CUDAPlace place = platform::CUDAPlace(device_id);
            if (platform::is_same_place(self->Place(), place)) {
              return self;
            } else {
              auto new_var = self->NewVarBase(place, blocking);
              new_var->SetOverridedStopGradient(self->OverridedStopGradient());
              return new_var;
            }
#endif
          },
          py::arg("device_id") = py::none(),
          py::arg("blocking") = true,
          R"DOC(
        Returns a copy of this Tensor in GPU memory.

        If this Tensor is already in GPU memory and device_id is default,
        then no copy is performed and the original Tensor is returned.

        Args:
            device_id(int, optional): The destination GPU device id. Default: None, means current device.
            blocking(bool, optional): If False and the source is in pinned memory, the copy will be
              asynchronous with respect to the host. Otherwise, the argument has no effect. Default: False.

        Examples:
            .. code-block:: python

              # required: gpu
              import paddle
              x = paddle.to_tensor(1.0, place=paddle.CPUPlace())
              print(x.place)        # Place(cpu)

              y = x.cuda()
              print(y.place)        # Place(gpu:0)

              y = x.cuda(None)
              print(y.place)        # Place(gpu:0)

              paddle.device.set_device("gpu:1")
              y = x.cuda(None)
              print(y.place)        # Place(gpu:1)
       )DOC")
      .def(
          "_share_memory",
          [](const std::shared_ptr<imperative::VarBase> &self) {
#ifndef _WIN32
            PADDLE_ENFORCE_EQ(
                platform::is_cpu_place(self->Place()),
                true,
                platform::errors::InvalidArgument(
                    "Sharing memory only support CPU Tensor currently"));
            // 1. get LoDTensor
            auto *t = self->MutableVar()->GetMutable<framework::LoDTensor>();
            // 2. allocate shared memory
            void *data_ptr = t->data();
            size_t data_size =
                t->numel() * framework::SizeOfType(
                                 framework::TransToProtoVarType(t->dtype()));
            auto shared_writer_holder =
                memory::allocation::AllocateMemoryMapWriterAllocation(
                    data_size);
            // 3. maintain mmap fd set & backup ipc_name
            const std::string &ipc_name = shared_writer_holder->ipc_name();
            memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);
            // 4. copy data & reset holder
            memory::Copy(platform::CPUPlace(),
                         shared_writer_holder->ptr(),
                         platform::CPUPlace(),
                         data_ptr,
                         data_size);
            t->ResetHolder(shared_writer_holder);
            return *t;
#else
             PADDLE_THROW(platform::errors::PermissionDenied(
                 "Sharing memory in Windows OS is not supported currently"));
#endif
          },
          py::return_value_policy::reference)
#if defined(PADDLE_WITH_CUDA)
      .def(
          "_uva",
          [](const std::shared_ptr<imperative::VarBase> &self, int device_id) {
            PADDLE_ENFORCE_EQ(platform::is_cpu_place(self->Place()),
                              true,
                              platform::errors::InvalidArgument(
                                  "Unified virtual addressing only support "
                                  "CPU Tensor currently."));
            auto *self_tensor =
                self->MutableVar()->GetMutable<framework::LoDTensor>();
            tensor_uva(self_tensor, device_id);
          },
          py::arg("device_id") = 0,
          py::return_value_policy::reference,
          R"DOC(
        Returns self tensor with the UVA(unified virtual addressing).

        Args:
            device_id(int, optional): The destination GPU device id. Default: None, means current device.

        Examples:
            .. code-block:: python

              # required: gpu
              import paddle
              x = paddle.to_tensor([1, 2, 3], place=paddle.CPUPlace())
              x._uva()
              print(x)
       )DOC")
#endif
      .def("copy_", &imperative::VarBase::CopyFrom)
      .def(
          "_copy_to",
          [](const std::shared_ptr<imperative::VarBase> &self,
             const platform::CPUPlace &place,
             bool blocking) {
            auto new_var = self->NewVarBase(place, blocking);
            // Note(zhiqiu): Since NewVarBase may use GpuCopyAsync to
            // copy data from the tensor of self to the tensor of new varbase,
            // we need to ensure that the varbase self is not destructed until
            // the GpuCopyAsync is completed. Otherwise, the memory may be
            // freed
            // when varbase self is destructed.
            // To do that, we increase the reference count of self by 1 and
            // add a cuda event to wait the GpuCopyAsync's completion.
            if (!blocking) {
              IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
            }
            return new_var;
          },
          py::return_value_policy::copy)
      .def(
          "_copy_to",
          [](const std::shared_ptr<imperative::VarBase> &self,
             const platform::CUDAPinnedPlace &place,
             bool blocking) {
            auto new_var = self->NewVarBase(place, blocking);
            if (!blocking) {
              IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
            }
            return new_var;
          },
          py::return_value_policy::copy)
      .def(
          "_copy_to",
          [](const std::shared_ptr<imperative::VarBase> &self,
             const platform::XPUPlace &place,
             bool blocking) {
            auto new_var = self->NewVarBase(place, blocking);
            if (!blocking) {
              IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
            }
            return new_var;
          },
          py::return_value_policy::copy)
      .def(
          "_copy_to",
          [](const std::shared_ptr<imperative::VarBase> &self,
             const platform::CUDAPlace &place,
             bool blocking) {
            auto new_var = self->NewVarBase(place, blocking);
            if (!blocking) {
              IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
            }
            return new_var;
          },
          py::return_value_policy::copy)
      .def(
          "_copy_to",
          [](const std::shared_ptr<imperative::VarBase> &self,
             const platform::NPUPlace &place,
             bool blocking) {
            auto new_var = self->NewVarBase(place, blocking);
            if (!blocking) {
              IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
            }
            return new_var;
          },
          py::return_value_policy::copy)
      .def(
          "_copy_to",
          [](const std::shared_ptr<imperative::VarBase> &self,
             const platform::IPUPlace &place,
             bool blocking) {
            auto new_var = self->NewVarBase(place, blocking);
            if (!blocking) {
              IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
            }
            return new_var;
          },
          py::return_value_policy::copy)
      .def(
          "_copy_to",
          [](const std::shared_ptr<imperative::VarBase> &self,
             const platform::MLUPlace &place,
             bool blocking) {
            auto new_var = self->NewVarBase(place, blocking);
            if (!blocking) {
              IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
            }
            return new_var;
          },
          py::return_value_policy::copy)
      .def(
          "_copy_to",
          [](const std::shared_ptr<imperative::VarBase> &self,
             const platform::CustomPlace &place,
             bool blocking) {
            auto new_var = self->NewVarBase(place, blocking);
            if (!blocking) {
              IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
            }
            return new_var;
          },
          py::return_value_policy::copy)
      .def(
          "_copy_to",
          [](const std::shared_ptr<imperative::VarBase> &self,
             const platform::Place &place,
             bool blocking) {
            auto new_var = self->NewVarBase(place, blocking);
            if (!blocking) {
              IncreaseVarbaseReferenceCountUntilCopyComplete(self, place);
            }
            return new_var;
          },
          py::return_value_policy::copy)
      .def(
          "value",
          [](imperative::VarBase &self) { return self.MutableVar(); },
          py::return_value_policy::reference)
      .def("_clear",
           [](const std::shared_ptr<imperative::VarBase> &self) {
             auto *t = self->MutableVar()->GetMutable<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(
                 t->IsInitialized(),
                 true,
                 platform::errors::InvalidArgument(
                     "Tensor %s has not been initialized!", self->Name()));
             t->clear();
           })
      .def("_offset",
           [](const std::shared_ptr<imperative::VarBase> &self) {
             auto *t = self->MutableVar()->GetMutable<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(
                 t->IsInitialized(),
                 true,
                 platform::errors::InvalidArgument(
                     "Tensor %s has not been initialized!", self->Name()));
             return t->offset();
           })
      .def("_share_buffer_to",
           [](const std::shared_ptr<imperative::VarBase> &self,
              std::shared_ptr<imperative::VarBase> &dst) {
             auto *src = self->MutableVar()->GetMutable<framework::LoDTensor>();
             auto *dst_ = dst->MutableVar()->GetMutable<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(
                 src->IsInitialized(),
                 true,
                 platform::errors::InvalidArgument(
                     "Tensor %s has not been initialized!", self->Name()));
             dst_->ShareBufferWith(*src);
             dst_->ShareDataTypeWith(*src);
           })
      .def("_is_shared_buffer_with",
           [](const std::shared_ptr<imperative::VarBase> &self,
              std::shared_ptr<imperative::VarBase> &dst) {
             auto *src = self->MutableVar()->GetMutable<framework::LoDTensor>();
             auto *dst_ = dst->MutableVar()->GetMutable<framework::LoDTensor>();
             if (!src->IsInitialized() || !dst_->IsInitialized()) {
               return false;
             }
             return dst_->IsSharedBufferWith(*src);
           })
      .def("_share_underline_tensor_to",
           [](const std::shared_ptr<imperative::VarBase> &self,
              std::shared_ptr<imperative::VarBase> &dst) {
             auto *src = self->MutableVar()->GetMutable<framework::LoDTensor>();
             auto *dst_ = dst->MutableVar()->GetMutable<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(
                 src->IsInitialized(),
                 true,
                 platform::errors::InvalidArgument(
                     "Tensor %s has not been initialized!", self->Name()));
             dst_->ShareBufferWith(*src);
             dst_->ShareDataTypeWith(*src);
             dst_->Resize(src->dims());
           })
      .def("_is_shared_underline_tensor_with",
           [](const std::shared_ptr<imperative::VarBase> &self,
              std::shared_ptr<imperative::VarBase> &dst) {
             auto *src = self->MutableVar()->GetMutable<framework::LoDTensor>();
             auto *dst_ = dst->MutableVar()->GetMutable<framework::LoDTensor>();
             if (!src->IsInitialized() || !dst_->IsInitialized()) {
               return false;
             }
             return dst_->IsSharedBufferWith(*src);
           })
      .def("_slice",
           [](const std::shared_ptr<imperative::VarBase> &self,
              int64_t begin_idx,
              int64_t end_idx) {
             auto *t = self->MutableVar()->GetMutable<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(
                 t->IsInitialized(),
                 true,
                 platform::errors::InvalidArgument(
                     "Tensor %s has not been initialized!", self->Name()));
             return t->Slice(begin_idx, end_idx);
           })
      .def("_copy_gradient_from",
           [](std::shared_ptr<imperative::VarBase> &self,
              const imperative::VarBase &src) { self->_CopyGradientFrom(src); })
      .def("_numel",
           [](std::shared_ptr<imperative::VarBase> &self) {
             auto *t = self->MutableVar()->GetMutable<framework::LoDTensor>();
             return t->numel();
           })
      .def("element_size", &imperative::VarBase::ElementSize, R"DOC(
        Returns the size in bytes of an element in the Tensor.

        Examples:
          .. code-block:: python

            import paddle

            x = paddle.to_tensor(1, dtype='bool')
            x.element_size() # 1

            x = paddle.to_tensor(1, dtype='float16')
            x.element_size() # 2

            x = paddle.to_tensor(1, dtype='float32')
            x.element_size() # 4

            x = paddle.to_tensor(1, dtype='float64')
            x.element_size() # 8

            x = paddle.to_tensor(1, dtype='complex128')
            x.element_size() # 16
       )DOC")
      .def_property(
          "name", &imperative::VarBase::Name, &imperative::VarBase::SetName)
      .def_property("stop_gradient",
                    &imperative::VarBase::OverridedStopGradient,
                    &imperative::VarBase::SetOverridedStopGradient)
      .def_property("persistable",
                    &imperative::VarBase::Persistable,
                    &imperative::VarBase::SetPersistable)
      .def_property_readonly(
          "shape",
          [](imperative::VarBase &self) {
            if (self.Var().IsType<framework::LoDTensor>()) {
              return phi::vectorize<int>(
                  self.Var().Get<framework::LoDTensor>().dims());
            } else if (self.Var().IsType<phi::SelectedRows>()) {
              return phi::vectorize<int>(
                  self.Var().Get<phi::SelectedRows>().value().dims());
            } else if (self.Var().IsType<framework::Strings>()) {
              return std::vector<int>{static_cast<int>(
                  self.Var().Get<framework::Strings>().size())};
            } else if (self.Var().IsType<framework::Vocab>()) {
              return std::vector<int>{
                  static_cast<int>(self.Var().Get<framework::Vocab>().size())};
            } else {
              VLOG(2) << "It is meaningless to get shape of "
                         "variable type "
                      << GetTypeName(self);
              return std::vector<int>();
            }
          })
      .def_property_readonly(
          "layout",
          [](imperative::VarBase &self) {
            if (self.Var().IsType<framework::LoDTensor>()) {
              auto layout = self.Var().Get<framework::LoDTensor>().layout();
              return paddle::framework::DataLayoutToString(layout);
            }
            return std::string("");
          })
      .def_property_readonly("is_leaf",
                             &imperative::VarBase::IsLeaf,
                             R"DOC(
      Whether a Tensor is leaf Tensor.

      For the Tensor whose stop_gradient is ``True`` , it will be leaf Tensor.

      For the Tensor whose stop_gradient is ``False`` , it will be leaf Tensor too if it is created by user.

      Returns:
          bool: Whether a Tensor is leaf Tensor.

      Examples:
          .. code-block:: python

              import paddle

              x = paddle.to_tensor(1.)
              print(x.is_leaf) # True

              x = paddle.to_tensor(1., stop_gradient=True)
              y = x + 1
              print(x.is_leaf) # True
              print(y.is_leaf) # True

              x = paddle.to_tensor(1., stop_gradient=False)
              y = x + 1
              print(x.is_leaf) # True
              print(y.is_leaf) # False
       )DOC")
      .def_property_readonly(
          "place",
          [](imperative::VarBase &self) { return self.Place(); },
          py::return_value_policy::copy)
      .def_property_readonly("_place_str",
                             [](imperative::VarBase &self) {
                               std::stringstream ostr;
                               ostr << self.Place();
                               return ostr.str();
                             })
      .def_property_readonly("type", &imperative::VarBase::Type)
      .def_property_readonly("dtype", &imperative::VarBase::DataType);

  py::class_<imperative::jit::ProgramDescTracer>(m, "ProgramDescTracer", "")
      .def("create_program_desc",
           &imperative::jit::ProgramDescTracer::CreateProgramDesc)
      .def("reset", &imperative::jit::ProgramDescTracer::Reset);

  py::enum_<paddle::imperative::AmpLevel>(m, "AmpLevel", py::arithmetic())
      .value("O0", paddle::imperative::AmpLevel::O0)
      .value("O1", paddle::imperative::AmpLevel::O1)
      .value("O2", paddle::imperative::AmpLevel::O2)
      .value("O3", paddle::imperative::AmpLevel::O3)
      .export_values();

  py::class_<imperative::Tracer, std::shared_ptr<imperative::Tracer>>(
      m, "Tracer", R"DOC()DOC")
      .def("__init__",
           [](imperative::Tracer &self) { new (&self) imperative::Tracer(); })
      .def_property("_enable_program_desc_tracing",
                    &imperative::Tracer::IsProgramDescTracingEnabled,
                    &imperative::Tracer::SetEnableProgramDescTracing)
      .def_property("_amp_level",
                    &imperative::Tracer::GetAmpLevel,
                    &imperative::Tracer::SetAmpLevel)
      .def_property("_amp_dtype",
                    &imperative::Tracer::GetAmpDtype,
                    &imperative::Tracer::SetAmpDtype)
      .def_property("_has_grad",
                    &imperative::Tracer::HasGrad,
                    &imperative::Tracer::SetHasGrad)
      .def_property(
          "_expected_place",
          [](const imperative::Tracer &self) -> py::object {
            return py::cast(self.ExpectedPlace());
          },
          [](imperative::Tracer &self, const py::object &obj) {
            if (py::isinstance<platform::CUDAPlace>(obj)) {
              auto p = obj.cast<platform::CUDAPlace *>();
              self.SetExpectedPlace(*p);
              // TODO(jiabin): Support eager here when we need to make all
              // dygraph in eager mode
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::XPUPlace>(obj)) {
              auto p = obj.cast<platform::XPUPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::CPUPlace>(obj)) {
              auto p = obj.cast<platform::CPUPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::CUDAPinnedPlace>(obj)) {
              auto p = obj.cast<platform::CUDAPinnedPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::NPUPlace>(obj)) {
              auto p = obj.cast<platform::NPUPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::IPUPlace>(obj)) {
              auto p = obj.cast<platform::IPUPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::MLUPlace>(obj)) {
              auto p = obj.cast<platform::MLUPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::CustomPlace>(obj)) {
              auto p = obj.cast<platform::CustomPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<platform::Place>(obj)) {
              auto p = obj.cast<platform::Place *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "Incompatible Place Type: supports XPUPlace, CUDAPlace, "
                  "CPUPlace, NPUPlace, IPUPlace, MLUPlace"
                  "and CUDAPinnedPlace, "
                  "but got Unknown Type!"));
            }
          })
      .def("_get_program_desc_tracer",
           &imperative::Tracer::GetProgramDescTracer,
           py::return_value_policy::reference)
      .def("_generate_unique_name",
           &imperative::Tracer::GenerateUniqueName,
           py::arg("key") = "dygraph_tmp")
      .def("_set_amp_op_list",
           [](imperative::Tracer &self,
              std::unordered_set<std::string> &allow_ops,
              std::unordered_set<std::string> &block_ops) {
             // NOTE(zhiqiu): The automatic conversion in pybind11 between
             // c++
             // STL and python set/list/dict involve a copy operation that
             // prevents pass-by-reference semantics, so it is ok to swap.
             // The reaseon why not directly pass
             // std::shared_ptr<std::unordered_set<std::string>>
             // is that pybind11 forbid shared_ptr<T> where T is not custom
             // type.
             imperative::AmpOperators::Instance().GetMutableAllowOps()->swap(
                 allow_ops);
             imperative::AmpOperators::Instance().GetMutableBlockOps()->swap(
                 block_ops);
             VLOG(5) << "AMP operators changed, "
                     << imperative::AmpOperators::Instance();
           })
      .def("_get_amp_op_list",
           [](imperative::Tracer &self) {
             return std::make_tuple(
                 *(imperative::AmpOperators::Instance().GetMutableAllowOps()),
                 *(imperative::AmpOperators::Instance().GetMutableBlockOps()));
           })
      .def("_get_kernel_signature",
           [](imperative::Tracer &self,
              const std::string &type,
              const PyNameVarBaseMap &ins,
              const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs) {
             // TODO(xiongkun): move this function outside of tracer.
             auto ins_map = ConvertToNameTensorMap(ins);
             auto outs_map = ConvertToNameTensorMap(outs);
             {
               auto input_to_vector =
                   [](paddle::small_vector<const char *> &vec) {
                     return std::vector<std::string>(vec.begin(), vec.end());
                   };
               auto output_to_vector =
                   [](paddle::small_vector<const char *> &vec) {
                     return std::vector<std::string>(vec.begin(), vec.end());
                   };
               auto attr_to_vector =
                   [](paddle::small_vector<const char *> &vec) {
                     return std::vector<std::string>(vec.begin(), vec.end());
                   };
               auto ret = self.GetExpectedKernelSignature(
                   type, ins_map, outs_map, attrs);
               auto kernelsig_ins = input_to_vector(ret.input_names);
               auto kernelsig_attrs = attr_to_vector(ret.attr_names);
               auto kernelsig_outs = output_to_vector(ret.output_names);
               return std::make_tuple(
                   kernelsig_ins, kernelsig_attrs, kernelsig_outs);
             }
           })
      .def("trace",
           [](imperative::Tracer &self,
              const std::string &type,
              const PyNameVarBaseMap &ins,
              const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs,
              const platform::CustomPlace &place,
              bool trace_backward,
              const std::map<std::string, std::string> &inplace_map = {}) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp<imperative::VarBase>(type,
                                                 std::move(ins_map),
                                                 std::move(outs_map),
                                                 std::move(attrs),
                                                 place,
                                                 trace_backward,
                                                 inplace_map);
             }
           })
      .def("trace",
           [](imperative::Tracer &self,
              const std::string &type,
              const PyNameVarBaseMap &ins,
              const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs,
              const platform::XPUPlace &place,
              bool trace_backward,
              const std::map<std::string, std::string> &inplace_map = {}) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp<imperative::VarBase>(type,
                                                 std::move(ins_map),
                                                 std::move(outs_map),
                                                 std::move(attrs),
                                                 place,
                                                 trace_backward,
                                                 inplace_map);
             }
           })
      .def("trace",
           [](imperative::Tracer &self,
              const std::string &type,
              const PyNameVarBaseMap &ins,
              const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs,
              const platform::CUDAPlace &place,
              bool trace_backward,
              const std::map<std::string, std::string> &inplace_map = {}) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp<imperative::VarBase>(type,
                                                 std::move(ins_map),
                                                 std::move(outs_map),
                                                 std::move(attrs),
                                                 place,
                                                 trace_backward,
                                                 inplace_map);
             }
           })
      .def("trace",
           [](imperative::Tracer &self,
              const std::string &type,
              const PyNameVarBaseMap &ins,
              const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs,
              const platform::NPUPlace &place,
              bool trace_backward,
              const std::map<std::string, std::string> &inplace_map = {}) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp<imperative::VarBase>(type,
                                                 std::move(ins_map),
                                                 std::move(outs_map),
                                                 std::move(attrs),
                                                 place,
                                                 trace_backward,
                                                 inplace_map);
             }
           })
      .def("trace",
           [](imperative::Tracer &self,
              const std::string &type,
              const PyNameVarBaseMap &ins,
              const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs,
              const platform::IPUPlace &place,
              bool trace_backward,
              const std::map<std::string, std::string> &inplace_map = {}) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp<imperative::VarBase>(type,
                                                 std::move(ins_map),
                                                 std::move(outs_map),
                                                 std::move(attrs),
                                                 place,
                                                 trace_backward,
                                                 inplace_map);
             }
           })
      .def("trace",
           [](imperative::Tracer &self,
              const std::string &type,
              const PyNameVarBaseMap &ins,
              const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs,
              const platform::MLUPlace &place,
              bool trace_backward,
              const std::map<std::string, std::string> &inplace_map = {}) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp<imperative::VarBase>(type,
                                                 std::move(ins_map),
                                                 std::move(outs_map),
                                                 std::move(attrs),
                                                 place,
                                                 trace_backward,
                                                 inplace_map);
             }
           })
      .def("trace",
           [](imperative::Tracer &self,
              const std::string &type,
              const PyNameVarBaseMap &ins,
              const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs,
              const platform::CPUPlace &place,
              bool trace_backward,
              const std::map<std::string, std::string> &inplace_map = {}) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp<imperative::VarBase>(type,
                                                 std::move(ins_map),
                                                 std::move(outs_map),
                                                 std::move(attrs),
                                                 place,
                                                 trace_backward,
                                                 inplace_map);
             }
           });

  // define parallel context
  py::class_<imperative::ParallelStrategy> parallel_strategy(
      m, "ParallelStrategy", "");
  parallel_strategy.def(py::init())
      .def_property(
          "nranks",
          [](const imperative::ParallelStrategy &self) { return self.nranks_; },
          [](imperative::ParallelStrategy &self, int nranks) {
            self.nranks_ = nranks;
          })
      .def_property(
          "local_rank",
          [](const imperative::ParallelStrategy &self) {
            return self.local_rank_;
          },
          [](imperative::ParallelStrategy &self, int local_rank) {
            self.local_rank_ = local_rank;
          })
      .def_property(
          "trainer_endpoints",
          [](const imperative::ParallelStrategy &self) {
            return self.trainer_endpoints_;
          },
          [](imperative::ParallelStrategy &self, std::vector<std::string> eps) {
            self.trainer_endpoints_ = eps;
          })
      .def_property(
          "current_endpoint",
          [](const imperative::ParallelStrategy &self) {
            return self.current_endpoint_;
          },
          [](imperative::ParallelStrategy &self, const std::string &ep) {
            self.current_endpoint_ = ep;
          })
      .def_property(
          "nrings",
          [](const imperative::ParallelStrategy &self) { return self.nrings_; },
          [](imperative::ParallelStrategy &self, int nrings) {
            self.nrings_ = nrings;
          });

  m.def("varbase_copy", &VarBaseCopy<platform::Place>);
  m.def("varbase_copy", &VarBaseCopy<platform::CPUPlace>);
  m.def("varbase_copy", &VarBaseCopy<platform::CUDAPlace>);
  m.def("varbase_copy", &VarBaseCopy<platform::XPUPlace>);
  m.def("varbase_copy", &VarBaseCopy<platform::CUDAPinnedPlace>);
  m.def("varbase_copy", &VarBaseCopy<platform::NPUPlace>);
  m.def("varbase_copy", &VarBaseCopy<platform::CustomPlace>);
  m.def("varbase_copy", &VarBaseCopy<platform::MLUPlace>);

  m.def(
      "dygraph_partial_grad",
      [](const std::vector<std::shared_ptr<imperative::VarBase>> &input_targets,
         const std::vector<std::shared_ptr<imperative::VarBase>>
             &output_targets,
         const std::vector<std::shared_ptr<imperative::VarBase>> &output_grads,
         const std::vector<std::shared_ptr<imperative::VarBase>> &no_grad_vars,
         const platform::Place &place,
         bool create_graph,
         bool retain_graph,
         bool allow_unused,
         bool only_inputs) {
        imperative::PartialGradEngine engine(input_targets,
                                             output_targets,
                                             output_grads,
                                             no_grad_vars,
                                             place,
                                             create_graph,
                                             retain_graph,
                                             allow_unused,
                                             only_inputs);
        engine.Execute();
        return engine.GetResult();
      },
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "dygraph_run_backward",
      [](const std::vector<std::shared_ptr<imperative::VarBase>> &tensors,
         const std::vector<std::shared_ptr<imperative::VarBase>> &grad_tensors,
         bool retain_graph,
         const imperative::Tracer &tracer) {
        auto *engine = tracer.GetEngine();
        engine->Init(tensors, grad_tensors, retain_graph);
        VLOG(3) << "Start backward";
        engine->Execute();
        VLOG(3) << "Finish backward";
      },
      py::call_guard<py::gil_scoped_release>());

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) ||          \
    defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_ASCEND_CL) || \
    defined(PADDLE_WITH_GLOO) || defined(PADDLE_WITH_CNCL)
  py::class_<imperative::ParallelContext,
             std::shared_ptr<imperative::ParallelContext>>(m,
                                                           "ParallelContext");

  py::class_<imperative::Reducer, std::shared_ptr<imperative::Reducer>>(
      m, "Reducer", R"DOC()DOC")
      .def(py::init<const std::vector<std::shared_ptr<imperative::VarBase>> &,
                    const std::vector<std::vector<size_t>> &,
                    const std::vector<bool> &,
                    std::shared_ptr<imperative::ParallelContext>,
                    const std::vector<size_t> &,
                    bool>())
      .def("prepare_for_backward",
           &imperative::Reducer::PrepareForBackward,
           py::arg("vars"),
           py::call_guard<py::gil_scoped_release>());

  m.def("assign_group_by_size",
        &imperative::AssignGroupBySize,
        py::arg("vars"),
        py::arg("is_sparse_gradient"),
        py::arg("group_size_limits") = std::vector<size_t>{25 * 1024 * 1024},
        py::arg("tensor_indices") = std::vector<int64_t>{},
        py::call_guard<py::gil_scoped_release>());
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  py::class_<imperative::NCCLParallelContext,
             imperative::ParallelContext,
             std::shared_ptr<imperative::NCCLParallelContext>>(
      m, "NCCLParallelContext")
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::CUDAPlace &>())
      .def("init", [](imperative::NCCLParallelContext &self) { self.Init(); })
      .def("init_with_ring_id",
           &imperative::NCCLParallelContext::InitWithRingID,
           py::arg("ring_id"));
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
  py::class_<imperative::BKCLParallelContext,
             imperative::ParallelContext,
             std::shared_ptr<imperative::BKCLParallelContext>>(
      m, "BKCLParallelContext")
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::XPUPlace &>())
      .def("init", [](imperative::BKCLParallelContext &self) { self.Init(); })
      .def("init_with_ring_id",
           &imperative::BKCLParallelContext::InitWithRingID,
           py::arg("ring_id"));
#endif

#if defined(PADDLE_WITH_GLOO)
  // xiongkun
  py::class_<imperative::GLOOParallelContext,
             imperative::ParallelContext,
             std::shared_ptr<imperative::GLOOParallelContext>>(
      m, "GLOOParallelContext")
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::CPUPlace &>())
      .def("init", [](imperative::GLOOParallelContext &self) { self.Init(); })
      .def("init_with_ring_id",
           &imperative::GLOOParallelContext::InitWithRingID,
           py::arg("ring_id"));
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
  py::class_<imperative::HCCLParallelContext,
             imperative::ParallelContext,
             std::shared_ptr<imperative::HCCLParallelContext>>(
      m, "HCCLParallelContext")
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::NPUPlace &>())
      .def("init", [](imperative::HCCLParallelContext &self) { self.Init(); })
      .def("init_with_ring_id",
           &imperative::HCCLParallelContext::InitWithRingID,
           py::arg("ring_id"));
#endif

#if defined(PADDLE_WITH_CNCL)
  py::class_<imperative::CNCLParallelContext,
             imperative::ParallelContext,
             std::shared_ptr<imperative::CNCLParallelContext>>(
      m, "CNCLParallelContext")
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::MLUPlace &>())
      .def("init", [](imperative::CNCLParallelContext &self) { self.Init(); })
      .def("init_with_ring_id",
           &imperative::CNCLParallelContext::InitWithRingID,
           py::arg("ring_id"));
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_ASCEND_CL)
  py::class_<imperative::HeterParallelContext,
             imperative::ParallelContext,
             std::shared_ptr<imperative::HeterParallelContext>>(
      m, "HeterParallelContext")
      .def(py::init<const imperative::ParallelStrategy &, const int &>())
      .def("init", [](imperative::HeterParallelContext &self) { self.Init(); });
#endif

  m.def("pylayer_apply",
        [](const platform::CPUPlace &place,
           const py::object &cls,
           const py::args args,
           const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });

  m.def("pylayer_apply",
        [](const platform::CUDAPlace &place,
           const py::object &cls,
           const py::args args,
           const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });

  m.def("pylayer_apply",
        [](const platform::XPUPlace &place,
           const py::object &cls,
           const py::args args,
           const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });

  m.def("pylayer_apply",
        [](const platform::CUDAPinnedPlace &place,
           const py::object &cls,
           const py::args args,
           const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });

  m.def("pylayer_apply",
        [](const platform::NPUPlace &place,
           const py::object &cls,
           const py::args args,
           const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });
  m.def("pylayer_apply",
        [](const platform::MLUPlace &place,
           const py::object &cls,
           const py::args args,
           const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });
  m.def("pylayer_apply",
        [](const platform::CustomPlace &place,
           const py::object &cls,
           const py::args args,
           const py::kwargs kwargs) {
          return imperative::PyLayerApply(place, cls, args, kwargs);
        });

#if defined(PADDLE_WITH_CUDA)
  m.def(
      "to_uva_tensor",
      [](const py::object &obj, int device_id) {
        const auto &tracer = imperative::GetCurrentTracer();
        auto new_tensor = std::shared_ptr<imperative::VarBase>(
            new imperative::VarBase(tracer->GenerateUniqueName()));
        auto array = obj.cast<py::array>();
        if (py::isinstance<py::array_t<int32_t>>(array)) {
          SetUVATensorFromPyArray<int32_t>(new_tensor, array, device_id);
        } else if (py::isinstance<py::array_t<int64_t>>(array)) {
          SetUVATensorFromPyArray<int64_t>(new_tensor, array, device_id);
        } else if (py::isinstance<py::array_t<float>>(array)) {
          SetUVATensorFromPyArray<float>(new_tensor, array, device_id);
        } else if (py::isinstance<py::array_t<double>>(array)) {
          SetUVATensorFromPyArray<double>(new_tensor, array, device_id);
        } else if (py::isinstance<py::array_t<int8_t>>(array)) {
          SetUVATensorFromPyArray<int8_t>(new_tensor, array, device_id);
        } else if (py::isinstance<py::array_t<int16_t>>(array)) {
          SetUVATensorFromPyArray<int16_t>(new_tensor, array, device_id);
        } else if (py::isinstance<py::array_t<paddle::platform::float16>>(
                       array)) {
          SetUVATensorFromPyArray<paddle::platform::float16>(
              new_tensor, array, device_id);
        } else if (py::isinstance<py::array_t<bool>>(array)) {
          SetUVATensorFromPyArray<bool>(new_tensor, array, device_id);
        } else {
          // obj may be any type, obj.cast<py::array>() may be failed,
          // then the array.dtype will be string of unknown meaning.
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Input object type error or incompatible array data type. "
              "tensor.set() supports array with bool, float16, float32, "
              "float64, int8, int16, int32, int64,"
              "please check your input or input array data type."));
        }
        return new_tensor;
      },
      py::arg("obj"),
      py::arg("device_id") = 0,
      py::return_value_policy::reference,
      R"DOC(
  Returns tensor with the UVA(unified virtual addressing) created from numpy array.

  Args:
      obj(numpy.ndarray): The input numpy array, supporting bool, float16, float32,
                          float64, int8, int16, int32, int64 dtype currently.

      device_id(int, optional): The destination GPU device id.
                                Default: 0, means current device.

  Returns:

      new_tensor(paddle.Tensor): Return the UVA Tensor with the sample dtype and
                                 shape with the input numpy array.

  Examples:
      .. code-block:: python

        # required: gpu
        import numpy as np
        import paddle

        data = np.random.randint(10, size=(3, 4))
        tensor = paddle.fluid.core.to_uva_tensor(data)
        print(tensor)
)DOC");

#endif

#if defined(PADDLE_WITH_CUDA)
  m.def(
      "async_write",
      [](const imperative::VarBase &src,
         imperative::VarBase &dst,
         const imperative::VarBase &offset,
         const imperative::VarBase &count) {
        PADDLE_ENFORCE_EQ(
            platform::is_gpu_place(src.Place()),
            true,
            platform::errors::InvalidArgument(
                "Required `src` device should be CUDAPlace, but received %d. ",
                src.Place()));
        PADDLE_ENFORCE_EQ(
            platform::is_cuda_pinned_place(dst.Place()),
            true,
            platform::errors::InvalidArgument(
                "Required `dst` device should be CUDAPinnedPlace, "
                "but received %d. ",
                dst.Place()));
        PADDLE_ENFORCE_EQ(
            platform::is_cpu_place(offset.Place()),
            true,
            platform::errors::InvalidArgument("Required `offset` device should "
                                              "be CPUPlace, but received %d. ",
                                              offset.Place()));
        PADDLE_ENFORCE_EQ(
            platform::is_cpu_place(count.Place()),
            true,
            platform::errors::InvalidArgument(
                "Required `count` device should be CPUPlace, but received %d. ",
                count.Place()));

        // TODO(daisiming): In future, add index as arguments following
        // async_read.
        auto &src_tensor = src.Var().Get<framework::LoDTensor>();
        auto *dst_tensor = dst.MutableVar()->GetMutable<framework::LoDTensor>();
        auto &offset_tensor = offset.Var().Get<framework::LoDTensor>();
        auto &count_tensor = count.Var().Get<framework::LoDTensor>();
        const auto &deviceId = paddle::platform::GetCurrentDeviceId();

        PADDLE_ENFORCE_EQ(offset_tensor.dims().size(),
                          1,
                          platform::errors::InvalidArgument(
                              "`offset` tensor should be one-dimensional."));
        PADDLE_ENFORCE_EQ(count_tensor.dims().size(),
                          1,
                          platform::errors::InvalidArgument(
                              "`count` tensor should be one-dimensional."));
        PADDLE_ENFORCE_EQ(offset_tensor.numel(),
                          count_tensor.numel(),
                          platform::errors::InvalidArgument(
                              "`offset` and `count` tensor size dismatch."));
        PADDLE_ENFORCE_EQ(
            src_tensor.dims().size(),
            dst_tensor->dims().size(),
            platform::errors::InvalidArgument(
                "`src` and `dst` should have the same tensor shape, "
                "except for the first dimension."));
        for (int i = 1; i < src_tensor.dims().size(); i++) {
          PADDLE_ENFORCE_EQ(
              src_tensor.dims()[i],
              dst_tensor->dims()[i],
              platform::errors::InvalidArgument(
                  "`src` and `dst` should have the same tensor shape, "
                  "except for the first dimension."));
        }

        auto stream =
            paddle::platform::get_current_stream(deviceId)->raw_stream();

        int64_t size = src_tensor.numel() / src_tensor.dims()[0];
        auto *src_data = src_tensor.data<float>();
        auto *dst_data = dst_tensor->mutable_data<float>(dst.Place());
        const int64_t *offset_data = offset_tensor.data<int64_t>();
        const int64_t *count_data = count_tensor.data<int64_t>();
        int64_t src_offset = 0, dst_offset, c;
        for (int64_t i = 0; i < offset_tensor.numel(); i++) {
          dst_offset = offset_data[i], c = count_data[i];
          PADDLE_ENFORCE_LE(src_offset + c,
                            src_tensor.dims()[0],
                            platform::errors::InvalidArgument(
                                "Invalid offset or count index"));
          PADDLE_ENFORCE_LE(dst_offset + c,
                            dst_tensor->dims()[0],
                            platform::errors::InvalidArgument(
                                "Invalid offset or count index"));
          cudaMemcpyAsync(dst_data + (dst_offset * size),
                          src_data + (src_offset * size),
                          c * size * sizeof(float),
                          cudaMemcpyDeviceToHost,
                          stream);
          src_offset += c;
        }
      },
      R"DOC(
  This api provides a way to write pieces of source tensor to destination tensor
  inplacely and asynchronously. In which, we use `offset` and `count` to determine
  where to copy. `offset` means the begin points of the copy pieces of `src`, and
  `count` means the lengths of the copy pieces of `src`. To be noted, the copy process
  will run asynchronously from cuda to pin memory. We can simply remember this as
  "gpu async_write to pin_memory".

  Arguments:

    src (Tensor): The source tensor, and the data type should be `float32` currently.
                  Besides, `src` should be placed on CUDAPlace.

    dst (Tensor): The destination tensor, and the data type should be `float32` currently.
                  Besides, `dst` should be placed on CUDAPinnedPlace. The shape of `dst`
                  should be the same with `src` except for the first dimension.

    offset (Tensor): The offset tensor, and the data type should be `int64` currently.
                     Besides, `offset` should be placed on CPUPlace. The shape of `offset`
                     should be one-dimensional.

    count (Tensor): The count tensor, and the data type should be `int64` currently.
                    Besides, `count` should be placed on CPUPlace. The shape of `count`
                    should be one-dimensinal.

  Examples:
      .. code-block:: python

          import numpy as np
          import paddle
          from paddle.fluid import core
          from paddle.device import cuda

          if core.is_compiled_with_cuda():
              src = paddle.rand(shape=[100, 50, 50])
              dst = paddle.emtpy(shape=[200, 50, 50]).pin_memory()
              offset = paddle.to_tensor(
                  np.array([0, 60], dtype="int64"), place=paddle.CPUPlace())
              count = paddle.to_tensor(
                  np.array([40, 60], dtype="int64"), place=paddle.CPUPlace())

              stream = cuda.Stream()
              with cuda.stream_guard(stream):
                  core.async_write(src, dst, offset, count)

              offset_a = paddle.gather(dst, paddle.to_tensor(np.arange(0, 40)))
              offset_b = paddle.gather(dst, paddle.to_tensor(np.arange(60, 120)))
              offset_array = paddle.concat([offset_a, offset_b], axis=0)
              print(np.allclose(src.numpy(), offset_array.numpy())) # True
)DOC");

  m.def(
      "async_read",
      [](const imperative::VarBase &src,
         imperative::VarBase &dst,
         const imperative::VarBase &index,
         imperative::VarBase &buffer,
         const imperative::VarBase &offset,
         const imperative::VarBase &count) {
        PADDLE_ENFORCE_EQ(platform::is_cuda_pinned_place(src.Place()),
                          true,
                          platform::errors::InvalidArgument(
                              "Required `src` device should be "
                              "CUDAPinnedPlace, but received %d.",
                              src.Place()));
        PADDLE_ENFORCE_EQ(
            platform::is_gpu_place(dst.Place()),
            true,
            platform::errors::InvalidArgument(
                "Required `dst` device should be CUDAPlace, but received %d.",
                dst.Place()));
        PADDLE_ENFORCE_EQ(
            platform::is_cpu_place(index.Place()),
            true,
            platform::errors::InvalidArgument(
                "Required `index` device should be CPUPlace, but received %d.",
                index.Place()));
        PADDLE_ENFORCE_EQ(
            platform::is_cuda_pinned_place(buffer.Place()),
            true,
            platform::errors::InvalidArgument(
                "Required `buffer` device should be CUDAPinnedPlace, "
                "but received %d.",
                buffer.Place()));
        PADDLE_ENFORCE_EQ(
            platform::is_cpu_place(offset.Place()),
            true,
            platform::errors::InvalidArgument(
                "Required `offset` device should be CPUPlace, but received %d.",
                offset.Place()));
        PADDLE_ENFORCE_EQ(
            platform::is_cpu_place(count.Place()),
            true,
            platform::errors::InvalidArgument(
                "Required `count` device should be CPUPlace, but received %d.",
                count.Place()));

        auto &src_tensor = src.Var().Get<framework::LoDTensor>();
        auto *dst_tensor = dst.MutableVar()->GetMutable<framework::LoDTensor>();
        auto &index_tensor = index.Var().Get<framework::LoDTensor>();
        auto *buffer_tensor =
            buffer.MutableVar()->GetMutable<framework::LoDTensor>();
        auto &offset_tensor = offset.Var().Get<framework::LoDTensor>();
        auto &count_tensor = count.Var().Get<framework::LoDTensor>();
        auto *dst_data = dst_tensor->mutable_data<float>(dst.Place());
        const auto &deviceId = paddle::platform::GetCurrentDeviceId();

        PADDLE_ENFORCE_EQ(src_tensor.dims().size(),
                          dst_tensor->dims().size(),
                          platform::errors::InvalidArgument(
                              "`src` and `dst` should have same tensor shape, "
                              "except for the first dimension."));
        PADDLE_ENFORCE_EQ(
            src_tensor.dims().size(),
            buffer_tensor->dims().size(),
            platform::errors::InvalidArgument(
                "`src` and `buffer` should have same tensor shape, "
                "except for the first dimension."));
        for (int i = 1; i < src_tensor.dims().size(); i++) {
          PADDLE_ENFORCE_EQ(
              src_tensor.dims()[i],
              dst_tensor->dims()[i],
              platform::errors::InvalidArgument(
                  "`src` and `dst` should have the same tensor shape, "
                  "except for the first dimension."));
          PADDLE_ENFORCE_EQ(
              src_tensor.dims()[i],
              buffer_tensor->dims()[i],
              platform::errors::InvalidArgument(
                  "`src` and `buffer` should have the same tensor shape, "
                  "except for the first dimension."));
        }
        PADDLE_ENFORCE_EQ(index_tensor.dims().size(),
                          1,
                          platform::errors::InvalidArgument(
                              "`index` tensor should be one-dimensional."));

        auto stream =
            paddle::platform::get_current_stream(deviceId)->raw_stream();

        int64_t numel = 0;  // total copy length
        int64_t copy_flag = offset_tensor.dims()[0];
        int64_t size = src_tensor.numel() / src_tensor.dims()[0];

        if (copy_flag != 0) {
          PADDLE_ENFORCE_EQ(offset_tensor.dims().size(),
                            1,
                            platform::errors::InvalidArgument(
                                "`offset` tensor should be one-dimensional."));
          PADDLE_ENFORCE_EQ(count_tensor.dims().size(),
                            1,
                            platform::errors::InvalidArgument(
                                "`count` tensor should be one-dimensional."));
          PADDLE_ENFORCE_EQ(offset_tensor.numel(),
                            count_tensor.numel(),
                            platform::errors::InvalidArgument(
                                "`offset` and `count` tensor size dismatch."));
          auto *offset_data = offset_tensor.data<int64_t>();
          auto *count_data = count_tensor.data<int64_t>();
          for (int64_t i = 0; i < count_tensor.numel(); i++) {
            numel += count_data[i];
          }
          PADDLE_ENFORCE_LE(numel + index_tensor.numel(),
                            buffer_tensor->dims()[0],
                            platform::errors::InvalidArgument(
                                "Buffer tensor size is too small."));
          PADDLE_ENFORCE_LE(numel + index_tensor.numel(),
                            dst_tensor->dims()[0],
                            platform::errors::InvalidArgument(
                                "Target tensor size is too small."));

          int64_t src_offset, dst_offset = 0, c;
          auto *src_data = src_tensor.data<float>();
          for (int64_t i = 0; i < offset_tensor.numel(); i++) {
            src_offset = offset_data[i], c = count_data[i];
            PADDLE_ENFORCE_LE(src_offset + c,
                              src_tensor.dims()[0],
                              platform::errors::InvalidArgument(
                                  "Invalid offset or count index."));
            PADDLE_ENFORCE_LE(dst_offset + c,
                              dst_tensor->dims()[0],
                              platform::errors::InvalidArgument(
                                  "Invalid offset or count index."));
            cudaMemcpyAsync(dst_data + (dst_offset * size),
                            src_data + (src_offset * size),
                            c * size * sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream);
            dst_offset += c;
          }
        } else {
          PADDLE_ENFORCE_LE(index_tensor.numel(),
                            buffer_tensor->dims()[0],
                            platform::errors::InvalidArgument(
                                "Buffer tensor size is too small."));
        }

        // Select the index data to the buffer
        auto index_select = [](const phi::DenseTensor &src_tensor,
                               const phi::DenseTensor &index_tensor,
                               phi::DenseTensor *buffer_tensor) {
          auto *src_data = src_tensor.data<float>();
          auto *index_data = index_tensor.data<int64_t>();
          auto *buffer_data =
              buffer_tensor->mutable_data<float>(buffer_tensor->place());
          const int &slice_size = src_tensor.numel() / src_tensor.dims()[0];
          const int &copy_bytes = slice_size * sizeof(float);
          int64_t c = 0;
          for (int64_t i = 0; i < index_tensor.numel(); i++) {
            std::memcpy(buffer_data + c * slice_size,
                        src_data + index_data[i] * slice_size,
                        copy_bytes);
            c += 1;
          }
        };
        index_select(src_tensor, index_tensor, buffer_tensor);

        // Copy the data to device memory
        cudaMemcpyAsync(dst_data + (numel * size),
                        buffer_tensor->data<float>(),
                        index_tensor.numel() * size * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream);
      },
      R"DOC(
  This api provides a way to read from pieces of source tensor to destination tensor
  asynchronously. In which, we use `index`, `offset` and `count` to determine where
  to read. `index` means the index position of src tensor we want to read. `offset`
  and count means the begin points and length of pieces of src tensor we want to read.
  To be noted, the copy process will run asynchronously from pin memory to cuda place.
  We can simply remember this as "cuda async_read from pin_memory".

  Arguments:

    src (Tensor): The source tensor, and the data type should be `float32` currently.
                  Besides, `src` should be placed on CUDAPinnedPlace.

    dst (Tensor): The destination tensor, and the data type should be `float32` currently.
                  Besides, `dst` should be placed on CUDAPlace. The shape of `dst` should
                  be the same with `src` except for the first dimension.

    index (Tensor): The index tensor, and the data type should be `int64` currently.
                    Besides, `index` should be on CPUplace. The shape of `index` should
                    be one-dimensional.

    buffer (Tensor): The buffer tensor, used to buffer index copy tensor temporarily.
                     The data type should be `float32` currently, and should be placed
                     on CUDAPinnedPlace. The shape of `buffer` should be the same with `src` except for the first dimension.

    offset (Tensor): The offset tensor, and the data type should be `int64` currently.
                     Besides, `offset` should be placed on CPUPlace. The shape of `offset`
                     should be one-dimensional.

    count (Tensor): The count tensor, and the data type should be `int64` currently.
                    Besides, `count` should be placed on CPUPlace. The shape of `count`
                    should be one-dimensinal.

  Examples:
      .. code-block:: python

          import numpy as np
          import paddle
          from paddle.fluid import core
          from paddle.device import cuda

          if core.is_compiled_with_cuda():
              src = paddle.rand(shape=[100, 50, 50], dtype="float32").pin_memory()
              dst = paddle.empty(shape=[100, 50, 50], dtype="float32")
              offset = paddle.to_tensor(
                  np.array([0, 60], dtype="int64"), place=paddle.CPUPlace())
              count = paddle.to_tensor(
                  np.array([40, 60], dtype="int64"), place=paddle.CPUPlace())
              buffer = paddle.empty(shape=[50, 50, 50], dtype="float32").pin_memory()
              index = paddle.to_tensor(
                  np.array([1, 3, 5, 7, 9], dtype="int64")).cpu()

              stream = cuda.Stream()
              with cuda.stream_guard(stream):
                  core.async_read(src, dst, index, buffer, offset, count)

)DOC");
#endif
}

}  // namespace pybind
}  // namespace paddle
