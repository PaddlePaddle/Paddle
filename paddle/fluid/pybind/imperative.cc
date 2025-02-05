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
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

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
#include "paddle/fluid/imperative/data_loader.h"
#include "paddle/fluid/imperative/gloo_context.h"
#include "paddle/fluid/imperative/heter_ccl_context.h"
#include "paddle/fluid/imperative/hooks.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/nccl_context.h"
#include "paddle/fluid/imperative/partial_grad_engine.h"
#include "paddle/fluid/imperative/profiler.h"
#include "paddle/fluid/imperative/reducer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/xccl_context.h"
#include "paddle/fluid/pybind/cuda_streams_py.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/fluid/pybind/slice_utils.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/fluid/pybind/uva_utils.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/phi/core/memory/allocation/mmap_allocator.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/type_defs.h"

namespace paddle::pybind {

std::atomic<int> VarBaseUniqueNameID{0};

namespace py = ::pybind11;

class PyVariableWrapperHook : public imperative::VariableWrapperHook {
 public:
  explicit PyVariableWrapperHook(PyObject *func) : py_func_(func) {
    Py_INCREF(py_func_);
  }

  ~PyVariableWrapperHook() override {  // NOLINT
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
      throw e;
    } catch (std::exception &e) {
      PADDLE_THROW(common::errors::Unavailable(
          "Hook function of Tensor raises an exception: %s.", e.what()));
    } catch (...) {
      PADDLE_THROW(common::errors::Fatal(
          "Hook function of Tensor raises an unknown exception."));
    }

    PADDLE_ENFORCE_NOT_NULL(res,
                            common::errors::Unavailable(
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

static const phi::Place PyObjectToPlace(const py::object &place_obj) {
  if (py::isinstance<phi::CPUPlace>(place_obj)) {
    return place_obj.cast<phi::CPUPlace>();
  } else if (py::isinstance<phi::GPUPlace>(place_obj)) {
    return place_obj.cast<phi::GPUPlace>();
  } else if (py::isinstance<phi::XPUPlace>(place_obj)) {
    return place_obj.cast<phi::XPUPlace>();
  } else if (py::isinstance<phi::GPUPinnedPlace>(place_obj)) {
    return place_obj.cast<phi::GPUPinnedPlace>();
  } else if (py::isinstance<phi::IPUPlace>(place_obj)) {
    return place_obj.cast<phi::IPUPlace>();
  } else if (py::isinstance<phi::Place>(place_obj)) {
    return place_obj.cast<phi::Place>();
  } else if (py::isinstance<phi::CustomPlace>(place_obj)) {
    return place_obj.cast<phi::CustomPlace>();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Place should be one of "
        "Place/CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace/IPUPlace/"
        "CustomPlace"));
  }
}

// only initialize varbase, but not its tensor.
static void InitVarBaseOnly(imperative::VarBase *self,
                            const std::string &name,
                            bool persistable = false,
                            int stop_gradient = -1) {
  auto name_ = name.empty()
                   ? imperative::GetCurrentTracer()->GenerateUniqueName(
                         "generated_tensor")
                   : name;

  VLOG(5) << "Init Tensor as: / name: " << name_
          << " / persistable: " << persistable
          << " / stop_gradient: " << stop_gradient;
  new (self) imperative::VarBase(name_);
  if (stop_gradient != -1) {
    self->SetOverriddenStopGradient(stop_gradient);
  }
  self->SetPersistable(persistable);
  self->SetType(framework::proto::VarType::DENSE_TENSOR);
}

// initialize varbase and its tensor.
static void InitVarBaseAndTensor(imperative::VarBase *self,
                                 const py::array &array,
                                 const phi::Place &place,
                                 const std::string &name,
                                 bool persistable = false,
                                 bool zero_copy = false,
                                 int stop_gradient = -1) {
  InitVarBaseOnly(self, name, persistable, stop_gradient);
  auto *tensor = self->MutableVar()->GetMutable<phi::DenseTensor>();
  VLOG(4) << "zero_copy: " << zero_copy;
  if (phi::is_cpu_place(place)) {
    SetTensorFromPyArray<phi::CPUPlace>(tensor, array, place, zero_copy);
  } else if (phi::is_xpu_place(place)) {
    SetTensorFromPyArray<phi::XPUPlace>(tensor, array, place, zero_copy);
  } else if (phi::is_gpu_place(place)) {
    SetTensorFromPyArray<phi::GPUPlace>(tensor, array, place, zero_copy);
  } else if (phi::is_cuda_pinned_place(place)) {
    SetTensorFromPyArray<phi::GPUPinnedPlace>(tensor, array, place, zero_copy);
  } else if (phi::is_ipu_place(place)) {
    SetTensorFromPyArray<phi::IPUPlace>(tensor, array, place, zero_copy);
  } else if (phi::is_custom_place(place)) {
    SetTensorFromPyArray<phi::CustomPlace>(tensor, array, place, zero_copy);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Place should be one of "
        "CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace/IPUPlace/"));
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
  if (name.empty()) {
    name =
        imperative::GetCurrentTracer()->GenerateUniqueName("generated_tensor");
  }
  VLOG(5) << "Init Tensor as: / name: " << name
          << " / persistable: " << persistable << " / zero_copy: " << zero_copy
          << " / stop_gradient: " << stop_gradient << " / at " << place;
  new (self) imperative::VarBase(name);
  self->SetPersistable(persistable);
  auto *tensor = self->MutableVar()->GetMutable<phi::DenseTensor>();
  if (stop_gradient != -1) {
    self->SetOverriddenStopGradient(stop_gradient);
  }
  SetTensorFromPyArray<P>(tensor, array, place, zero_copy);
  self->SetType(framework::proto::VarType::DENSE_TENSOR);
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
  auto name_ = name.empty()
                   ? imperative::GetCurrentTracer()->GenerateUniqueName(
                         "generated_tensor")
                   : name;
  new (self) imperative::VarBase(name_);
  self->SetPersistable(false);
  self->SetType(framework::proto::VarType::DENSE_TENSOR);
  self->SetDataType(framework::TransToProtoVarType(tensor.dtype()));
  auto *new_tensor = self->MutableVar()->GetMutable<phi::DenseTensor>();
  // Same place, share data directly
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
  auto name_ = name.empty()
                   ? imperative::GetCurrentTracer()->GenerateUniqueName(
                         "generated_tensor")
                   : name;
  new (self) imperative::VarBase(name_);
  self->SetPersistable(false);
  self->SetType(framework::proto::VarType::DENSE_TENSOR);
  self->SetDataType(framework::TransToProtoVarType(tensor.dtype()));
  auto *new_tensor = self->MutableVar()->GetMutable<phi::DenseTensor>();
  // Same place, share data directly
  if (phi::is_same_place(place, tensor.place())) {
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
            .Get<phi::DenseTensor>());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "We should only get paddle::Tensor or VarBase in this "
        "method, when you reach this means we got another type index."));
  }
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
          py_ivar, common::errors::InvalidArgument("Python Object is NULL"));
      result.emplace_back(
          PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar));
    }
  } else if (PyTuple_Check(py_obj)) {  // Tuple of VarBase
    size_t len = PyTuple_GET_SIZE(py_obj);
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      PyObject *py_ivar = PyTuple_GET_ITEM(py_obj, i);
      PADDLE_ENFORCE_NOT_NULL(
          py_ivar, common::errors::InvalidArgument("Python Object is NULL"));
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
      common::errors::InvalidArgument(py::str(py::handle(PyErr_Occurred()))));
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
      common::errors::InvalidArgument(py::str(py::handle(PyErr_Occurred()))));
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
    dst.SetOverriddenStopGradient(src->OverriddenStopGradient());
    if (!src->SharedVar()->IsEmpty()) {
      if (src->Var().IsType<phi::DenseTensor>()) {
        auto &src_tensor = src->Var().Get<phi::DenseTensor>();
        auto *dst_tensor = dst.MutableVar()->GetMutable<phi::DenseTensor>();
        framework::TensorCopy(src_tensor, dst_device, dst_tensor);
        if (blocking) {
          phi::DeviceContextPool::Instance().Get(dst_device)->Wait();
          auto src_device = src_tensor.place();
          if (!(src_device == dst_device)) {
            phi::DeviceContextPool::Instance().Get(src_device)->Wait();
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
          phi::DeviceContextPool::Instance().Get(dst_device)->Wait();
          auto src_device = src_selected_rows.value().place();
          if (!(src_device == dst_device)) {
            phi::DeviceContextPool::Instance().Get(src_device)->Wait();
          }
        }
      }

      if (!blocking) {
        IncreaseVarbaseReferenceCountUntilCopyComplete(src, dst_device);
      }

    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The source Tensor(%s) can not copy when it is empty.", src->Name()));
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The destination Tensor(%s) can not copy when it is not empty.",
        dst.Name()));
  }
}

// Bind Methods
void BindImperative(py::module *m_ptr) {
  auto &m = *m_ptr;

#ifndef _WIN32
  // Dygraph DataLoader signal handler
  m.def("_set_process_pids", [](int64_t key, py::object &obj) {
    PADDLE_ENFORCE_EQ(
        py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj),
        true,
        common::errors::InvalidArgument(
            "The subprocess ids set in DataLoader is illegal."
            "Expected data type is tuple or list, but received %s",
            obj.get_type()));
    py::list pids = py::cast<py::list>(obj);
    std::set<pid_t> pids_set = {};
    for (auto &&pid : pids) {
      pids_set.insert(pid.cast<pid_t>());
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
            common::errors::InvalidArgument(
                "The batch data read into DataLoader is illegal."
                "Expected data type is tuple or list, but received %s",
                obj.get_type()));
        py::list batch = py::cast<py::list>(obj);
        py::list tensors;
        for (auto &&item : batch) {
          // 1. cast to python array
          auto array = item.cast<py::array>();
          PADDLE_ENFORCE_NE(
              string::Sprintf("%s", array.dtype()).compare("object"),
              0,
              common::errors::InvalidArgument(
                  "Failed to convert input data to a regular ndarray.\n  * "
                  "Usually this means the input data contains nested "
                  "lists with different lengths.\n  * Check the reader "
                  "function passed to 'set_(sample/sample_list/batch)"
                  "_generator' to locate the data causes this issue."));
          // 2. construct DenseTensor
          phi::DenseTensor t;
          SetTensorFromPyArray<phi::CPUPlace>(&t, array, phi::CPUPlace(), true);
          // 3. allocate shared memory
          void *data_ptr = t.data();
          size_t data_size = t.numel() * phi::SizeOf(t.dtype());
          auto shared_writer_holder =
              memory::allocation::AllocateMemoryMapWriterAllocation(data_size);
          // 4. maintain mmap fd set & backup ipc_name
          const std::string &ipc_name = shared_writer_holder->ipc_name();
          memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);
          // 5. copy data & reset holder
          memory::Copy(phi::CPUPlace(),
                       shared_writer_holder->ptr(),
                       phi::CPUPlace(),
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
            common::errors::InvalidArgument(
                "Failed to convert input data to a regular ndarray.\n  * "
                "Usually this means the input data contains nested "
                "lists with different lengths.\n  * Check the reader "
                "function passed to 'set_(sample/sample_list/batch)"
                "_generator' to locate the data causes this issue."));
        // 2. construct DenseTensor
        phi::DenseTensor t;
        SetTensorFromPyArray<phi::CPUPlace>(&t, array, phi::CPUPlace(), true);
        // 3. allocate shared memory
        void *data_ptr = t.data();
        size_t data_size = t.numel() * phi::SizeOf(t.dtype());
        auto shared_writer_holder =
            memory::allocation::AllocateMemoryMapWriterAllocation(data_size);
        // 4. maintain mmap fd set & backup ipc_name
        const std::string &ipc_name = shared_writer_holder->ipc_name();
        memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);
        // 5. copy data & reset holder
        memory::Copy(phi::CPUPlace(),
                     shared_writer_holder->ptr(),
                     phi::CPUPlace(),
                     data_ptr,
                     data_size);
        t.ResetHolder(shared_writer_holder);

        return t;
      },
      py::return_value_policy::take_ownership);

  m.def("_remove_tensor_list_mmap_fds", [](py::list &tensor_list) {
    for (auto &&tensor : tensor_list) {
      auto t = tensor.cast<phi::DenseTensor>();
      auto *mmap_writer_allocation =
          dynamic_cast<memory::allocation::MemoryMapWriterAllocation *>(
              t.Holder().get());
      PADDLE_ENFORCE_NOT_NULL(
          mmap_writer_allocation,
          common::errors::NotFound("The shared memory of DenseTensor in "
                                   "DataLoader's child process has been "
                                   "released."));
      memory::allocation::MemoryMapFdSet::Instance().Remove(
          mmap_writer_allocation->ipc_name());
    }
  });

  m.def("_cleanup_mmap_fds",
        []() { memory::allocation::MemoryMapFdSet::Instance().Clear(); });

  m.def("_set_max_memory_map_allocation_pool_size", [](int32_t size) {
    memory::allocation::MemoryMapAllocationPool::Instance().SetMaxPoolSize(
        size);
  });
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
  m.def("_has_grad", []() { return egr::Controller::Instance().HasGrad(); });
  m.def("_set_has_grad", [](bool has_grad) {
    return egr::Controller::Instance().SetHasGrad(has_grad);
  });
  m.def("_get_amp_attrs",
        []() { return egr::Controller::Instance().GetCurrentAmpAttrs(); });
  m.def("_set_amp_op_list",
        [](std::unordered_set<std::string> &allow_ops,
           std::unordered_set<std::string> &block_ops) {
          imperative::AmpOperators::Instance().GetMutableAllowOps()->swap(
              allow_ops);
          imperative::AmpOperators::Instance().GetMutableBlockOps()->swap(
              block_ops);
          VLOG(5) << "AMP operators changed, "
                  << imperative::AmpOperators::Instance();
        });
  m.def("_get_amp_op_list", []() {
    return std::make_tuple(
        *(imperative::AmpOperators::Instance().GetMutableAllowOps()),
        *(imperative::AmpOperators::Instance().GetMutableBlockOps()));
  });

  py::enum_<paddle::imperative::AmpLevel>(m, "AmpLevel", py::arithmetic())
      .value("O0", paddle::imperative::AmpLevel::O0)
      .value("OD", paddle::imperative::AmpLevel::OD)
      .value("O1", paddle::imperative::AmpLevel::O1)
      .value("O2", paddle::imperative::AmpLevel::O2)
      .value("O3", paddle::imperative::AmpLevel::O3)
      .export_values();

  py::class_<imperative::AmpAttrs, std::shared_ptr<imperative::AmpAttrs>>(
      m, "AmpAttrs", R"DOC()DOC")
      .def_property("_use_promote",
                    &imperative::AmpAttrs::GetUsePromote,
                    &imperative::AmpAttrs::SetUsePromote)
      .def_property("_amp_level",
                    &imperative::AmpAttrs::GetAmpLevel,
                    &imperative::AmpAttrs::SetAmpLevel)
      .def_property("_amp_dtype",
                    &imperative::AmpAttrs::GetAmpDtype,
                    &imperative::AmpAttrs::SetAmpDtype);

  py::class_<imperative::Tracer, std::shared_ptr<imperative::Tracer>>(
      m, "Tracer", R"DOC()DOC")
      .def(py::init([]() { return std::make_unique<imperative::Tracer>(); }))
      .def_property("_use_promote",
                    &imperative::Tracer::GetUsePromote,
                    &imperative::Tracer::SetUsePromote)
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
            if (py::isinstance<phi::GPUPlace>(obj)) {
              auto p = obj.cast<phi::GPUPlace *>();
              self.SetExpectedPlace(*p);
              // TODO(jiabin): Support eager here when we need to make all
              // dygraph in eager mode
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<phi::XPUPlace>(obj)) {
              auto p = obj.cast<phi::XPUPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<phi::CPUPlace>(obj)) {
              auto p = obj.cast<phi::CPUPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<phi::GPUPinnedPlace>(obj)) {
              auto p = obj.cast<phi::GPUPinnedPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<phi::IPUPlace>(obj)) {
              auto p = obj.cast<phi::IPUPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<phi::CustomPlace>(obj)) {
              auto p = obj.cast<phi::CustomPlace *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else if (py::isinstance<phi::Place>(obj)) {
              auto p = obj.cast<phi::Place *>();
              self.SetExpectedPlace(*p);
              VLOG(4) << "Tracer(" << &self << ")"
                      << " set expected place " << *p;
            } else {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "Incompatible Place Type: supports XPUPlace, CUDAPlace, "
                  "CPUPlace, IPUPlace"
                  "and CUDAPinnedPlace, "
                  "but got Unknown Type!"));
            }
          })
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

  m.def("varbase_copy", &VarBaseCopy<phi::Place>);
  m.def("varbase_copy", &VarBaseCopy<phi::CPUPlace>);
  m.def("varbase_copy", &VarBaseCopy<phi::GPUPlace>);
  m.def("varbase_copy", &VarBaseCopy<phi::XPUPlace>);
  m.def("varbase_copy", &VarBaseCopy<phi::GPUPinnedPlace>);
  m.def("varbase_copy", &VarBaseCopy<phi::CustomPlace>);

  m.def(
      "dygraph_partial_grad",
      [](const std::vector<std::shared_ptr<imperative::VarBase>> &input_targets,
         const std::vector<std::shared_ptr<imperative::VarBase>>
             &output_targets,
         const std::vector<std::shared_ptr<imperative::VarBase>> &output_grads,
         const std::vector<std::shared_ptr<imperative::VarBase>> &no_grad_vars,
         const phi::Place &place,
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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) ||     \
    defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_GLOO) || \
    defined(PADDLE_WITH_CUSTOM_DEVICE)
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
                    const phi::GPUPlace &>())
      .def("init", [](imperative::NCCLParallelContext &self) { self.Init(); })
      .def("init_with_ring_id",
           &imperative::NCCLParallelContext::InitWithRingID,
           py::arg("ring_id"));
#endif

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  py::class_<imperative::XCCLParallelContext,
             imperative::ParallelContext,
             std::shared_ptr<imperative::XCCLParallelContext>>(
      m, "XCCLParallelContext")
      .def(py::init<const imperative::ParallelStrategy &,
                    const phi::CustomPlace &>())
      .def("init", [](imperative::XCCLParallelContext &self) { self.Init(); })
      .def("init_with_ring_id",
           &imperative::XCCLParallelContext::InitWithRingID,
           py::arg("ring_id"));
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
  py::class_<imperative::BKCLParallelContext,
             imperative::ParallelContext,
             std::shared_ptr<imperative::BKCLParallelContext>>(
      m, "BKCLParallelContext")
      .def(py::init<const imperative::ParallelStrategy &,
                    const phi::XPUPlace &>())
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
                    const phi::CPUPlace &>())
      .def("init", [](imperative::GLOOParallelContext &self) { self.Init(); })
      .def("init_with_ring_id",
           &imperative::GLOOParallelContext::InitWithRingID,
           py::arg("ring_id"));
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_CUSTOM_DEVICE)
  py::class_<imperative::HeterParallelContext,
             imperative::ParallelContext,
             std::shared_ptr<imperative::HeterParallelContext>>(
      m, "HeterParallelContext")
      .def(py::init<const imperative::ParallelStrategy &, const int &>())
      .def("init", [](imperative::HeterParallelContext &self) { self.Init(); });
#endif

#if defined(PADDLE_WITH_CUDA)
  m.def(
      "to_uva_tensor",
      [](const py::object &obj, int device_id) {
        const auto &tracer = imperative::GetCurrentTracer();
        auto new_tensor =
            std::make_shared<imperative::VarBase>(tracer->GenerateUniqueName());
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
        } else if (py::isinstance<py::array_t<phi::dtype::float16>>(array)) {
          SetUVATensorFromPyArray<phi::dtype::float16>(
              new_tensor, array, device_id);
        } else if (py::isinstance<py::array_t<bool>>(array)) {
          SetUVATensorFromPyArray<bool>(new_tensor, array, device_id);
        } else {
          // obj may be any type, obj.cast<py::array>() may be failed,
          // then the array.dtype will be string of unknown meaning.
          PADDLE_THROW(common::errors::InvalidArgument(
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

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import numpy as np
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> data = np.random.randint(10, size=(3, 4))
            >>> tensor = paddle.base.core.to_uva_tensor(data)
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
            phi::is_gpu_place(src.Place()),
            true,
            common::errors::InvalidArgument(
                "Required `src` device should be CUDAPlace, but received %d. ",
                src.Place()));
        PADDLE_ENFORCE_EQ(
            phi::is_cuda_pinned_place(dst.Place()),
            true,
            common::errors::InvalidArgument(
                "Required `dst` device should be CUDAPinnedPlace, "
                "but received %d. ",
                dst.Place()));
        PADDLE_ENFORCE_EQ(
            phi::is_cpu_place(offset.Place()),
            true,
            common::errors::InvalidArgument("Required `offset` device should "
                                            "be CPUPlace, but received %d. ",
                                            offset.Place()));
        PADDLE_ENFORCE_EQ(
            phi::is_cpu_place(count.Place()),
            true,
            common::errors::InvalidArgument(
                "Required `count` device should be CPUPlace, but received %d. ",
                count.Place()));

        // TODO(daisiming): In future, add index as arguments following
        // async_read.
        auto &src_tensor = src.Var().Get<phi::DenseTensor>();
        auto *dst_tensor = dst.MutableVar()->GetMutable<phi::DenseTensor>();
        auto &offset_tensor = offset.Var().Get<phi::DenseTensor>();
        auto &count_tensor = count.Var().Get<phi::DenseTensor>();
        const auto &deviceId = paddle::platform::GetCurrentDeviceId();

        PADDLE_ENFORCE_EQ(offset_tensor.dims().size(),
                          1,
                          common::errors::InvalidArgument(
                              "`offset` tensor should be one-dimensional."));
        PADDLE_ENFORCE_EQ(count_tensor.dims().size(),
                          1,
                          common::errors::InvalidArgument(
                              "`count` tensor should be one-dimensional."));
        PADDLE_ENFORCE_EQ(offset_tensor.numel(),
                          count_tensor.numel(),
                          common::errors::InvalidArgument(
                              "`offset` and `count` tensor size dismatch."));
        PADDLE_ENFORCE_EQ(
            src_tensor.dims().size(),
            dst_tensor->dims().size(),
            common::errors::InvalidArgument(
                "`src` and `dst` should have the same tensor shape, "
                "except for the first dimension."));
        for (int i = 1; i < src_tensor.dims().size(); i++) {
          PADDLE_ENFORCE_EQ(
              src_tensor.dims()[i],
              dst_tensor->dims()[i],
              common::errors::InvalidArgument(
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
          PADDLE_ENFORCE_LE(
              src_offset + c,
              src_tensor.dims()[0],
              common::errors::InvalidArgument("Invalid offset or count index"));
          PADDLE_ENFORCE_LE(
              dst_offset + c,
              dst_tensor->dims()[0],
              common::errors::InvalidArgument("Invalid offset or count index"));
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
                    should be one-dimensional.

  Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> from paddle.base import core
            >>> from paddle.device import cuda
            >>> if core.is_compiled_with_cuda():
            ...     src = paddle.rand(shape=[100, 50, 50])
            ...     dst = paddle.empty(shape=[200, 50, 50]).pin_memory()
            ...     offset = paddle.to_tensor(
            ...         np.array([0, 60], dtype="int64"), place=paddle.CPUPlace())
            ...     count = paddle.to_tensor(
            ...         np.array([40, 60], dtype="int64"), place=paddle.CPUPlace())
            ...
            ...     stream = cuda.Stream()
            ...     with cuda.stream_guard(stream):
            ...         core.eager.async_write(src, dst, offset, count)
            ...
            ...     offset_a = paddle.gather(dst, paddle.to_tensor(np.arange(0, 40)))
            ...     offset_b = paddle.gather(dst, paddle.to_tensor(np.arange(60, 120)))
            ...     offset_array = paddle.concat([offset_a, offset_b], axis=0)
            ...     print(np.allclose(src.numpy(), offset_array.numpy()))
            True
)DOC");

  m.def(
      "async_read",
      [](const imperative::VarBase &src,
         imperative::VarBase &dst,
         const imperative::VarBase &index,
         imperative::VarBase &buffer,
         const imperative::VarBase &offset,
         const imperative::VarBase &count) {
        PADDLE_ENFORCE_EQ(
            phi::is_cuda_pinned_place(src.Place()),
            true,
            common::errors::InvalidArgument("Required `src` device should be "
                                            "CUDAPinnedPlace, but received %d.",
                                            src.Place()));
        PADDLE_ENFORCE_EQ(
            phi::is_gpu_place(dst.Place()),
            true,
            common::errors::InvalidArgument(
                "Required `dst` device should be CUDAPlace, but received %d.",
                dst.Place()));
        PADDLE_ENFORCE_EQ(
            phi::is_cpu_place(index.Place()),
            true,
            common::errors::InvalidArgument(
                "Required `index` device should be CPUPlace, but received %d.",
                index.Place()));
        PADDLE_ENFORCE_EQ(
            phi::is_cuda_pinned_place(buffer.Place()),
            true,
            common::errors::InvalidArgument(
                "Required `buffer` device should be CUDAPinnedPlace, "
                "but received %d.",
                buffer.Place()));
        PADDLE_ENFORCE_EQ(
            phi::is_cpu_place(offset.Place()),
            true,
            common::errors::InvalidArgument(
                "Required `offset` device should be CPUPlace, but received %d.",
                offset.Place()));
        PADDLE_ENFORCE_EQ(
            phi::is_cpu_place(count.Place()),
            true,
            common::errors::InvalidArgument(
                "Required `count` device should be CPUPlace, but received %d.",
                count.Place()));

        auto &src_tensor = src.Var().Get<phi::DenseTensor>();
        auto *dst_tensor = dst.MutableVar()->GetMutable<phi::DenseTensor>();
        auto &index_tensor = index.Var().Get<phi::DenseTensor>();
        auto *buffer_tensor =
            buffer.MutableVar()->GetMutable<phi::DenseTensor>();
        auto &offset_tensor = offset.Var().Get<phi::DenseTensor>();
        auto &count_tensor = count.Var().Get<phi::DenseTensor>();
        auto *dst_data = dst_tensor->mutable_data<float>(dst.Place());
        const auto &deviceId = paddle::platform::GetCurrentDeviceId();

        PADDLE_ENFORCE_EQ(src_tensor.dims().size(),
                          dst_tensor->dims().size(),
                          common::errors::InvalidArgument(
                              "`src` and `dst` should have same tensor shape, "
                              "except for the first dimension."));
        PADDLE_ENFORCE_EQ(
            src_tensor.dims().size(),
            buffer_tensor->dims().size(),
            common::errors::InvalidArgument(
                "`src` and `buffer` should have same tensor shape, "
                "except for the first dimension."));
        for (int i = 1; i < src_tensor.dims().size(); i++) {
          PADDLE_ENFORCE_EQ(
              src_tensor.dims()[i],
              dst_tensor->dims()[i],
              common::errors::InvalidArgument(
                  "`src` and `dst` should have the same tensor shape, "
                  "except for the first dimension."));
          PADDLE_ENFORCE_EQ(
              src_tensor.dims()[i],
              buffer_tensor->dims()[i],
              common::errors::InvalidArgument(
                  "`src` and `buffer` should have the same tensor shape, "
                  "except for the first dimension."));
        }
        PADDLE_ENFORCE_EQ(index_tensor.dims().size(),
                          1,
                          common::errors::InvalidArgument(
                              "`index` tensor should be one-dimensional."));

        auto stream =
            paddle::platform::get_current_stream(deviceId)->raw_stream();

        int64_t numel = 0;  // total copy length
        int64_t copy_flag = offset_tensor.dims()[0];
        int64_t size = src_tensor.numel() / src_tensor.dims()[0];

        if (copy_flag != 0) {
          PADDLE_ENFORCE_EQ(offset_tensor.dims().size(),
                            1,
                            common::errors::InvalidArgument(
                                "`offset` tensor should be one-dimensional."));
          PADDLE_ENFORCE_EQ(count_tensor.dims().size(),
                            1,
                            common::errors::InvalidArgument(
                                "`count` tensor should be one-dimensional."));
          PADDLE_ENFORCE_EQ(offset_tensor.numel(),
                            count_tensor.numel(),
                            common::errors::InvalidArgument(
                                "`offset` and `count` tensor size dismatch."));
          auto *offset_data = offset_tensor.data<int64_t>();
          auto *count_data = count_tensor.data<int64_t>();
          for (int64_t i = 0; i < count_tensor.numel(); i++) {
            numel += count_data[i];
          }
          PADDLE_ENFORCE_LE(numel + index_tensor.numel(),
                            buffer_tensor->dims()[0],
                            common::errors::InvalidArgument(
                                "Buffer tensor size is too small."));
          PADDLE_ENFORCE_LE(numel + index_tensor.numel(),
                            dst_tensor->dims()[0],
                            common::errors::InvalidArgument(
                                "Target tensor size is too small."));

          int64_t src_offset, dst_offset = 0, c;
          auto *src_data = src_tensor.data<float>();
          for (int64_t i = 0; i < offset_tensor.numel(); i++) {
            src_offset = offset_data[i], c = count_data[i];
            PADDLE_ENFORCE_LE(src_offset + c,
                              src_tensor.dims()[0],
                              common::errors::InvalidArgument(
                                  "Invalid offset or count index."));
            PADDLE_ENFORCE_LE(dst_offset + c,
                              dst_tensor->dims()[0],
                              common::errors::InvalidArgument(
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
                            common::errors::InvalidArgument(
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
          const int &slice_size =
              static_cast<int>(src_tensor.numel()) / src_tensor.dims()[0];
          const int &copy_bytes = static_cast<int>(slice_size) * sizeof(float);
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
                    should be one-dimensional.

  Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> from paddle.base import core
            >>> from paddle.device import cuda
            ...
            >>> if core.is_compiled_with_cuda():
            ...     src = paddle.rand(shape=[100, 50, 50], dtype="float32").pin_memory()
            ...     dst = paddle.empty(shape=[100, 50, 50], dtype="float32")
            ...     offset = paddle.to_tensor(
            ...         np.array([0, 60], dtype="int64"), place=paddle.CPUPlace())
            ...     count = paddle.to_tensor(
            ...         np.array([40, 60], dtype="int64"), place=paddle.CPUPlace())
            ...     buffer = paddle.empty(shape=[50, 50, 50], dtype="float32").pin_memory()
            ...     index = paddle.to_tensor(
            ...         np.array([1, 3, 5, 7, 9], dtype="int64")).cpu()
            ...
            ...     stream = cuda.Stream()
            ...     with cuda.stream_guard(stream):
            ...         core.eager.async_read(src, dst, index, buffer, offset, count)
)DOC");
#endif
}

}  // namespace paddle::pybind
