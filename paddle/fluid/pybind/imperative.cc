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
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/backward_strategy.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/data_loader.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/nccl_context.h"
#include "paddle/fluid/imperative/partial_grad_engine.h"
#include "paddle/fluid/imperative/profiler.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/pybind/op_function.h"
#include "paddle/fluid/pybind/pybind_boost_headers.h"
#include "paddle/fluid/pybind/tensor_py.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

class Layer : public imperative::Layer {
 public:
  using imperative::Layer::Layer;  // Inherit constructors

  std::vector<std::shared_ptr<imperative::VarBase>> Forward(
      const std::vector<std::shared_ptr<imperative::VarBase>> &inputs)
      override {
    PYBIND11_OVERLOAD(std::vector<std::shared_ptr<imperative::VarBase>>, Layer,
                      Forward, inputs);  // NOLINT
  }
};

static const platform::Place PyObjectToPlace(const py::object &place_obj) {
  if (py::isinstance<platform::CPUPlace>(place_obj)) {
    return place_obj.cast<platform::CPUPlace>();
  } else if (py::isinstance<platform::CUDAPlace>(place_obj)) {
    return place_obj.cast<platform::CUDAPlace>();
  } else if (py::isinstance<platform::CUDAPinnedPlace>(place_obj)) {
    return place_obj.cast<platform::CUDAPinnedPlace>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place should be one of CPUPlace/CUDAPlace/CUDAPinnedPlace"));
  }
}

static void InitTensorForVarBase(imperative::VarBase *self,
                                 const py::array &array,
                                 const platform::Place place,
                                 bool persistable = false,
                                 bool zero_copy = false,
                                 std::string name = "") {
  if (name == "") {
    name = imperative::GetCurrentTracer()->GenerateUniqueName("generated_var");
  }
  new (self) imperative::VarBase(name);
  auto *tensor = self->MutableVar()->GetMutable<framework::LoDTensor>();
  if (platform::is_cpu_place(place)) {
    SetTensorFromPyArray<platform::CPUPlace>(
        tensor, array, BOOST_GET_CONST(platform::CPUPlace, place), zero_copy);
  } else if (platform::is_gpu_place(place)) {
    SetTensorFromPyArray<platform::CUDAPlace>(
        tensor, array, BOOST_GET_CONST(platform::CUDAPlace, place), zero_copy);
  } else if (platform::is_cuda_pinned_place(place)) {
    SetTensorFromPyArray<platform::CUDAPinnedPlace>(
        tensor, array, BOOST_GET_CONST(platform::CUDAPinnedPlace, place),
        zero_copy);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place should be one of CPUPlace/CUDAPlace/CUDAPinnedPlace"));
  }
  self->SetPersistable(persistable);
  self->SetType(framework::proto::VarType::LOD_TENSOR);
  self->SetDataType(tensor->type());
}

static void InitVarBaseFromNumpyWithKwargs(imperative::VarBase *self,
                                           const py::kwargs &kwargs) {
  PADDLE_ENFORCE_EQ(
      kwargs.contains("value"), true,
      platform::errors::NotFound(
          "The kwargs used to create Varbase misses argument: value"));

  auto persistable = kwargs.contains("persistable")
                         ? kwargs["persistable"].cast<bool>()
                         : false;
  auto array = kwargs.contains("value") ? kwargs["value"].cast<py::array>()
                                        : py::array();
  auto zero_copy =
      kwargs.contains("zero_copy") ? kwargs["zero_copy"].cast<bool>() : false;
  auto name = kwargs.contains("name") ? kwargs["name"].cast<std::string>() : "";
  auto default_place = imperative::GetCurrentTracer()->ExpectedPlace();
  auto place = kwargs.contains("place") ? PyObjectToPlace(kwargs["place"])
                                        : default_place;
  InitTensorForVarBase(self, array, place, persistable, zero_copy, name);
}

template <typename P>
static void InitVarBaseFromNumpyWithArg(imperative::VarBase *self,
                                        const py::array &array, const P &place,
                                        bool persistable = false,
                                        bool zero_copy = false,
                                        std::string name = "") {
  // 0: self, 1: value, 2: place, 3: persistable, 4: zero_copy, 5: name
  if (name == "") {
    name = imperative::GetCurrentTracer()->GenerateUniqueName("generated_var");
  }
  new (self) imperative::VarBase(name);
  self->SetPersistable(persistable);
  auto *tensor = self->MutableVar()->GetMutable<framework::LoDTensor>();
  SetTensorFromPyArray<P>(tensor, array, place, zero_copy);
  self->SetType(framework::proto::VarType::LOD_TENSOR);
  self->SetDataType(tensor->type());
}

static void InitVarBaseFromNumpyWithArgDefault(imperative::VarBase *self,
                                               const py::array &array) {
  auto place = imperative::GetCurrentTracer()->ExpectedPlace();
  InitTensorForVarBase(self, array, place);
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

using PyNameVarBaseMap = std::unordered_map<std::string, py::handle>;

template <typename T>
static T PyObjectCast(PyObject *obj) {
  try {
    return py::cast<T>(py::handle(obj));
  } catch (py::cast_error &) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Python object is not type of %s", typeid(T).name()));
  }
}

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
      PyErr_Occurred(), nullptr,
      platform::errors::InvalidArgument(py::str(py::handle(PyErr_Occurred()))));
  return result;
}

static void ParseIndexingSlice(framework::LoDTensor *tensor, PyObject *_index,
                               std::vector<int> *slice_axes,
                               std::vector<int> *slice_starts,
                               std::vector<int> *slice_ends,
                               std::vector<int> *slice_strides,
                               std::vector<int> *decrease_axis,
                               std::vector<int> *infer_flags) {
  // We allow indexing by Integers, Slices, and tuples of those
  // types.
  // Ellipsis and None are not supported yet.
  // wrap to tuple
  PyObject *index = !PyTuple_Check(_index) ? PyTuple_Pack(1, _index) : _index;
  PADDLE_ENFORCE_EQ(
      tensor->IsInitialized(), true,
      platform::errors::InvalidArgument("tensor has not been initialized"));
  const auto &shape = tensor->dims();
  const int rank = shape.size();
  const int size = PyTuple_GET_SIZE(index);
  PADDLE_ENFORCE_EQ(
      size <= rank, true,
      platform::errors::InvalidArgument(
          "too many indices (%d) for tensor of dimension %d", size, rank));
  for (int dim = 0; dim < size; ++dim) {
    PyObject *slice_item = PyTuple_GetItem(index, dim);
    PADDLE_ENFORCE_EQ(
        PyNumber_Check(slice_item) || PySlice_Check(slice_item), true,
        platform::errors::InvalidArgument(
            "We allow indexing by Integers, Slices, and tuples of "
            "these types, but received %s in %dth slice item",
            std::string(Py_TYPE(slice_item)->tp_name), dim + 1));
    infer_flags->push_back(1);
    int dim_len = shape[dim];
    if (PyNumber_Check(slice_item)) {
      // integer
      int start = static_cast<int>(PyLong_AsLong(slice_item));
      auto s_t = start;
      start = start < 0 ? start + dim_len : start;
      if (start >= dim_len) {
        std::string str_error_message =
            "The starting index " + std::to_string(s_t) +
            " of slice is out of bounds in tensor " + std::to_string(dim) +
            "-th axis, it shound be in the range of [" +
            std::to_string(-dim_len) + ", " + std::to_string(dim_len) + ")";
        // py::index_error is corresponding to IndexError in Python
        // Used to indicate out of bounds access in __getitem__, __setitem__
        throw py::index_error(str_error_message);
      }
      slice_axes->push_back(dim);
      slice_starts->push_back(start);
      slice_ends->push_back(start + 1);
      slice_strides->push_back(1);
      decrease_axis->push_back(dim);
    } else {
      // slice
      Py_ssize_t start, end, step;
// The parameter type for the slice parameter was PySliceObject* before 3.2
#if PY_VERSION_HEX >= 0x03020000
      PySlice_GetIndices(slice_item, dim_len, &start, &end, &step);
#else
      PySlice_GetIndices(reinterpret_cast<PySliceObject *>(slice_item), dim_len,
                         &start, &end, &step);
#endif
      // :: or : or 0:dim_len:1
      if (start == 0 && end == dim_len && step == 1) continue;
      slice_axes->push_back(dim);
      slice_starts->push_back(start);
      slice_ends->push_back(end);
      slice_strides->push_back(step);
    }
  }
  if (!PyTuple_Check(_index)) Py_DecRef(index);
}

// Bind Methods
void BindImperative(py::module *m_ptr) {
  auto &m = *m_ptr;

  BindOpFunctions(&m);

#ifndef _WIN32
  // Dygraph DataLoader signal handler
  m.def("_set_process_pids", [](int64_t key, py::object &obj) {
    PADDLE_ENFORCE_EQ(
        py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj), true,
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
              string::Sprintf("%s", array.dtype()).compare("object"), 0,
              platform::errors::InvalidArgument(
                  "Faild to convert input data to a regular ndarray.\n  * "
                  "Usually this means the input data contains nested "
                  "lists with different lengths.\n  * Check the reader "
                  "function passed to 'set_(sample/sample_list/batch)"
                  "_generator' to locate the data causes this issue."));
          // 2. construcct LoDTensor
          framework::LoDTensor t;
          SetTensorFromPyArray<platform::CPUPlace>(&t, array,
                                                   platform::CPUPlace(), true);
          // 3. allocate shared memory
          void *data_ptr = t.data<void>();
          size_t data_size = t.numel() * framework::SizeOfType(t.type());
          auto shared_writer_holder =
              memory::allocation::AllocateMemoryMapWriterAllocation(data_size);
          // 4. maintain mmap fd set & backup ipc_name
          const std::string &ipc_name = shared_writer_holder->ipc_name();
          memory::allocation::MemoryMapFdSet::Instance().Insert(ipc_name);
          // 5. copy data & reset holder
          memory::Copy(platform::CPUPlace(), shared_writer_holder->ptr(),
                       platform::CPUPlace(), data_ptr, data_size);
          t.ResetHolder(shared_writer_holder);
          // 6. append to result list
          tensors.append(t);
        }
        return tensors;
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

  py::class_<imperative::detail::BackwardStrategy> backward_strategy(
      m, "BackwardStrategy", R"DOC(

    BackwardStrategy is a descriptor of how to run the backward process.

    **Note**:
        **This API is only available in** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **Mode**

    Attribute:
        **sort_sum_gradient**:

        If framework will sum the gradient by the reverse order of trace. eg. x_var ( :ref:`api_guide_Variable` ) will be the input of multiple OP such as :ref:`api_fluid_layers_scale` , this attr will decide if framework will sum gradient of `x_var` by the reverse order.

        By Default: False

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle.fluid as fluid

                x = np.ones([2, 2], np.float32)
                with fluid.dygraph.guard():
                    x_var = fluid.dygraph.to_variable(x)
                    sums_inputs = []
                    # x_var will be multi-scales' input here
                    for _ in range(10):
                        sums_inputs.append(fluid.layers.scale(x_var))
                    ret2 = fluid.layers.sums(sums_inputs)
                    loss2 = fluid.layers.reduce_sum(ret2)
                    backward_strategy = fluid.dygraph.BackwardStrategy()
                    backward_strategy.sort_sum_gradient = True
                    loss2.backward(backward_strategy)
      )DOC");
  backward_strategy.def(py::init())
      .def_property("sort_sum_gradient",
                    [](const imperative::detail::BackwardStrategy &self) {
                      return self.sorted_sum_gradient_;
                    },
                    [](imperative::detail::BackwardStrategy &self,
                       bool sorted_sum_gradient) {
                      self.sorted_sum_gradient_ = sorted_sum_gradient;
                    });

  m.def("start_imperative_gperf_profiler",
        []() { imperative::StartProfile(); });

  m.def("stop_imperative_gperf_profiler", []() { imperative::StopProfile(); });

  m.def("_is_dygraph_debug_enabled",
        []() { return imperative::IsDebugEnabled(); });
  m.def("_dygraph_debug_level", []() { return imperative::GetDebugLevel(); });
  m.def("_switch_tracer",
        [](const std::shared_ptr<imperative::Tracer> &tracer) {
          imperative::SetCurrentTracer(tracer);
        });

  py::class_<imperative::VarBase, std::shared_ptr<imperative::VarBase>>(
      m, "VarBase",
      R"DOC()DOC")
      .def_static("_alive_vars", &imperative::VarBase::AliveVarNames)
      .def("__init__",
           [](imperative::VarBase &self, framework::proto::VarType::Type dtype,
              const std::vector<int> &dims, const py::handle &name,
              framework::proto::VarType::Type type, bool persistable) {
             std::string act_name = "";
             if (!name.ptr() || name.ptr() == Py_None) {
               act_name = imperative::GetCurrentTracer()->GenerateUniqueName(
                   "generated_var");
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
               tensor->Resize(framework::make_ddim(dims));
             }
           })
      .def("__init__", &InitVarBaseFromNumpyWithArg<platform::CPUPlace>,
           py::arg("value"), py::arg("place"), py::arg("persistable") = false,
           py::arg("zero_copy") = false, py::arg("name") = "")
      .def("__init__", &InitVarBaseFromNumpyWithArg<platform::CUDAPlace>,
           py::arg("value"), py::arg("place"), py::arg("persistable") = false,
           py::arg("zero_copy") = false, py::arg("name") = "")
      .def("__init__", &InitVarBaseFromNumpyWithArg<platform::CUDAPinnedPlace>,
           py::arg("value"), py::arg("place"), py::arg("persistable") = false,
           py::arg("zero_copy") = false, py::arg("name") = "")
      .def("__init__", &InitVarBaseFromNumpyWithArgDefault, py::arg("value"))
      .def("__init__", &InitVarBaseFromNumpyWithKwargs)
      .def("__getitem__",
           [](std::shared_ptr<imperative::VarBase> &self, py::handle _index) {
             std::vector<int> slice_axes, slice_starts, slice_ends,
                 slice_strides, decrease_axis, infer_flags;
             auto tensor =
                 self->MutableVar()->GetMutable<framework::LoDTensor>();
             ParseIndexingSlice(tensor, _index.ptr(), &slice_axes,
                                &slice_starts, &slice_ends, &slice_strides,
                                &decrease_axis, &infer_flags);

             // release gil and do tracing
             py::gil_scoped_release release;
             const auto &tracer = imperative::GetCurrentTracer();
             if (slice_axes.empty()) {
               return self;
             } else {
               imperative::NameVarBaseMap ins = {{"Input", {self}}};
               framework::AttributeMap attrs = {
                   {"axes", slice_axes},
                   {"starts", slice_starts},
                   {"ends", slice_ends},
                   {"infer_flags", infer_flags},
                   {"decrease_axis", decrease_axis}};
               auto out = std::shared_ptr<imperative::VarBase>(
                   new imperative::VarBase(tracer->GenerateUniqueName()));
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
               return out;
             }
           })
      .def("numpy",
           [](imperative::VarBase &self) -> py::array {
             const auto &tensor =
                 self.MutableVar()->Get<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(
                 tensor.IsInitialized(), true,
                 platform::errors::InvalidArgument(
                     "Tensor of %s is Empty, please check if it has no data.",
                     self.Name()));
             return TensorToPyArray(tensor, true);
           },
           R"DOC(
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Returns a numpy array shows the value of current :ref:`api_guide_Variable_en`

        Returns:
            ndarray: The numpy value of current Variable.

        Returns type:
            ndarray: dtype is same as current Variable

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import Linear
                import numpy as np

                data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
                with fluid.dygraph.guard():
                    linear = Linear(32, 64)
                    data = to_variable(data)
                    x = linear(data)
                    print(x.numpy())

       )DOC")
      .def("detach",
           [](const imperative::VarBase &self) {
             const auto &tensor = self.Var().Get<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(tensor.IsInitialized(), true,
                               platform::errors::InvalidArgument(
                                   "%s has not been initialized", self.Name()));
             return self.NewVarBase(tensor.place(), false);
           },
           py::return_value_policy::copy, R"DOC(
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Returns a new Variable, detached from the current graph.

        Returns:
             ( :ref:`api_guide_Variable_en` | dtype is same as current Variable): The detached Variable.


        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import Linear
                import numpy as np

                data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
                with fluid.dygraph.guard():
                    linear = Linear(32, 64)
                    data = to_variable(data)
                    x = linear(data)
                    y = x.detach()

       )DOC")
      .def("clear_gradient", &imperative::VarBase::ClearGradient, R"DOC(

        **Notes**:
        **1. This API is ONLY available in Dygraph mode**

        **2. Use it only Variable has gradient, normally we use this for Parameters since other temporal Variable will be deleted by Python's GC**

        Clear  (set to ``0`` ) the Gradient of Current Variable

        Returns:  None

        Examples:
             .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                x = np.ones([2, 2], np.float32)
                with fluid.dygraph.guard():
                    inputs2 = []
                    for _ in range(10):
                         tmp = fluid.dygraph.base.to_variable(x)
                         tmp.stop_gradient=False
                         inputs2.append(tmp)
                    ret2 = fluid.layers.sums(inputs2)
                    loss2 = fluid.layers.reduce_sum(ret2)
                    backward_strategy = fluid.dygraph.BackwardStrategy()
                    backward_strategy.sort_sum_gradient = True
                    loss2.backward(backward_strategy)
                    print(loss2.gradient())
                    loss2.clear_gradient()
                    print("After clear {}".format(loss2.gradient()))
      )DOC")
      .def("_run_backward",
           [](imperative::VarBase &self,
              const imperative::detail::BackwardStrategy &bckst,
              const imperative::Tracer &tracer) {
             // TODO(jiabin): when we impl more backward execution we can select
             // them
             auto *engine = tracer.GetEngine();
             engine->Init(&self, bckst);
             VLOG(3) << "Start backward";
             engine->Execute();
             VLOG(3) << "Finish backward";
           },
           py::call_guard<py::gil_scoped_release>())
      .def("_grad_name", &imperative::VarBase::GradVarName)
      .def("_grad_value",
           [](imperative::VarBase &self) {
             return self.MutableGradVar()->Get<framework::LoDTensor>();
           },
           py::return_value_policy::reference)
      .def("_set_grad_type",
           [](imperative::VarBase &self, framework::proto::VarType::Type type) {
             self.MutableGradVarBase()->SetType(type);
           })
      .def("_grad_ivar",
           [](const imperative::VarBase &self) {
             auto &grad_var = self.GradVarBase();
             if (grad_var && grad_var->Var().IsInitialized()) {
               auto *tensor =
                   grad_var->MutableVar()->IsType<framework::LoDTensor>()
                       ? grad_var->MutableVar()
                             ->GetMutable<framework::LoDTensor>()
                       : grad_var->MutableVar()
                             ->GetMutable<framework::SelectedRows>()
                             ->mutable_value();
               if (tensor->IsInitialized()) {
                 return grad_var;
               }
             }
             return std::shared_ptr<imperative::VarBase>(nullptr);
           },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::CPUPlace &place,
              bool blocking) { return self.NewVarBase(place, blocking); },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::CUDAPlace &place,
              bool blocking) { return self.NewVarBase(place, blocking); },
           py::return_value_policy::copy)
      .def("value", [](imperative::VarBase &self) { return self.MutableVar(); },
           py::return_value_policy::reference)
      .def_property("name", &imperative::VarBase::Name,
                    &imperative::VarBase::SetName)
      .def_property("stop_gradient",
                    &imperative::VarBase::OverridedStopGradient,
                    &imperative::VarBase::SetOverridedStopGradient)
      .def_property("persistable", &imperative::VarBase::Persistable,
                    &imperative::VarBase::SetPersistable)
      .def_property_readonly(
          "shape",
          [](imperative::VarBase &self) {
            if (self.Var().IsType<framework::LoDTensor>()) {
              return framework::vectorize<int>(
                  self.Var().Get<framework::LoDTensor>().dims());
            } else if (self.Var().IsType<framework::SelectedRows>()) {
              return framework::vectorize<int>(
                  self.Var().Get<framework::SelectedRows>().value().dims());
            } else {
              VLOG(2) << "It is meaningless to get shape of variable type "
                      << GetTypeName(self);
              return std::vector<int>();
            }
          })
      .def_property_readonly("type", &imperative::VarBase::Type)
      .def_property_readonly("dtype", &imperative::VarBase::DataType);

  py::class_<imperative::Layer, Layer /* <--- trampoline*/> layer(m, "Layer");
  layer.def(py::init<>())
      .def("forward",
           [](imperative::Layer &self,
              const std::vector<std::shared_ptr<imperative::VarBase>> &inputs) {
             return self.Forward(inputs);
           });

  py::class_<imperative::jit::ProgramDescTracer>(m, "ProgramDescTracer", "")
      .def("create_program_desc",
           &imperative::jit::ProgramDescTracer::CreateProgramDesc)
      .def("reset", &imperative::jit::ProgramDescTracer::Reset);

  py::class_<imperative::Tracer, std::shared_ptr<imperative::Tracer>>(
      m, "Tracer",
      R"DOC()DOC")
      .def("__init__",
           [](imperative::Tracer &self) { new (&self) imperative::Tracer(); })
      .def_property("_enable_program_desc_tracing",
                    &imperative::Tracer::IsProgramDescTracingEnabled,
                    &imperative::Tracer::SetEnableProgramDescTracing)
      .def_property("_train_mode", &imperative::Tracer::HasGrad,
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
            } else if (py::isinstance<platform::CPUPlace>(obj)) {
              auto p = obj.cast<platform::CPUPlace *>();
              self.SetExpectedPlace(*p);
            } else if (py::isinstance<platform::CUDAPinnedPlace>(obj)) {
              auto p = obj.cast<platform::CUDAPinnedPlace *>();
              self.SetExpectedPlace(*p);
            } else {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "Incompatible Place Type: supports CUDAPlace, CPUPlace, "
                  "and CUDAPinnedPlace, "
                  "but got Unknown Type!"));
            }
          })
      .def("_get_program_desc_tracer",
           &imperative::Tracer::GetProgramDescTracer,
           py::return_value_policy::reference)
      .def("_generate_unique_name", &imperative::Tracer::GenerateUniqueName,
           py::arg("key") = "tmp")
      .def("trace",
           [](imperative::Tracer &self, const std::string &type,
              const PyNameVarBaseMap &ins, const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs, const platform::CUDAPlace &place,
              bool trace_backward) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(ins_map), std::move(outs_map),
                            std::move(attrs), place, trace_backward);
             }
           })
      .def("trace",
           [](imperative::Tracer &self, const std::string &type,
              const PyNameVarBaseMap &ins, const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs, const platform::CPUPlace &place,
              bool trace_backward) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(ins_map), std::move(outs_map),
                            std::move(attrs), place, trace_backward);
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
      .def_property("local_rank",
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
      .def_property("current_endpoint",
                    [](const imperative::ParallelStrategy &self) {
                      return self.current_endpoint_;
                    },
                    [](imperative::ParallelStrategy &self,
                       const std::string &ep) { self.current_endpoint_ = ep; });

  m.def(
      "dygraph_partial_grad",
      [](const std::vector<std::shared_ptr<imperative::VarBase>> &input_targets,
         const std::vector<std::shared_ptr<imperative::VarBase>>
             &output_targets,
         const std::vector<std::shared_ptr<imperative::VarBase>> &output_grads,
         const std::vector<std::shared_ptr<imperative::VarBase>> &no_grad_vars,
         const platform::Place &place,
         const imperative::detail::BackwardStrategy &strategy,
         bool create_graph, bool retain_graph, bool allow_unused,
         bool only_inputs) {
        imperative::PartialGradEngine engine(
            input_targets, output_targets, output_grads, no_grad_vars, place,
            strategy, create_graph, retain_graph, allow_unused, only_inputs);
        engine.Execute();
        return engine.GetResult();
      },
      py::call_guard<py::gil_scoped_release>());

#if defined(PADDLE_WITH_NCCL)
  py::class_<imperative::NCCLParallelContext> nccl_ctx(m,
                                                       "NCCLParallelContext");

  nccl_ctx
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::CUDAPlace &>())
      .def("init", [](imperative::NCCLParallelContext &self) { self.Init(); });
#endif
}

}  // namespace pybind
}  // namespace paddle
