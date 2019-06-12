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
#include <unordered_map>
#include <utility>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/profiler.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"

#include "paddle/fluid/pybind/pybind_boost_headers.h"

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
                      Forward,
                      inputs);  // NOLINT
  }
};

class PYBIND11_HIDDEN PyOpBase : public imperative::OpBase {
 public:
  using imperative::OpBase::OpBase;  // Inherit constructors

  PyOpBase(const std::string &name) : OpBase(name) {}
};

// Function like obj.attr_name in Python.
static PyObject *GetPythonAttribute(PyObject *obj, const char *attr_name) {
  // NOTE(zjl): PyObject_GetAttrString would return nullptr when attr_name
  // is not inside obj, but it would also set the error flag of Python.
  // If the error flag is set in C++, C++ code would not raise Exception,
  // but Python would raise Exception once C++ call ends.
  // To avoid unexpected Exception raised in Python, we check whether
  // attribute exists before calling PyObject_GetAttrString.
  //
  // Caution: PyObject_GetAttrString would increase reference count of PyObject.
  // Developer should call Py_DECREF manually after the attribute is not used.
  if (PyObject_HasAttrString(obj, attr_name)) {
    return PyObject_GetAttrString(obj, attr_name);
  } else {
    return nullptr;
  }
}

template <typename T>
static T PyObjectCast(PyObject *obj) {
  try {
    return py::cast<T>(py::handle(obj));
  } catch (py::cast_error &) {
    PADDLE_THROW("Python object is not type of %s", typeid(T).name());
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

  const char *kIVarField = "_ivar";
  PyObject *py_ivar = GetPythonAttribute(py_obj, kIVarField);
  std::vector<std::shared_ptr<imperative::VarBase>> result;

  if (py_ivar) {  // Variable
    result.emplace_back(
        PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar));
    Py_DECREF(py_ivar);
  } else if (PyList_Check(py_obj)) {  // List of Variable
    size_t len = PyList_GET_SIZE(py_obj);
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      PyObject *py_ivar =
          PyObject_GetAttrString(PyList_GET_ITEM(py_obj, i), kIVarField);
      PADDLE_ENFORCE_NOT_NULL(py_ivar);
      result.emplace_back(
          PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar));
      Py_DECREF(py_ivar);
    }
  } else if (PyTuple_Check(py_obj)) {  // Tuple of Variable
    size_t len = PyTuple_GET_SIZE(py_obj);
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      PyObject *py_ivar =
          PyObject_GetAttrString(PyTuple_GET_ITEM(py_obj, i), kIVarField);
      PADDLE_ENFORCE_NOT_NULL(py_ivar);
      result.emplace_back(
          PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar));
      Py_DECREF(py_ivar);
    }
  } else {
    PADDLE_THROW(
        "unsupported type %s, must be Variable, List[Variable] or "
        "tuple[Variable]",
        py::str(handle));
  }

  PADDLE_ENFORCE(PyErr_Occurred() == nullptr,
                 py::str(py::handle(PyErr_Occurred())));

  return result;
}

using PyVarBaseMap = std::unordered_map<std::string, py::handle>;

static imperative::VarBasePtrMap ConvertToVarBasePtrMap(
    const PyVarBaseMap &map) {
  imperative::VarBasePtrMap result;
  for (auto &pair : map) {
    auto var_vec = GetVarBaseListFromPyHandle(pair.second);
    if (!var_vec.empty()) {
      result.emplace(pair.first, std::move(var_vec));
    }
  }
  return result;
}

// Bind Methods
void BindImperative(pybind11::module *m_ptr) {
  auto &m = *m_ptr;

  py::class_<imperative::detail::BackwardStrategy> backward_strategy(
      m, "BackwardStrategy", R"DOC(

    BackwardStrategy is a descriptor of a how to run the backward process. Now it has:

    1. :code:`sort_sum_gradient`, which will sum the gradient by the reverse order of trace.

    Examples:

     .. code-block:: python
        import numpy as np
        import paddle.fluid as fluid
        from paddle.fluid import FC

        x = np.ones([2, 2], np.float32)
        with fluid.dygraph.guard():
            inputs2 = []
            for _ in range(10):
                inputs2.append(fluid.dygraph.base.to_variable(x))
            ret2 = fluid.layers.sums(inputs2)
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

  py::class_<imperative::VarBase, std::shared_ptr<imperative::VarBase>>(
      m, "VarBase", R"DOC()DOC")
      .def(
          py::init<const std::string &, paddle::framework::proto::VarType::Type,
                   const std::vector<int64_t>, const paddle::platform::CPUPlace,
                   bool, bool>())
      .def(
          py::init<const std::string &, paddle::framework::proto::VarType::Type,
                   const std::vector<int64_t>,
                   const paddle::platform::CUDAPlace, bool, bool>())
      .def("_run_backward",
           [](imperative::VarBase &self,
              const imperative::detail::BackwardStrategy &bckst) {
             self.RunBackward(bckst);
           })
      .def("_grad_name", &imperative::VarBase::GradName)
      .def("_grad_value", &imperative::VarBase::GradValue)
      .def("_clear_gradient", &imperative::VarBase::ClearGradient)
      .def("_grad_ivar",
           [](const imperative::VarBase &self) { return self.grads_; },
           py::return_value_policy::reference)
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::CPUPlace &place,
              bool blocking) {
             return self.NewVarBase(place, blocking).release();
           },
           py::return_value_policy::take_ownership)
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::CUDAPlace &place,
              bool blocking) {
             return self.NewVarBase(place, blocking).release();
           },
           py::return_value_policy::take_ownership)
      .def("value",
           [](const imperative::VarBase &self) { return self.var_.get(); },
           py::return_value_policy::reference)
      .def_property("name", &imperative::VarBase::Name,
                    &imperative::VarBase::SetName)
      .def_property_readonly("shape", &imperative::VarBase::Shape)
      .def_property_readonly("dtype", &imperative::VarBase::DataType)
      .def_property("persistable", &imperative::VarBase::IsPersistable,
                    &imperative::VarBase::SetPersistable)
      .def_property("stop_gradient", &imperative::VarBase::IsStopGradient,
                    &imperative::VarBase::SetStopGradient);

  py::class_<imperative::OpBase, PyOpBase>(m, "OpBase", R"DOC()DOC")
      .def(py::init<const std::string &>())
      .def("register_backward_hooks",
           [](imperative::OpBase &self, const py::object &callable) {
             self.RegisterBackwardHooks(callable);
           })
      .def_property("_trace_id",
                    [](const imperative::OpBase &self) {
                      py::gil_scoped_release release;
                      return self.trace_id_;
                    },
                    [](imperative::OpBase &self, int trace_id) {
                      py::gil_scoped_release release;
                      self.trace_id_ = trace_id;
                    },
                    py::return_value_policy::reference)
      .def_property_readonly("type", &imperative::OpBase::Type);

  py::class_<imperative::Layer, Layer /* <--- trampoline*/> layer(m, "Layer");
  layer.def(py::init<>())
      .def("forward",
           [](imperative::Layer &self,
              const std::vector<std::shared_ptr<imperative::VarBase>> &inputs) {
             return self.Forward(inputs);
           });

  // NOTE(zjl): Tracer use PyVarBaseMap as its parameter but not VarBasePtrMap.
  // We call Python C-API to convert PyVarBaseMap to VarBasePtrMap, instead
  // making conversion in Python code. This speed up Tracer.trace() about 6%
  // in ptb model and make time cost in Python to be nearly zero.
  py::class_<imperative::Tracer>(m, "Tracer", "")
      .def("__init__",
           [](imperative::Tracer &self, framework::BlockDesc *root_block) {
             new (&self) imperative::Tracer(root_block);
           })
      .def("trace",
           [](imperative::Tracer &self, imperative::OpBase *op,
              const PyVarBaseMap &inputs, const PyVarBaseMap &outputs,
              framework::AttributeMap attrs_map,
              const platform::CPUPlace expected_place,
              const bool stop_gradient = false) {
             auto ins = ConvertToVarBasePtrMap(inputs);
             auto outs = ConvertToVarBasePtrMap(outputs);
             {
               py::gil_scoped_release release;
               self.Trace(op, std::move(ins), &outs, attrs_map, expected_place,
                          stop_gradient);
             }
           })
      .def("trace", [](imperative::Tracer &self, imperative::OpBase *op,
                       const PyVarBaseMap &inputs, const PyVarBaseMap &outputs,
                       framework::AttributeMap attrs_map,
                       const platform::CUDAPlace expected_place,
                       const bool stop_gradient = false) {
        auto ins = ConvertToVarBasePtrMap(inputs);
        auto outs = ConvertToVarBasePtrMap(outputs);
        {
          py::gil_scoped_release release;
          self.Trace(op, std::move(ins), &outs, attrs_map, expected_place,
                     stop_gradient);
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
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
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
