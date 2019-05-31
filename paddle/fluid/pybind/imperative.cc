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

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <memory>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/profiler.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"

#include "paddle/fluid/pybind/pybind_boost_headers.h"

namespace paddle {
namespace pybind {

class Layer : public imperative::Layer {
 public:
  using imperative::Layer::Layer;  // Inherit constructors

  std::vector<imperative::VarBase *> Forward(
      const std::vector<imperative::VarBase *> &inputs) override {
    PYBIND11_OVERLOAD(std::vector<imperative::VarBase *>, Layer, Forward,
                      inputs);  // NOLINT
  }
};

class PYBIND11_HIDDEN PyOpBase : public imperative::OpBase {
 public:
  using imperative::OpBase::OpBase;  // Inherit constructors

  PyOpBase(const std::string &name) : OpBase(name) {}
};

// Bind Methods
void BindImperative(pybind11::module *m_ptr) {
  namespace py = ::pybind11;

  auto &m = *m_ptr;

  py::class_<imperative::detail::BackwardStrategy> backward_strategy(
      m, "BackwardStrategy", R"DOC()DOC");
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

  py::class_<imperative::VarBase>(m, "VarBase", R"DOC()DOC")
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
      .def("forward", [](imperative::Layer &self,
                         const std::vector<imperative::VarBase *> &inputs) {
        return self.Forward(inputs);
      });

  py::class_<imperative::Tracer>(*m, "Tracer", "")
      .def("__init__",
           [](imperative::Tracer &self, framework::BlockDesc *root_block) {
             new (&self) imperative::Tracer(root_block);
           })
      .def("trace",
           [](imperative::Tracer &self, imperative::OpBase *op,
              const imperative::VarBasePtrMap &inputs,
              imperative::VarBasePtrMap *outputs,
              framework::AttributeMap attrs_map,
              const platform::CPUPlace expected_place,
              const bool stop_gradient = false) {
             py::gil_scoped_release release;
             return self.Trace(op, inputs, outputs, attrs_map, expected_place,
                               stop_gradient);
           })
      .def("trace", [](imperative::Tracer &self, imperative::OpBase *op,
                       const imperative::VarBasePtrMap &inputs,
                       imperative::VarBasePtrMap *outputs,
                       framework::AttributeMap attrs_map,
                       const platform::CUDAPlace expected_place,
                       const bool stop_gradient = false) {
        py::gil_scoped_release release;
        return self.Trace(op, inputs, outputs, attrs_map, expected_place,
                          stop_gradient);
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
