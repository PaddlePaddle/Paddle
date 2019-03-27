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

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"

#include "paddle/fluid/pybind/pybind_boost_headers.h"

namespace paddle {
namespace pybind {

// Bind Methods
void BindImperative(pybind11::module* m) {
  pybind11::class_<imperative::Tracer>(*m, "Tracer", "")
      .def("__init__",
           [](imperative::Tracer& self, framework::BlockDesc* root_block) {
             new (&self) imperative::Tracer(root_block);
           })
      .def("trace",
           [](imperative::Tracer& self, imperative::OpBase* op,
              const imperative::VarBasePtrMap& inputs,
              imperative::VarBasePtrMap* outputs,
              framework::AttributeMap attrs_map,
              const platform::CPUPlace expected_place,
              const bool stop_gradient = false) {
             pybind11::gil_scoped_release release;
             return self.Trace(op, inputs, outputs, attrs_map, expected_place,
                               stop_gradient);
           })
      .def("trace",
           [](imperative::Tracer& self, imperative::OpBase* op,
              const imperative::VarBasePtrMap& inputs,
              imperative::VarBasePtrMap* outputs,
              framework::AttributeMap attrs_map,
              const platform::CUDAPlace expected_place,
              const bool stop_gradient = false) {
             pybind11::gil_scoped_release release;
             return self.Trace(op, inputs, outputs, attrs_map, expected_place,
                               stop_gradient);
           })
      .def("py_trace", &imperative::Tracer::PyTrace,
           pybind11::return_value_policy::take_ownership);

  // define parallel context
  pybind11::class_<imperative::ParallelStrategy> parallel_strategy(
      *m, "ParallelStrategy", "");
  parallel_strategy.def(pybind11::init())
      .def_property(
          "nranks",
          [](const imperative::ParallelStrategy& self) { return self.nranks_; },
          [](imperative::ParallelStrategy& self, int nranks) {
            self.nranks_ = nranks;
          })
      .def_property("local_rank",
                    [](const imperative::ParallelStrategy& self) {
                      return self.local_rank_;
                    },
                    [](imperative::ParallelStrategy& self, int local_rank) {
                      self.local_rank_ = local_rank;
                    })
      .def_property(
          "trainer_endpoints",
          [](const imperative::ParallelStrategy& self) {
            return self.trainer_endpoints_;
          },
          [](imperative::ParallelStrategy& self, std::vector<std::string> eps) {
            self.trainer_endpoints_ = eps;
          })
      .def_property("current_endpoint",
                    [](const imperative::ParallelStrategy& self) {
                      return self.current_endpoint_;
                    },
                    [](imperative::ParallelStrategy& self,
                       const std::string& ep) { self.current_endpoint_ = ep; });

  pybind11::class_<imperative::NCCLParallelContext> nccl_ctx(
      *m, "NCCLParallelContext");

  nccl_ctx
      .def(pybind11::init<const imperative::ParallelStrategy&,
                          const platform::CUDAPlace&>())
      .def("init", [](imperative::NCCLParallelContext& self) { self.Init(); });
}

}  // namespace pybind
}  // namespace paddle
