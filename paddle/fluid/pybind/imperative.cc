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
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"

#include "paddle/fluid/pybind/pybind_boost_headers.h"

namespace paddle {
namespace pybind {

void create_var_map_from_dict(const pybind11::dict& dict,
                              imperative::VarBasePtrMap* map) {
  for (auto& item : dict) {
    std::vector<imperative::VarBase*> varbase_vector;
    try {
      const std::vector<pybind11::object>& list =
          item.second.cast<std::vector<pybind11::object>>();
      varbase_vector.reserve(list.size());
      for (auto& py_var : list) {
        varbase_vector.emplace_back(
            py_var.attr("_ivar").cast<imperative::VarBase*>());
      }
    } catch (const std::runtime_error& e) {
      const pybind11::object& py_var = item.second.cast<pybind11::object>();
      varbase_vector.push_back(
          py_var.attr("_ivar").cast<imperative::VarBase*>());
    }
    map->insert(std::make_pair(item.first.cast<std::string>(),
                               std::move(varbase_vector)));
  }
}

// Bind Methods
void BindTracer(pybind11::module* m) {
  pybind11::class_<imperative::Tracer>(*m, "Tracer", "")
      .def("__init__",
           [](imperative::Tracer& self, framework::BlockDesc* root_block) {
             new (&self) imperative::Tracer(root_block);
           })
      .def("_wait", &imperative::Tracer::Wait)
      .def("trace",
           [](imperative::Tracer& self, pybind11::object py_op,
              imperative::OpBase* op, pybind11::dict inputs,
              pybind11::dict outputs, framework::AttributeMap attrs_map,
              const platform::CPUPlace expected_place,
              const bool stop_gradient = false) {
             self.HoldPyObject(py_op);
             {
               pybind11::gil_scoped_release release;
               op->trace_id_ = self.trace_id_;
               imperative::VarBasePtrMap varbase_inputs;
               create_var_map_from_dict(inputs, &varbase_inputs);
               imperative::VarBasePtrMap varbase_outputs;
               create_var_map_from_dict(outputs, &varbase_outputs);
               self.Trace(op, varbase_inputs, &varbase_outputs, attrs_map,
                          expected_place, stop_gradient);
               op->RegisterBackwardHooks([&self](imperative::OpBase* op_base) {
                 self.ReleasePyObject(op_base->trace_id_);
               });
             }
           })
      .def("trace",
           [](imperative::Tracer& self, pybind11::object py_op,
              imperative::OpBase* op, pybind11::dict inputs,
              pybind11::dict outputs, framework::AttributeMap attrs_map,
              const platform::CUDAPlace expected_place,
              const bool stop_gradient = false) {
             self.HoldPyObject(py_op);
             {
               pybind11::gil_scoped_release release;
               op->trace_id_ = self.trace_id_;
               imperative::VarBasePtrMap varbase_inputs;
               create_var_map_from_dict(inputs, &varbase_inputs);
               imperative::VarBasePtrMap varbase_outputs;
               create_var_map_from_dict(outputs, &varbase_outputs);
               self.Trace(op, varbase_inputs, &varbase_outputs, attrs_map,
                          expected_place, stop_gradient);
               op->RegisterBackwardHooks([&self](imperative::OpBase* op_base) {
                 self.ReleasePyObject(op_base->trace_id_);
               });
             }
           })
      .def("py_trace", &imperative::Tracer::PyTrace,
           pybind11::return_value_policy::take_ownership);
}

}  // namespace pybind
}  // namespace paddle
