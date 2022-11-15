// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/pybind/bind_cost_model.h"

#include <pybind11/stl.h>

#include "paddle/fluid/framework/ir/cost_model.h"
#include "paddle/fluid/framework/program_desc.h"

namespace py = pybind11;
using paddle::framework::CostData;
using paddle::framework::CostModel;
using paddle::framework::ProgramDesc;

namespace paddle {
namespace pybind {

void BindCostModel(py::module* m) {
  py::class_<CostData>(*m, "CostData")
      .def(py::init<>())
      .def("get_whole_time_ms", &CostData::GetWholeTimeMs)
      .def("get_op_time_ms", &CostData::GetOpTimeMs);

  py::class_<CostModel>(*m, "CostModel")
      .def(py::init<>())
      .def("profile_measure",
           [](CostModel& self,
              py::object py_main_program,
              py::object py_startup_program,
              const std::string& device,
              const std::vector<std::string>& fetch_cost_list) {
             py::object py_main_program_desc = py_main_program.attr("desc");
             ProgramDesc* main_program_desc =
                 py_main_program_desc.cast<ProgramDesc*>();

             py::object py_startup_program_desc =
                 py_startup_program.attr("desc");
             ProgramDesc* startup_program_desc =
                 py_startup_program_desc.cast<ProgramDesc*>();
             return self.ProfileMeasure(*main_program_desc,
                                        *startup_program_desc,
                                        device,
                                        fetch_cost_list);
           });
}

}  // namespace pybind
}  // namespace paddle
