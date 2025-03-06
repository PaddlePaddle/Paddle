// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "paddle/cinn/ir/group_schedule/search/measurer.h"
#include "paddle/common/enforce.h"


namespace cinn::pybind {

namespace py = pybind11;
using namespace cinn::ir::search;  // NOLINT

void BindSearch(pybind11::module *m) {
  py::class_<MeasureResult>(*m, "MeasureResult")
      .def(py::init())
      .def_property_readonly("compile_time", [](const MeasureResult& self) { return self.compile_time.count(); })
      .def_property_readonly("avg_kernel_execute_time", [](const MeasureResult& self) { return self.avg_kernel_execute_time.count(); })
      .def_property_readonly("avg_total_execute_time", [](const MeasureResult& self) { return self.avg_total_execute_time.count(); })
      .def_readonly("err_msg", &MeasureResult::err_msg);

  py::class_<Measurer, std::shared_ptr<Measurer>>(*m, "Measurer")
      .def(py::init(
        [](std::shared_ptr<::pir::Program> program){
            LOG(INFO) << "[Pybind] Pass-in Program when binding `measurer`: \n" << *program;
            return std::make_shared<Measurer>(program.get());
        }))
      .def("compile", &Measurer::Compile)
      .def("run", &Measurer::Run
           /*[](Measurer &self,
              const std::unordered_map<std::string, std::vector<int64_t>> &input_name_and_shape,
              int repeat) {
              LOG(INFO) << __PRETTY_FUNCTION__ << " | "<< "Pybind before run";  
              self.Run(input_name_and_shape, repeat);
              LOG(INFO) << __PRETTY_FUNCTION__ << " | "<< "Pybind after run"; 
           }*/)
      .def("result", &Measurer::Result);
}
}  // namespace cinn::pybind
