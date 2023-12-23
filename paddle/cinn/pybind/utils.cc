// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/utils/error.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/cinn/utils/random_engine.h"

namespace py = pybind11;

namespace cinn {
namespace pybind {

using cinn::utils::EventType;
using cinn::utils::HostEvent;
using cinn::utils::HostEventRecorder;
using cinn::utils::ProfilerHelper;

void BindUtils(py::module *m) {
  py::enum_<EventType>(*m, "EventType")
      .value("kOrdinary", EventType::kOrdinary)
      .value("kGraph", EventType::kGraph)
      .value("kProgram", EventType::kProgram)
      .value("kFusePass", EventType::kFusePass)
      .value("kCompute", EventType::kCompute)
      .value("kSchedule", EventType::kSchedule)
      .value("kOptimize", EventType::kOptimize)
      .value("kCodeGen", EventType::kCodeGen)
      .value("kCompile", EventType::kCompile)
      .value("kInstruction", EventType::kInstruction)
      .export_values();

  py::class_<ProfilerHelper>(*m, "ProfilerHelper")
      .def_static("enable_all", &ProfilerHelper::EnableAll)
      .def_static("enable_cpu", &ProfilerHelper::EnableCPU)
      .def_static("enable_cuda", &ProfilerHelper::EnableCUDA)
      .def_static("is_enable", &ProfilerHelper::IsEnable)
      .def_static("is_enable_cpu", &ProfilerHelper::IsEnableCPU)
      .def_static("is_enable_cuda", &ProfilerHelper::IsEnableCUDA);

  py::class_<HostEventRecorder>(*m, "HostEventRecorder")
      .def_static("instance", &HostEventRecorder::GetInstance)
      .def_static("table", &HostEventRecorder::Table)
      .def("events", &HostEventRecorder::Events)
      .def("clear", &HostEventRecorder::Clear);

  py::class_<HostEvent>(*m, "HostEvent")
      .def(py::init<const std::string &, double, EventType>())
      .def_property(
          "annotation",
          [](HostEvent &self) -> const std::string & {
            return self.annotation_;
          },
          [](HostEvent &self, const std::string &v) { self.annotation_ = v; })
      .def_property(
          "duration",
          [](HostEvent &self) -> const double { return self.duration_; },
          [](HostEvent &self, double v) { self.duration_ = v; })
      .def_property(
          "type",
          [](HostEvent &self) -> const EventType & { return self.type_; },
          [](HostEvent &self, const EventType &v) { self.type_ = v; });

  py::class_<utils::LinearRandomEngine>(*m, "LinearRandomEngine");
  py::class_<utils::ErrorMessageLevel>(*m, "ErrorMessageLevel");
}

}  // namespace pybind
}  // namespace cinn
