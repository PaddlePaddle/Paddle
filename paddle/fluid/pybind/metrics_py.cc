/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <memory>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/fleet/metrics.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/phi/common/place.h"

#include "paddle/fluid/pybind/metrics_py.h"

#if defined(PADDLE_WITH_PSLIB)
namespace paddle::pybind {
void BindMetrics(py::module* m) {
  py::class_<framework::Metric, std::shared_ptr<framework::Metric>>(*m,
                                                                    "Metric")
      .def(py::init([]() { return framework::Metric::SetInstance(); }))
      .def("init_metric",
           &framework::Metric::InitMetric,
           py::call_guard<py::gil_scoped_release>())
      .def("flip_phase",
           &framework::Metric::FlipPhase,
           py::call_guard<py::gil_scoped_release>())
      .def("get_metric_msg",
           &framework::Metric::GetMetricMsg,
           py::call_guard<py::gil_scoped_release>())
      .def("get_wuauc_metric_msg",
           &framework::Metric::GetWuAucMetricMsg,
           py::call_guard<py::gil_scoped_release>())
      .def("get_metric_name_list",
           &framework::Metric::GetMetricNameList,
           py::call_guard<py::gil_scoped_release>());
}  // end Metrics
}  // namespace paddle::pybind
#endif
