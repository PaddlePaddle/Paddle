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

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/pybind/box_helper_py.h"
#include "paddle/phi/core/framework/data_feed.pb.h"
#ifdef PADDLE_WITH_BOX_PS
#include <boxps_public.h>
#endif

namespace py = pybind11;

namespace paddle::pybind {
void BindBoxHelper(py::module* m) {
  py::class_<framework::BoxHelper, std::shared_ptr<framework::BoxHelper>>(
      *m, "BoxPS")
      .def(py::init([](paddle::framework::Dataset* dataset) {
        return std::make_shared<paddle::framework::BoxHelper>(dataset);
      }))
      .def("set_date",
           &framework::BoxHelper::SetDate,
           py::call_guard<py::gil_scoped_release>())
      .def("begin_pass",
           &framework::BoxHelper::BeginPass,
           py::call_guard<py::gil_scoped_release>())
      .def("end_pass",
           &framework::BoxHelper::EndPass,
           py::call_guard<py::gil_scoped_release>())
      .def("wait_feed_pass_done",
           &framework::BoxHelper::WaitFeedPassDone,
           py::call_guard<py::gil_scoped_release>())
      .def("preload_into_memory",
           &framework::BoxHelper::PreLoadIntoMemory,
           py::call_guard<py::gil_scoped_release>())
      .def("load_into_memory",
           &framework::BoxHelper::LoadIntoMemory,
           py::call_guard<py::gil_scoped_release>())
      .def("slots_shuffle",
           &framework::BoxHelper::SlotsShuffle,
           py::call_guard<py::gil_scoped_release>());
}  // end BoxHelper

#ifdef PADDLE_WITH_BOX_PS
void BindBoxWrapper(py::module* m) {
  py::class_<framework::BoxWrapper, std::shared_ptr<framework::BoxWrapper>>(
      *m, "BoxWrapper")
      .def(py::init([](int embedx_dim, int expand_embed_dim) {
        // return std::make_shared<paddle::framework::BoxHelper>(dataset);
        return framework::BoxWrapper::SetInstance(embedx_dim, expand_embed_dim);
      }))
      .def("save_base",
           &framework::BoxWrapper::SaveBase,
           py::call_guard<py::gil_scoped_release>())
      .def("feed_pass",
           &framework::BoxWrapper::FeedPass,
           py::call_guard<py::gil_scoped_release>())
      .def("set_test_mode",
           &framework::BoxWrapper::SetTestMode,
           py::call_guard<py::gil_scoped_release>())
      .def("save_delta",
           &framework::BoxWrapper::SaveDelta,
           py::call_guard<py::gil_scoped_release>())
      .def("initialize_gpu_and_load_model",
           &framework::BoxWrapper::InitializeGPUAndLoadModel,
           py::call_guard<py::gil_scoped_release>())
      .def("initialize_auc_runner",
           &framework::BoxWrapper::InitializeAucRunner,
           py::call_guard<py::gil_scoped_release>())
      .def("init_metric",
           &framework::BoxWrapper::InitMetric,
           py::call_guard<py::gil_scoped_release>())
      .def("get_metric_msg",
           &framework::BoxWrapper::GetMetricMsg,
           py::call_guard<py::gil_scoped_release>())
      .def("get_metric_name_list",
           &framework::BoxWrapper::GetMetricNameList,
           py::call_guard<py::gil_scoped_release>())
      .def("flip_phase",
           &framework::BoxWrapper::FlipPhase,
           py::call_guard<py::gil_scoped_release>())
      .def("init_afs_api",
           &framework::BoxWrapper::InitAfsAPI,
           py::call_guard<py::gil_scoped_release>())
      .def("finalize",
           &framework::BoxWrapper::Finalize,
           py::call_guard<py::gil_scoped_release>());
}  // end BoxWrapper
#endif

}  // namespace paddle::pybind
