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
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/pybind/box_helper_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindBoxHelper(py::module* m) {
  py::class_<framework::BoxHelper, std::shared_ptr<framework::BoxHelper>>(
      *m, "BoxPS")
      .def(py::init([](paddle::framework::Dataset* dataset) {
        return std::make_shared<paddle::framework::BoxHelper>(dataset);
      }))
      .def("begin_pass", &framework::BoxHelper::BeginPass)
      .def("end_pass", &framework::BoxHelper::EndPass)
      .def("wait_feed_pass_done", &framework::BoxHelper::WaitFeedPassDone)
      .def("preload_into_memory", &framework::BoxHelper::PreLoadIntoMemory)
      .def("load_into_memory", &framework::BoxHelper::LoadIntoMemory);
}  // end BoxHelper
}  // end namespace pybind
}  // end namespace paddle
