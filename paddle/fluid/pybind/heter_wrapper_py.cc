/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/fleet/heter_wrapper.h"
#include "paddle/fluid/pybind/heter_wrapper_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
#if defined(PADDLE_WITH_PSLIB) && !defined(PADDLE_WITH_HETERPS)
void BindHeterWrapper(py::module* m) {
  py::class_<framework::HeterWrapper, std::shared_ptr<framework::HeterWrapper>>(
      *m, "Heter")
      .def(py::init([]() { return framework::HeterWrapper::GetInstance(); }))
      .def("create_client2xpu_connection",
           &framework::HeterWrapper::CreateClient2XpuConnection)
      .def("set_xpu_list", &framework::HeterWrapper::SetXpuList)
      .def("start_xpu_service", &framework::HeterWrapper::StartXpuService)
      .def("end_pass", &framework::HeterWrapper::EndPass)
      .def("stop_xpu_service", &framework::HeterWrapper::StopXpuService);
}  // end HeterWrapper
#endif
}  // end namespace pybind
}  // end namespace paddle
