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

#include <string>
#include <vector>

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/gloo_wrapper_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindGlooWrapper(py::module* m) {
  py::class_<framework::GlooWrapper>(*m, "Gloo")
      .def(py::init())
      .def("init", &framework::GlooWrapper::Init)
      .def("rank", &framework::GlooWrapper::Rank)
      .def("size", &framework::GlooWrapper::Size)
      .def("barrier", &framework::GlooWrapper::Barrier)
      .def("all_reduce", &framework::GlooWrapper::AllReduce<uint64_t>)
      .def("all_reduce", &framework::GlooWrapper::AllReduce<int64_t>)
      .def("all_reduce", &framework::GlooWrapper::AllReduce<double>)
      .def("all_gather", &framework::GlooWrapper::AllGather<uint64_t>)
      .def("all_gather", &framework::GlooWrapper::AllGather<int64_t>)
      .def("all_gather", &framework::GlooWrapper::AllGather<double>);
}  // end BindGlooWrapper
}  // end namespace pybind
}  // end namespace paddle
