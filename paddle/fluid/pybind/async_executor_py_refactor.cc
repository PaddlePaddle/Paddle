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

// To avoid conflicting definition in gcc-4.8.2 headers and pyconfig.h (2.7.3)
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <string>
#include <vector>
#include "paddle/fluid/framework/async_executor_param.pb.h"
#include "paddle/fluid/framework/async_executor.h"
#include "paddle/fluid/pybind/async_executor_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindAsyncExecutor(py::module* m) {
  py::class_<framework::AsyncExecutor>(*m, "AsyncExecutor")
      .def("run_from_files", &framework::AsyncExecutor::RunFromFiles)
      .def("check_files", &framework::AsyncExecutor::CheckFiles);
}   // end BindAsyncExecutor
}   // end namespace pybind
}   // end namespace paddle

/* vim: set expandtab ts=2 sw=2 sts=2 tw=80: */
