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
#include <memory>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/async_executor.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/pybind/async_executor_py.h"

namespace py = pybind11;
namespace pd = paddle::framework;

namespace paddle {
namespace pybind {
using set_name_func = void (pd::DataFeedDesc::*)(const std::string&);
#ifdef PADDLE_WITH_PSLIB
void BindAsyncExecutor(py::module* m) {
  py::class_<framework::AsyncExecutor>(*m, "AsyncExecutor")
      .def(py::init([](framework::Scope* scope, const platform::Place& place) {
        return std::unique_ptr<framework::AsyncExecutor>(
            new framework::AsyncExecutor(scope, place));
      }))
      .def("run_from_files", &framework::AsyncExecutor::RunFromFile)
      .def("init_server", &framework::AsyncExecutor::InitServer)
      .def("init_worker", &framework::AsyncExecutor::InitWorker)
      .def("start_server", &framework::AsyncExecutor::StartServer)
      .def("stop_server", &framework::AsyncExecutor::StopServer)
      .def("gather_servers", &framework::AsyncExecutor::GatherServers)
      .def("init_model", &framework::AsyncExecutor::InitModel)
      .def("save_model", &framework::AsyncExecutor::SaveModel);
}  // end BindAsyncExecutor
#else
void BindAsyncExecutor(py::module* m) {
  py::class_<framework::AsyncExecutor>(*m, "AsyncExecutor")
      .def(py::init([](framework::Scope* scope, const platform::Place& place) {
        return std::unique_ptr<framework::AsyncExecutor>(
            new framework::AsyncExecutor(scope, place));
      }))
      .def("run_from_files", &framework::AsyncExecutor::RunFromFile);
}  // end BindAsyncExecutor
#endif
}  // end namespace pybind
}  // end namespace paddle
