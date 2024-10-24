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
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/fleet/nccl_wrapper.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/framework/data_feed.pb.h"

#include "paddle/fluid/pybind/nccl_wrapper_py.h"

namespace py = pybind11;

namespace paddle::pybind {
void BindNCCLWrapper(py::module* m) {
  py::class_<framework::NCCLWrapper>(*m, "Nccl")
      .def(py::init())
      .def("init_nccl", &framework::NCCLWrapper::InitNCCL)
      .def("set_nccl_id", &framework::NCCLWrapper::SetNCCLId)
      .def("set_rank_info", &framework::NCCLWrapper::SetRankInfo)
      .def("sync_var", &framework::NCCLWrapper::SyncVar);
}  // end NCCLWrapper
}  // namespace paddle::pybind
