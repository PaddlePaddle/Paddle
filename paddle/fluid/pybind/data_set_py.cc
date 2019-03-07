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
#include "paddle/fluid/framework/async_executor.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/pybind/data_set_py.h"

namespace py = pybind11;
namespace pd = paddle::framework;

namespace paddle {
namespace pybind {

void BindDataset(py::module* m) {
  py::class_<framework::Dataset>(*m, "Dataset")
      .def(py::init([]() {
        return std::unique_ptr<framework::Dataset>(new framework::Dataset());
      }))
      .def("set_filelist", &framework::Dataset::SetFileList)
      .def("set_thread_num", &framework::Dataset::SetThreadNum)
      .def("set_trainer_num", &framework::Dataset::SetTrainerNum)
      .def("set_data_feed_desc", &framework::Dataset::SetDataFeedDesc)
      .def("load_into_memory", &framework::Dataset::LoadIntoMemory)
      .def("local_shuffle", &framework::Dataset::LocalShuffle)
      .def("global_shuffle", &framework::Dataset::GlobalShuffle);
}

}  // end namespace pybind
}  // end namespace paddle
