/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/imperative/distributed/ProcessGroup.h"
#include "paddle/fluid/imperative/distributed/ProcessGroupNCCL.h"
#include "paddle/fluid/pybind/distributed_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindDistributed(py::module* m) {
  auto processGroup =
      py::class_<imperative::ProcessGroup,
                 std::shared_ptr<imperative::ProcessGroup>>(*m, "ProcessGroup")
          .def(py::init<int, int>())
          .def("rank", &imperative::ProcessGroup::getRank)
          .def("size", &imperative::ProcessGroup::getSize);
  // .def("name", &imperative::ProcessGroup::getBackendName)

  // .def(
  //      "allreduce",
  //      &imperative::ProcessGroup::allreduce,
  //      // py::arg("tensors"),
  //      py::arg("opts") = imperative::AllreduceOptions(),
  //      py::call_guard<py::gil_scoped_release>());
  // .def(
  //      "allreduce",

  // )

  auto processGroupNCCL =
      py::class_<imperative::ProcessGroupNCCL,
                 std::shared_ptr<imperative::ProcessGroupNCCL>>(
          *m, "ProcessGroupNCCL", processGroup)
          .def(py::init<const imperative::ProcessGroupStrategy&, int, int>(),
               py::call_guard<py::gil_scoped_release>());
}

}  // end namespace pybind
}  // namespace paddle
