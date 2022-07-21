// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>

#include "paddle/fluid/distributed/auto_parallel/device_mesh.h"
#include "paddle/fluid/distributed/auto_parallel/dist_attr.h"
#include "paddle/fluid/distributed/auto_parallel/dist_mapper.h"
#include "paddle/fluid/distributed/auto_parallel/process_mesh.h"
#include "paddle/fluid/pybind/auto_parallel_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

using paddle::distributed::auto_parallel::Device;
using paddle::distributed::auto_parallel::DeviceCapability;
using paddle::distributed::auto_parallel::DeviceMesh;
using paddle::distributed::auto_parallel::DistributedMapper;
using paddle::distributed::auto_parallel::Link;
using paddle::distributed::auto_parallel::LinkCapability;
using paddle::distributed::auto_parallel::OperatorDistributedAttribute;
using paddle::distributed::auto_parallel::ProcessMesh;
using paddle::distributed::auto_parallel::TensorDistributedAttribute;

void BindAutoParallel(py::module *m) {
  py::class_<ProcessMesh>(*m, "ProcessMesh")
      .def(py::init<const std::vector<int64_t>,
                    const std::vector<int64_t>,
                    const std::vector<std::string> &,
                    const std::string &>())
      .def_property_readonly("shape", &ProcessMesh::shape)
      .def_property_readonly("process_ids", &ProcessMesh::process_ids)
      .def_property_readonly("dim_names", &ProcessMesh::dim_names)
      .def_property("device_type",
                    &ProcessMesh::device_type,
                    &ProcessMesh::set_device_type);
  //  .def("dim_size",
  //       static_cast<int64_t
  //       (ProcessMesh::*)(int64_t)>(&ProcessMesh::dim_size), "Get the
  //       dimension size by the dimension number.")
  //  .def("dim_size",
  //       static_cast<int64_t
  //       (ProcessMesh::*)(string)>(&ProcessMesh::dim_size), "Get the dimension
  //       size by the dimension name.");
}

}  // namespace pybind
}  // namespace paddle
