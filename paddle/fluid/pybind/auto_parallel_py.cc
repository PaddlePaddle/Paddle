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

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "paddle/fluid/distributed/auto_parallel/device_mesh.h"
#include "paddle/fluid/distributed/auto_parallel/dist_attr.h"
#include "paddle/fluid/distributed/auto_parallel/dist_mapper.h"
#include "paddle/fluid/distributed/auto_parallel/process_mesh.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
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
using paddle::distributed::auto_parallel::Machine;
using paddle::distributed::auto_parallel::OperatorDistAttr;
using paddle::distributed::auto_parallel::ProcessMesh;
using paddle::distributed::auto_parallel::TensorDistAttr;
using paddle::framework::OpDesc;
using paddle::framework::VarDesc;

void BindAutoParallel(py::module *m) {
  py::class_<ProcessMesh>(*m, "ProcessMesh")
      .def(py::init<const std::vector<int64_t> &,
                    const std::vector<int64_t> &,
                    const std::vector<std::string> &>(),
           py::arg("shape"),
           py::arg("process_ids"),
           py::arg("dim_names"))
      .def_property_readonly(
          "shape", &ProcessMesh::shape, py::return_value_policy::reference)
      .def_property_readonly("process_ids",
                             &ProcessMesh::process_ids,
                             py::return_value_policy::reference)
      .def_property_readonly("dim_names",
                             &ProcessMesh::dim_names,
                             py::return_value_policy::reference)
      .def_property_readonly("size", &ProcessMesh::size)
      .def_property_readonly("ndim", &ProcessMesh::ndim)
      .def("dim_size",
           static_cast<int64_t (ProcessMesh::*)(int64_t) const>(
               &ProcessMesh::dim_size))
      .def("dim_size",
           static_cast<int64_t (ProcessMesh::*)(const std::string &) const>(
               &ProcessMesh::dim_size))
      .def("empty", &ProcessMesh::empty)
      .def("contains", &ProcessMesh::contains)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", &ProcessMesh::to_string);

  py::class_<DeviceCapability>(*m, "DeviceCapability")
      .def(py::init<>())
      .def_readwrite("sflops", &DeviceCapability::single_precision_flops)
      .def_readwrite("dflops", &DeviceCapability::double_precision_flops)
      .def_readwrite("memory", &DeviceCapability::memory_size_in_bytes)
      .def_readwrite("rate", &DeviceCapability::clock_rate_in_ghz)
      .def("__str__", &DeviceCapability::to_string);

  py::class_<Device>(*m, "Device")
      .def(py::init<int64_t, int64_t, int64_t, const std::string &>(),
           py::arg("global_id"),
           py::arg("local_id"),
           py::arg("machine_id"),
           py::arg("type"))
      .def_property_readonly("global_id", &Device::global_id)
      .def_property_readonly("local_id", &Device::local_id)
      .def_property_readonly("machine_id", &Device::machine_id)
      .def_property_readonly("type", &Device::type)
      .def_property("capability", &Device::capability, &Device::set_capability)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", &Device::to_string);

  py::class_<LinkCapability>(*m, "LinkCapability")
      .def(py::init<>())
      .def_readwrite("bandwidth", &LinkCapability::bandwidth)
      .def_readwrite("latency", &LinkCapability::latency)
      .def("__str__", &LinkCapability::to_string);

  py::class_<Link>(*m, "Link")
      .def(py::init<int64_t, int64_t, const std::string &>(),
           py::arg("source_id"),
           py::arg("target_id"),
           py::arg("type"))
      .def_property_readonly("source_id", &Link::source_id)
      .def_property_readonly("target_id", &Link::target_id)
      .def_property_readonly("type", &Link::type)
      .def_property("capability", &Link::capability, &Link::set_capability)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", &Link::to_string);

  py::class_<Machine>(*m, "Machine")
      .def_property_readonly("id", &Machine::id)
      .def_property_readonly(
          "devices", &Machine::devices, py::return_value_policy::reference)
      .def_property_readonly(
          "links", &Machine::links, py::return_value_policy::reference)
      .def("device", &Machine::device)
      .def("link", &Machine::link)
      .def("contains", &Machine::contains)
      .def("__str__", &Machine::to_string);

  py::class_<DeviceMesh>(*m, "DeviceMesh")
      .def(py::init<const std::string &,
                    const std::vector<int64_t> &,
                    const std::vector<int64_t> &,
                    const std::vector<std::string> &>(),
           py::arg("name"),
           py::arg("shape"),
           py::arg("process_ids"),
           py::arg("dim_names"))
      .def_property_readonly("name", &DeviceMesh::name)
      .def_property_readonly("shape", &DeviceMesh::shape)
      .def_property_readonly("device_ids",
                             &DeviceMesh::device_ids,
                             py::return_value_policy::reference)
      .def_property_readonly("dim_names",
                             &DeviceMesh::dim_names,
                             py::return_value_policy::reference)
      .def_property_readonly("device_type", &DeviceMesh::device_type)
      .def_property_readonly("size", &DeviceMesh::size)
      .def_property_readonly("ndim", &DeviceMesh::ndim)
      .def_property_readonly(
          "devices", &DeviceMesh::devices, py::return_value_policy::reference)
      .def_property_readonly(
          "links", &DeviceMesh::links, py::return_value_policy::reference)
      .def_property_readonly(
          "machines", &DeviceMesh::machines, py::return_value_policy::reference)
      .def("device", &DeviceMesh::device)
      .def("link", &DeviceMesh::link)
      .def("machine", &DeviceMesh::machine)
      .def("empty", &DeviceMesh::empty)
      .def("contains", &DeviceMesh::contains)
      .def("add_device", &DeviceMesh::add_device)
      .def("add_link", &DeviceMesh::add_link)
      .def("dim_size",
           static_cast<int64_t (DeviceMesh::*)(int64_t) const>(
               &DeviceMesh::dim_size))
      .def("dim_size",
           static_cast<int64_t (DeviceMesh::*)(const std::string &) const>(
               &DeviceMesh::dim_size))
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", &DeviceMesh::to_string);

  py::class_<TensorDistAttr>(*m, "TensorDistAttr")
      .def(py::init<const VarDesc &>())
      .def_property_readonly("tensor", &TensorDistAttr::tensor)
      .def_property("process_mesh",
                    &TensorDistAttr::process_mesh,
                    &TensorDistAttr::set_process_mesh)
      .def_property("dims_mapping",
                    &TensorDistAttr::dims_mapping,
                    &TensorDistAttr::set_dims_mapping)
      .def_property("batch_dim",
                    &TensorDistAttr::batch_dim,
                    &TensorDistAttr::set_batch_dim)
      .def_property("dynamic_dims",
                    &TensorDistAttr::dynamic_dims,
                    &TensorDistAttr::set_dynamic_dims)
      .def("is_annotated", &TensorDistAttr::is_annotated)
      .def("annotate", &TensorDistAttr::annotate)
      .def("verify", &TensorDistAttr::verify)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", &TensorDistAttr::to_string);

  py::class_<OperatorDistAttr>(*m, "OperatorDistAttr")
      .def(py::init<const OpDesc &>())
      .def_property_readonly("op", &OperatorDistAttr::op)
      .def_property("process_mesh",
                    &OperatorDistAttr::process_mesh,
                    &OperatorDistAttr::set_process_mesh)
      .def_property("impl_type",
                    &OperatorDistAttr::impl_type,
                    &OperatorDistAttr::set_impl_type)
      .def_property("impl_idx",
                    &OperatorDistAttr::impl_idx,
                    &OperatorDistAttr::set_impl_idx)
      .def("input", &OperatorDistAttr::input)
      .def("output", &OperatorDistAttr::output)
      .def("input_dist_attrs",
           &OperatorDistAttr::input_dist_attrs,
           py::return_value_policy::reference)
      .def("output_dist_attrs",
           &OperatorDistAttr::output_dist_attrs,
           py::return_value_policy::reference)
      .def("input_dist_attr",
           static_cast<TensorDistAttr &(
               OperatorDistAttr::*)(const std::string &)>(
               &OperatorDistAttr::input_dist_attr),
           py::return_value_policy::reference)
      .def("output_dist_attr",
           static_cast<TensorDistAttr &(
               OperatorDistAttr::*)(const std::string &)>(
               &OperatorDistAttr::output_dist_attr),
           py::return_value_policy::reference)
      .def("set_input_dist_attr", &OperatorDistAttr::set_input_dist_attr)
      .def("set_output_dist_attr", &OperatorDistAttr::set_output_dist_attr)
      .def("is_annotated", &OperatorDistAttr::is_annotated)
      .def("annotate", &OperatorDistAttr::annotate)
      .def("verify", &OperatorDistAttr::verify)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", &OperatorDistAttr::to_string);
}

}  // namespace pybind
}  // namespace paddle
