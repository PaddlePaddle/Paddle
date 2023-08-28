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

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/pybind/auto_parallel_py.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/auto_parallel/device_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_mapper.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/utils/optional.h"
#include "paddle/utils/pybind.h"

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/common.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dist_tensor_spec.h"
#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/r_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/s_to_r_reshard_function.h"
#endif

namespace py = pybind11;

namespace paddle {
namespace pybind {

using paddle::distributed::auto_parallel::DistTensorSpec;
using paddle::distributed::auto_parallel::kDefault;
using paddle::distributed::auto_parallel::OperatorDistAttr;
using paddle::distributed::auto_parallel::SPMDRuleBase;
using paddle::distributed::auto_parallel::SPMDRuleMap;
using paddle::framework::OpDesc;
using paddle::framework::VarDesc;
using phi::distributed::ProcessMesh;
using phi::distributed::TensorDistAttr;
using phi::distributed::auto_parallel::Device;
using phi::distributed::auto_parallel::DeviceCapability;
using phi::distributed::auto_parallel::DeviceMesh;
using phi::distributed::auto_parallel::DistributedMapper;
using phi::distributed::auto_parallel::Link;
using phi::distributed::auto_parallel::LinkCapability;
using phi::distributed::auto_parallel::Machine;

PyTypeObject *g_tensor_dist_attr_pytype = nullptr;

static inline const ProcessMesh *get_tensor_process_mesh(
    const TensorDistAttr &self) {
  if (self.process_mesh().empty()) {
    return nullptr;
  } else {
    return &self.process_mesh();
  }
}

static inline void set_tensor_process_mesh(TensorDistAttr *self,
                                           const ProcessMesh *process_mesh) {
  if (process_mesh) {
    self->set_process_mesh(*process_mesh);
  } else {
    self->set_process_mesh(ProcessMesh());
  }
}

static inline const ProcessMesh *get_operator_process_mesh(
    const OperatorDistAttr &self) {
  if (self.process_mesh().empty()) {
    return nullptr;
  } else {
    return &self.process_mesh();
  }
}

static inline void set_operator_process_mesh(OperatorDistAttr *self,
                                             const ProcessMesh *process_mesh) {
  if (process_mesh) {
    self->set_process_mesh(*process_mesh);
  } else {
    self->set_process_mesh(ProcessMesh());
  }
}

static inline void reset_tensor_dist_attr(TensorDistAttr *dist_attr) {
  dist_attr->set_process_mesh(ProcessMesh());
  std::vector<int64_t> dims_mapping(dist_attr->dims_mapping().size(), -1);
  dist_attr->set_dims_mapping(dims_mapping);
  dist_attr->clear_annotated();
}

static inline void reset_operator_dist_attr(OperatorDistAttr *dist_attr) {
  for (auto &item : dist_attr->input_dist_attrs()) {
    reset_tensor_dist_attr(&item.second);
  }
  for (auto &item : dist_attr->output_dist_attrs()) {
    reset_tensor_dist_attr(&item.second);
  }
  dist_attr->set_impl_type(kDefault);
  dist_attr->set_impl_idx(0);
  dist_attr->clear_annotated();
}

void BindAutoParallel(py::module *m) {
#ifdef PADDLE_WITH_DISTRIBUTE
  auto ReshardFunction =
      py::class_<phi::distributed::ReshardFunction>(*m, "ReshardFunction")
          .def(
              "is_suitable",
              [](phi::distributed::ReshardFunction &self,
                 py::handle py_tensor,
                 const phi::distributed::TensorDistAttr &dist_attr) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dist =
                    std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                        tensor.impl());
                return self.IsSuitable(*p_dist, dist_attr);
              },
              py::call_guard<py::gil_scoped_release>())
          .def(
              "eval",
              [](phi::distributed::ReshardFunction &self,
                 phi::DeviceContext *dev_ctx,
                 py::handle py_tensor,
                 const phi::distributed::TensorDistAttr &dist_attr) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                auto p_dist =
                    std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                        tensor.impl());
                auto res_dist = self.Eval(dev_ctx, *p_dist, dist_attr);
                return paddle::Tensor(res_dist);
              },
              py::call_guard<py::gil_scoped_release>());

  py::class_<phi::distributed::RToSReshardFunction>(
      *m, "RToSReshardFunction", ReshardFunction)
      .def(py::init<>());

  py::class_<phi::distributed::SToRReshardFunction>(
      *m, "SToRReshardFunction", ReshardFunction)
      .def(py::init<>());
#endif

  py::class_<ProcessMesh>(*m, "ProcessMesh")
      .def(py::init<>())
      .def(py::init<const std::vector<int64_t> &,
                    const std::vector<int64_t> &,
                    const std::vector<std::string> &>(),
           py::arg("shape"),
           py::arg("process_ids"),
           py::arg("dim_names"))
      .def_property_readonly("shape", &ProcessMesh::shape)
      .def_property_readonly("process_ids", &ProcessMesh::process_ids)
      .def_property_readonly("dim_names", &ProcessMesh::dim_names)
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
      .def("__copy__",
           [](const ProcessMesh &self) { return ProcessMesh(self); })
      .def(
          "__deepcopy__",
          [](const ProcessMesh &self, py::dict) { return ProcessMesh(self); },
          py::arg("memo"))
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
      .def_property_readonly("devices", &Machine::devices)
      .def_property_readonly("links", &Machine::links)
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
           py::arg("device_ids"),
           py::arg("dim_names"))
      .def_property_readonly("name", &DeviceMesh::name)
      .def_property_readonly("shape", &DeviceMesh::shape)
      .def_property_readonly("device_ids", &DeviceMesh::device_ids)
      .def_property_readonly("dim_names", &DeviceMesh::dim_names)
      .def_property_readonly("device_type", &DeviceMesh::device_type)
      .def_property_readonly("size", &DeviceMesh::size)
      .def_property_readonly("ndim", &DeviceMesh::ndim)
      .def_property_readonly("devices", &DeviceMesh::devices)
      .def_property_readonly("links", &DeviceMesh::links)
      .def_property_readonly("machines", &DeviceMesh::machines)
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
      .def("__copy__",
           [](const TensorDistAttr &self) { return TensorDistAttr(self); })
      .def(
          "__deepcopy__",
          [](const TensorDistAttr &self, py::dict) {
            return TensorDistAttr(self);
          },
          py::arg("memo"))
      .def("__str__", &DeviceMesh::to_string);

  py::class_<TensorDistAttr> py_dist_attr(*m, "TensorDistAttr");
  g_tensor_dist_attr_pytype =
      reinterpret_cast<PyTypeObject *>(py_dist_attr.ptr());
  py_dist_attr.def(py::init<>())
      .def(py::init([](const VarDesc &var_desc) {
        auto shape =
            paddle::distributed::auto_parallel::get_tensor_shape(&var_desc);
        return std::make_unique<TensorDistAttr>(shape);
      }))
      .def(py::init<const TensorDistAttr &>())
      .def_property(
          "process_mesh", &get_tensor_process_mesh, &set_tensor_process_mesh)
      .def_property("dims_mapping",
                    &TensorDistAttr::dims_mapping,
                    &TensorDistAttr::set_dims_mapping)
      .def_property("batch_dim",
                    &TensorDistAttr::batch_dim,
                    &TensorDistAttr::set_batch_dim)
      .def_property("dynamic_dims",
                    &TensorDistAttr::dynamic_dims,
                    &TensorDistAttr::set_dynamic_dims)
      .def_property("annotated",
                    &TensorDistAttr::annotated,
                    &TensorDistAttr::set_annotated)
      .def("is_annotated", &TensorDistAttr::is_annotated)
      .def("mark_annotated", &TensorDistAttr::mark_annotated)
      .def("clear_annotated", &TensorDistAttr::clear_annotated)
      .def(
          "verify",
          [](TensorDistAttr &self, const VarDesc *tensor) {
            auto shape =
                paddle::distributed::auto_parallel::get_tensor_shape(tensor);
            return self.verify(shape);
          },
          py::arg("tensor") = static_cast<VarDesc *>(nullptr))
      .def("reset", &reset_tensor_dist_attr)
      .def("serialize_to_string",
           [](TensorDistAttr &self) {
             return py::bytes(self.serialize_to_string());
           })
      .def("parse_from_string", &TensorDistAttr::parse_from_string)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__copy__",
           [](const TensorDistAttr &self) { return TensorDistAttr(self); })
      .def(
          "__deepcopy__",
          [](const TensorDistAttr &self, py::dict) {
            return TensorDistAttr(self);
          },
          py::arg("memo"))
      .def("__str__", &TensorDistAttr::to_string)
      .def("_is_partial", &TensorDistAttr::is_partial)
      .def("_partial_dims", &TensorDistAttr::partial_dims)
      .def("_clean_partial_dims", &TensorDistAttr::clean_partial_dims)
      .def("_clean_partial_status", &TensorDistAttr::clean_partial_status);

  py::class_<SPMDRuleBase>(*m, "SPMDRuleBase")
      .def("infer_forward", &SPMDRuleBase::InferForward)
      .def("infer_backward",
           static_cast<std::pair<std::vector<TensorDistAttr>,
                                 std::vector<TensorDistAttr>> (SPMDRuleBase::*)(
               const std::vector<DistTensorSpec> &,
               const std::vector<DistTensorSpec> &,
               const paddle::framework::AttributeMap &)>(
               &SPMDRuleBase::InferBackward));
  // .def("infer_backward", &SPMDRuleBase::InferBackward) [revert in future]

  py::class_<phi::distributed::SpmdRule>(*m, "SpmdRule")
      .def("infer_forward",
           [](const phi::distributed::SpmdRule &self,
              const std::vector<DistTensorSpec> &input_specs,
              const std::vector<phi::Attribute> &attrs) {
             phi::distributed::InferSpmdContext ctx;
             std::vector<phi::distributed::DistTensor> input_tensors;
             for (auto &spec : input_specs) {
               input_tensors.emplace_back(phi::distributed::DistTensor(
                   phi::make_ddim(spec.shape()), spec.dist_attr()));
             }
             for (auto &tensor : input_tensors) {
               ctx.EmplaceBackInput(phi::MetaTensor(tensor));
             }
             for (auto &attr : attrs) {
               ctx.EmplaceBackAttr(attr);
             }
             return self.InferForward(ctx);
           })
      .def("infer_backward",
           [](const phi::distributed::SpmdRule &self,
              const std::vector<DistTensorSpec> &input_specs,
              const std::vector<phi::Attribute> &attrs) {
             phi::distributed::InferSpmdContext ctx;
             std::vector<phi::distributed::DistTensor> input_tensors;
             for (auto &spec : input_specs) {
               input_tensors.emplace_back(phi::distributed::DistTensor(
                   phi::make_ddim(spec.shape()), spec.dist_attr()));
             }
             for (auto &tensor : input_tensors) {
               ctx.EmplaceBackInput(phi::MetaTensor(tensor));
             }
             for (auto &attr : attrs) {
               ctx.EmplaceBackAttr(attr);
             }
             return self.InferBackward(ctx);
           });

  py::class_<DistTensorSpec>(*m, "DistTensorSpec")
      .def(py::init<>())
      .def(py::init<const DistTensorSpec &>())
      .def(py::init<const std::vector<int64_t> &, const TensorDistAttr &>())
      .def("dims_mapping", &DistTensorSpec::dims_mapping)
      .def("set_dims_mapping", &DistTensorSpec::set_dims_mapping)
      .def("process_mesh", &DistTensorSpec::process_mesh)
      .def("set_process_mesh", &DistTensorSpec::set_process_mesh)
      .def_property("shape", &DistTensorSpec::shape, &DistTensorSpec::set_shape)
      .def("__str__", &DistTensorSpec::to_string)
      .def("__copy__",
           [](const DistTensorSpec &self) { return DistTensorSpec(self); })
      .def(
          "__deepcopy__",
          [](const DistTensorSpec &self, py::dict) {
            return DistTensorSpec(self);
          },
          py::arg("memo"));

  py::class_<OperatorDistAttr>(*m, "OperatorDistAttr")
      .def(py::init<>())
      .def(py::init<const OpDesc &>())
      .def(py::init<const OperatorDistAttr &>())
      .def_property(
          "op_type", &OperatorDistAttr::op_type, &OperatorDistAttr::set_op_type)
      .def_property("process_mesh",
                    &get_operator_process_mesh,
                    &set_operator_process_mesh)
      .def_property("impl_type",
                    &OperatorDistAttr::impl_type,
                    &OperatorDistAttr::set_impl_type)
      .def_property("impl_idx",
                    &OperatorDistAttr::impl_idx,
                    &OperatorDistAttr::set_impl_idx)
      .def_property("is_recompute",
                    &OperatorDistAttr::is_recompute,
                    &OperatorDistAttr::set_is_recompute)
      .def_property("execution_stream",
                    &OperatorDistAttr::execution_stream,
                    &OperatorDistAttr::set_execution_stream)
      .def_property("stream_priority",
                    &OperatorDistAttr::stream_priority,
                    &OperatorDistAttr::set_stream_priority)
      .def_property("scheduling_priority",
                    &OperatorDistAttr::scheduling_priority,
                    &OperatorDistAttr::set_scheduling_priority)
      .def_property("annotated",
                    &OperatorDistAttr::annotated,
                    &OperatorDistAttr::set_annotated)
      .def_property(
          "inputs_dist_attrs",
          static_cast<std::map<std::string, TensorDistAttr> &(
              OperatorDistAttr::*)()>(&OperatorDistAttr::input_dist_attrs),
          &OperatorDistAttr::set_input_dist_attrs)
      .def_property(
          "outputs_dist_attrs",
          static_cast<std::map<std::string, TensorDistAttr> &(
              OperatorDistAttr::*)()>(&OperatorDistAttr::output_dist_attrs),
          &OperatorDistAttr::set_output_dist_attrs)
      .def("get_input_dist_attr",
           static_cast<TensorDistAttr &(
               OperatorDistAttr::*)(const std::string &)>(
               &OperatorDistAttr::input_dist_attr),
           py::return_value_policy::reference)
      .def("get_output_dist_attr",
           static_cast<TensorDistAttr &(
               OperatorDistAttr::*)(const std::string &)>(
               &OperatorDistAttr::output_dist_attr),
           py::return_value_policy::reference)
      .def("set_input_dist_attr", &OperatorDistAttr::set_input_dist_attr)
      .def("set_output_dist_attr", &OperatorDistAttr::set_output_dist_attr)
      .def("del_input_dist_attr",  // TODO(aoyulong): move into dist_attr.cc
           [](OperatorDistAttr &self, const std::string &name) {
             self.input_dist_attrs().erase(name);
           })
      .def("del_output_dist_attr",  // TODO(aoyulong): move into dist_attr.cc
           [](OperatorDistAttr &self, const std::string &name) {
             self.output_dist_attrs().erase(name);
           })
      .def("is_annotated", &OperatorDistAttr::is_annotated)
      .def("mark_annotated", &OperatorDistAttr::mark_annotated)
      .def("clear_annotated", &OperatorDistAttr::clear_annotated)
      .def("get_input_dims_mapping",
           &OperatorDistAttr::input_dims_mapping,
           py::return_value_policy::reference)
      .def("set_input_dims_mapping", &OperatorDistAttr::set_input_dims_mapping)
      .def("get_output_dims_mapping",
           &OperatorDistAttr::output_dims_mapping,
           py::return_value_policy::reference)
      .def("set_output_dims_mapping",
           &OperatorDistAttr::set_output_dims_mapping)
      .def("verify",
           &OperatorDistAttr::verify,
           py::arg("op") = static_cast<OpDesc *>(nullptr))
      .def("is_annotated_input_dims_mapping",
           [](const OperatorDistAttr &self, const std::string &name) {
             return self.input_dist_attr(name).is_annotated("dims_mapping");
           })
      .def("is_annotated_output_dims_mapping",
           [](const OperatorDistAttr &self, const std::string &name) {
             return self.output_dist_attr(name).is_annotated("dims_mapping");
           })
      .def("rename_input", &OperatorDistAttr::rename_input)
      .def("rename_output", &OperatorDistAttr::rename_output)
      .def("reset", &reset_operator_dist_attr)
      .def("serialize_to_string",
           [](OperatorDistAttr &self) {
             return py::bytes(self.serialize_to_string());
           })
      .def("parse_from_string", &OperatorDistAttr::parse_from_string)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__copy__",
           [](const OperatorDistAttr &self) { return OperatorDistAttr(self); })
      .def(
          "__deepcopy__",
          [](const OperatorDistAttr &self, py::dict) {
            return OperatorDistAttr(self);
          },
          py::arg("memo"))
      .def("__str__", &OperatorDistAttr::to_string);

  m->def(
      "get_spmd_rule",
      [](const std::string op_type) {
        return SPMDRuleMap::Instance().Get(op_type);
      },
      py::return_value_policy::reference);

  m->def(
      "get_phi_spmd_rule",
      [](const std::string op_type) {
        return phi::distributed::SpmdRuleFactory::Instance().GetSpmdRule(
            op_type);
      },
      py::return_value_policy::reference);

  // TODO(liuzhenhai): DistributedMapper is not used for now, but
  // dist_mapper_test need the symbols forch DistributedMapper to be linked,
  // remove it latter
  m->def("touch_dist_mapper", []() {
    DistributedMapper mapper;
    return mapper.to_string();
  });
}

}  // namespace pybind
}  // namespace paddle
