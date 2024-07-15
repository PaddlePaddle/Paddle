// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <Python.h>
#include "pybind11/stl.h"

#include "paddle/fluid/pir/dialect/distributed/ir/dist_api.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_tools.h"
#include "paddle/fluid/pir/dialect/distributed/transforms/dist_to_dense_pass.h"
#include "paddle/fluid/pybind/dist_api.h"
#include "paddle/fluid/pybind/dist_static_op_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/enforce.h"

namespace py = pybind11;

namespace pybind11::detail {
template <typename Key,
          typename Value,
          typename Hash,
          typename Equal,
          typename Alloc>
struct type_caster<paddle::flat_hash_map<Key, Value, Hash, Equal, Alloc>>
    : map_caster<paddle::flat_hash_map<Key, Value, Hash, Equal, Alloc>,
                 Key,
                 Value> {};
}  // namespace pybind11::detail

using paddle::dialect::DistTypeInterface;
using paddle::dialect::OperationDistAttribute;
using paddle::dialect::ProcessMeshAttribute;
using paddle::dialect::TensorDistAttribute;

namespace paddle::pybind {

void BindOperationDistAttribute(py::module *m) {
  py::class_<OperationDistAttribute, pir::Attribute> dist_attr(
      *m, "OperationDistAttribute");
  dist_attr
      .def("__str__",
           [](OperationDistAttribute &self) {
             std::ostringstream print_stream;
             print_stream << self;
             return print_stream.str();
           })
      .def_property_readonly("process_mesh",
                             [](OperationDistAttribute &self) {
                               return self.process_mesh_attr().process_mesh();
                             })
      .def("num_operands", &OperationDistAttribute::num_operands)
      .def("operands", &OperationDistAttribute::operands)
      .def("operand", &OperationDistAttribute::operand)
      .def("num_results", &OperationDistAttribute::num_results)
      .def("results", &OperationDistAttribute::results)
      .def("result", &OperationDistAttribute::result);
}

void BindTensorDistAttribute(py::module *m) {
  py::class_<TensorDistAttribute, pir::Attribute> dist_attr(
      *m, "TensorDistAttribute");
  dist_attr
      .def("__str__",
           [](TensorDistAttribute &self) {
             std::ostringstream print_stream;
             print_stream << self;
             return print_stream.str();
           })
      .def_property_readonly("process_mesh",
                             [](TensorDistAttribute &self) {
                               return self.process_mesh_attr().process_mesh();
                             })
      .def_property_readonly(
          "dims_mapping",
          [](TensorDistAttribute &self) { return self.dims_mapping(); })
      .def_property_readonly(
          "partial_status",
          [](TensorDistAttribute &self) { return self.partial_status(); })
      .def_property_readonly(
          "partial_dims",
          [](TensorDistAttribute &self) { return self.partial_dims(); })
      .def_property_readonly("placements", [](TensorDistAttribute &self) {
        return self.placements();
      });
}

void BindDistType(py::module *m) {
  py::class_<DistTypeInterface, pir::Type> dist_type(*m, "DistType");
  dist_type.def("dist_attr", &DistTypeInterface::tensor_dist_attr);
}

void BindDistOpsAPI(pybind11::module *module) {
  {
    if (PyModule_AddFunctions(module->ptr(), DistOpsAPI) < 0) {
      {
        PADDLE_THROW(
            phi::errors::Fatal("Add C++ DistOpsAPI to core.ops failed!"));
      }
    }
  }
}

TensorDistAttribute CreateTensorDistAttribute(
    const phi::distributed::ProcessMesh &mesh,
    const std::vector<int64_t> &dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType> &partial_status = {}) {
  return TensorDistAttribute::get(
      pir::IrContext::Instance(), mesh, dims_mapping, partial_status);
}

OperationDistAttribute CreateOperationDistAttribute(
    const phi::distributed::ProcessMesh &mesh,
    const std::vector<pir::Attribute> &operands,
    const std::vector<pir::Attribute> &results) {
  return OperationDistAttribute::get(
      pir::IrContext::Instance(), mesh, operands, results);
}

void BindDistUtils(pybind11::module *m) {
  m->def("create_tensor_dist_attribute", CreateTensorDistAttribute);
  m->def("create_op_dist_attribute", CreateOperationDistAttribute);
  m->def("get_sub_meshes", phi::distributed::GetSubMeshes);
  m->def("cvt_to_dist_type", &dialect::CvtToPirDistType);
}

void BindDistPassAPI(pybind11::module *module) {
  module->def("apply_dist2dense_pass", paddle::dialect::DistToDensePass);
}

void BindOpsFunction(py::module *m) {
  m->def("reshard_v2",
         [](const pir::Value &x, const TensorDistAttribute &dist_attr) {
           return reshard(x, dist_attr);
         });
}

void BindDistApi(pybind11::module *module) {
  auto ir_module = module->def_submodule("pir");
  BindOperationDistAttribute(&ir_module);
  BindTensorDistAttribute(&ir_module);
  BindDistType(&ir_module);
  BindDistUtils(&ir_module);
  BindDistPassAPI(&ir_module);
  auto ops_modules = ir_module.def_submodule("ops");
  BindDistOpsAPI(&ops_modules);
  BindOpsFunction(&ops_modules);
}

}  // namespace paddle::pybind
