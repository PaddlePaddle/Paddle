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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pybind/dist_api.h"
#include "paddle/fluid/pybind/dist_static_op_function.h"
#include "paddle/phi/core/enforce.h"

namespace py = pybind11;

using paddle::dialect::OperationDistAttribute;
using paddle::dialect::TensorDistAttribute;

namespace paddle {
namespace pybind {

void BindOperationDistAttribute(py::module *m) {
  py::class_<OperationDistAttribute> dist_attr(*m, "OperationDistAttribute");
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
      .def("num_operand_dist_attrs",
           &OperationDistAttribute::num_operand_dist_attrs)
      .def("operand_dist_attrs", &OperationDistAttribute::operand_dist_attrs)
      .def("operand_dist_attr", &OperationDistAttribute::operand_dist_attr)
      .def("num_result_dist_attrs",
           &OperationDistAttribute::num_result_dist_attrs)
      .def("result_dist_attrs", &OperationDistAttribute::result_dist_attrs)
      .def("result_dist_attr", &OperationDistAttribute::result_dist_attr);
}

void BindTensorDistAttribute(py::module *m) {
  py::class_<TensorDistAttribute> dist_attr(*m, "TensorDistAttribute");
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
      .def_property_readonly("partial_dims", [](TensorDistAttribute &self) {
        return self.partial_dims();
      });
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

void BindDistApi(pybind11::module *module) {
  auto ir_module = module->def_submodule("pir");
  BindOperationDistAttribute(&ir_module);
  BindTensorDistAttribute(&ir_module);
  auto ops_modules = ir_module.def_submodule("ops");
  BindDistOpsAPI(&ops_modules);
}

}  // namespace pybind
}  // namespace paddle
