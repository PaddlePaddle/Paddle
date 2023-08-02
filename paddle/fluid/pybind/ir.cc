// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/ir.h"

#include <Python.h>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/value.h"
#include "paddle/phi/core/enforce.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using ir::Block;
using ir::Operation;
using ir::OpOperand;
using ir::OpResult;
using ir::Program;
using ir::Type;
using ir::Value;
using paddle::dialect::APIBuilder;
using paddle::dialect::DenseTensorType;
using pybind11::return_value_policy;

namespace paddle {
namespace pybind {

PyTypeObject *g_ir_opresult_pytype = nullptr;

void BindOpsAPI(pybind11::module *module);

void BindProgram(py::module *m) {
  py::class_<Program> program(*m, "Program");
  program
      .def(
          "__init__",
          [](Program &self) { new (&self) Program(ir::IrContext::Instance()); })
      .def("__str__",
           [](Program &self) {
             std::ostringstream print_stream;
             self.Print(print_stream);
             return print_stream.str();
           })
      .def("parameters_num", &Program::parameters_num)
      .def("block",
           py::overload_cast<>(&Program::block),
           return_value_policy::reference)
      .def("block",
           py::overload_cast<>(&Program::block, py::const_),
           return_value_policy::reference);
}

void BindBlock(py::module *m) {
  py::class_<Block> block(*m, "Block");
  block.def("front", &Block::front, return_value_policy::reference)
      .def("get_parent_program",
           [](Block &self) { return self.GetParentOp()->GetParentProgram(); })
      .def("get_ops",
           [](Block &self) -> py::list {
             py::list op_list;
             for (auto iter = self.begin(); iter != self.end(); iter++) {
               op_list.append(*iter);
             }
             return op_list;
           })
      .def("remove_op", [](Block &self, Operation *op) {
        auto op_iter = std::find(self.begin(), self.end(), op);
        self.erase(op_iter);
      });
}

void BindOperation(py::module *m) {
  py::class_<Operation> op(*m, "Operation");
  op.def("name", &Operation::name)
      .def("get_parent_block",
           py::overload_cast<>(&Operation::GetParent),
           return_value_policy::reference)
      .def("get_parent_block",
           py::overload_cast<>(&Operation::GetParent, py::const_),
           return_value_policy::reference)
      .def("num_operands", &Operation::num_operands)
      .def("num_results", &Operation::num_results)
      .def("operand", &Operation::operand)
      .def("result", &Operation::result)
      .def("operand_source", &Operation::operand_source)
      .def("operands",
           [](Operation &self) -> py::list {
             py::list op_list;
             for (uint32_t i = 0; i < self.num_operands(); i++) {
               op_list.append(self.operand(i));
             }
             return op_list;
           })
      .def("results",
           [](Operation &self) -> py::list {
             py::list op_list;
             for (uint32_t i = 0; i < self.num_results(); i++) {
               op_list.append(self.result(i));
             }
             return op_list;
           })
      .def("operands_source",
           [](Operation &self) -> py::list {
             py::list op_list;
             for (uint32_t i = 0; i < self.num_operands(); i++) {
               op_list.append(self.operand_source(i));
             }
             return op_list;
           })
      .def("get_input_names",
           [](Operation &self) -> py::list {
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto inputs_info = std::get<0>(yaml_interface.GetOpInfo());
             for (auto &input_info : inputs_info) {
               op_list.append(input_info.name);
             }
             return op_list;
           })
      .def("get_attr_names",
           [](Operation &self) -> py::list {
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto attrs_info = std::get<1>(yaml_interface.GetOpInfo());
             for (auto &attr_info : attrs_info) {
               op_list.append(attr_info.name);
             }
             return op_list;
           })
      .def("get_output_names",
           [](Operation &self) -> py::list {
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto outputs_info = std::get<2>(yaml_interface.GetOpInfo());
             for (auto &output_info : outputs_info) {
               op_list.append(output_info.name);
             }
             return op_list;
           })
      .def("replace_all_uses_with",
           [](Operation &self, const std::vector<OpResult> &op_results) {
             self.ReplaceAllUsesWith(op_results);
           });
}

void BindValue(py::module *m) {
  py::class_<Value> value(*m, "Value");
  value
      .def("get_defining_op",
           &Value::GetDefiningOp,
           return_value_policy::reference)
      .def("__eq__", &Value::operator==);
}

void BindOpOperand(py::module *m) {
  py::class_<OpOperand> op_operand(*m, "OpOperand");
  op_operand
      .def("source",
           [](OpOperand &self) { return self.source().dyn_cast<OpResult>(); })
      .def("set_source", [](OpOperand &self, const OpResult &result) {
        self.set_source(result);
      });
}

void BindOpResult(py::module *m) {
  py::class_<OpResult> op_result(*m, "OpResult");
  g_ir_opresult_pytype = reinterpret_cast<PyTypeObject *>(op_result.ptr());
  op_result
      .def("get_defining_op",
           &OpResult::GetDefiningOp,
           return_value_policy::reference)
      .def("use_empty", &OpResult::use_empty)
      .def("type", &OpResult::type)
      .def("set_stop_gradient",
           [](OpResult &self, bool stop_gradient) {
             auto *defining_op = self.owner();
             std::vector<ir::Attribute> stop_gradients;
             if (defining_op->HasAttribute(kAttrStopGradients)) {
               stop_gradients = defining_op->attribute(kAttrStopGradients)
                                    .dyn_cast<ir::ArrayAttribute>()
                                    .AsVector();
             } else {
               stop_gradients = std::vector<ir::Attribute>(
                   defining_op->num_results(),
                   ir::BoolAttribute::get(ir::IrContext::Instance(), false));
             }
             stop_gradients[self.GetResultIndex()] = ir::BoolAttribute::get(
                 ir::IrContext::Instance(), stop_gradient);
             defining_op->set_attribute(
                 kAttrStopGradients,
                 ir::ArrayAttribute::get(ir::IrContext::Instance(),
                                         stop_gradients));
           })
      .def("get_stop_gradient", [](OpResult &self) {
        auto *defining_op = self.owner();
        if (defining_op->HasAttribute(kAttrStopGradients)) {
          auto stop_gradients = defining_op->attribute(kAttrStopGradients)
                                    .dyn_cast<ir::ArrayAttribute>()
                                    .AsVector();
          return stop_gradients[self.GetResultIndex()]
              .dyn_cast<ir::BoolAttribute>()
              .data();
        } else {
          return false;
        }
      });
}

void BindType(py::module *m) {
  py::class_<Type> ir_type(*m, "Type");
  ir_type.def("__eq__", [](Type &self, Type &other) { return self == other; })
      .def("__str__", [](Type &self) {
        std::ostringstream print_stream;
        print_stream << self;
        return print_stream.str();
      });
}

void BindUtils(pybind11::module *m) {
  m->def("get_op_result_shape", [](const OpResult &op_result) {
    if (op_result.type().isa<DenseTensorType>()) {
      return phi::vectorize(
          op_result.type().dyn_cast<DenseTensorType>().dims());
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "get_op_result_shape currently only support op_result that is a "
          "DenseTensorType"));
    }
  });
  m->def("get_op_result_dtype", [](const OpResult &op_result) {
    if (op_result.type().isa<DenseTensorType>()) {
      return op_result.type().dyn_cast<DenseTensorType>().dtype();
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "get_op_result_dtype currently only support op_result that is a "
          "DenseTensorType"));
    }
  });
  m->def("set_global_program",
         [](Program *program) { APIBuilder::Instance().SetProgram(program); });
  m->def("set_insertion_point",
         [](Operation *op) { APIBuilder::Instance().SetInsertionPoint(op); });
  m->def("reset_insertion_point_to_start",
         []() { APIBuilder::Instance().ResetInsertionPointToStart(); });
  m->def("reset_insertion_point_to_end",
         []() { APIBuilder::Instance().ResetInsertionPointToEnd(); });
  m->def("translate_to_new_ir", &paddle::TranslateLegacyProgramToProgram);
}

void BindNewIR(pybind11::module *module) {
  auto ir_module = module->def_submodule("ir");
  BindProgram(&ir_module);
  BindBlock(&ir_module);
  BindOperation(&ir_module);
  BindValue(&ir_module);
  BindOpOperand(&ir_module);
  BindOpResult(&ir_module);
  BindType(&ir_module);
  BindUtils(&ir_module);
  auto ops_modules = ir_module.def_submodule("ops");
  BindOpsAPI(&ops_modules);
}

}  // namespace pybind
}  // namespace paddle
