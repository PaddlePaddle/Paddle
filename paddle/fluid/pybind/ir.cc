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

#include "paddle/fluid/pybind/pybind_variant_caster.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/interface/op_yaml_info.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/api_builder.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_dialect.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/utils.h"
#include "paddle/fluid/ir/transforms/inplace_pass.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/value.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_manager.h"
#include "paddle/ir/pass/pass_registry.h"
#include "paddle/ir/transforms/dead_code_elimination_pass.h"
#include "paddle/phi/core/enforce.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using ir::Block;
using ir::Operation;
using ir::OpOperand;
using ir::OpResult;
using ir::Pass;
using ir::PassManager;
using ir::Program;
using ir::Type;
using ir::Value;
using paddle::dialect::APIBuilder;
using paddle::dialect::DenseTensorType;
using pybind11::return_value_policy;

USE_PASS(dead_code_elimination);
USE_PASS(inplace);

namespace paddle {
namespace pybind {

PyTypeObject *g_ir_opresult_pytype = nullptr;

void BindOpsAPI(pybind11::module *module);

void BindProgram(py::module *m) {
  py::class_<Program, std::shared_ptr<Program>> program(*m, "Program", R"DOC(
    Create Python Program. Program is an abstraction of model structure, divided into
    computational graphs and weights. The Program has a main block that stores the computational
    graphs.

    A set of Program usually contains startup program and main program.
    A startup program is set to contain some initial work, eg. initialize the ``Parameter``, and the main
    program will contain the network structure and vars for train.

    A set of Program can be used for test or train, in train program ,
    Paddle will contain all content to build a train network,  in test
    program Paddle will prune some content which is irrelevant to test, eg.
    backward ops and vars.

    **Notes**:
        **we have** :ref:`api_paddle_static_default_startup_program` **and** :ref:`api_paddle_static_default_main_program`
        **by default, a pair of them will shared the parameters. The** :ref:`api_paddle_static_default_startup_program` **only run once to initialize parameters,**
        :ref:`api_paddle_static_default_main_program` **run in every mini batch and adjust the weights.**

    Returns:
        Program: An empty Program.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.static as static

            paddle.enable_static()

            main_program = static.Program()
            startup_program = static.Program()
            with static.program_guard(main_program=main_program, startup_program=startup_program):
                x = static.data(name="x", shape=[-1, 784], dtype='float32')
                y = static.data(name="y", shape=[-1, 1], dtype='int32')
                z = static.nn.fc(name="fc", x=x, size=10, activation="relu")

            print("main program is: {}".format(main_program))
            print("start up program is: {}".format(startup_program))
  )DOC");
  program
      .def(
          "__init__",
          [](Program &self) { new (&self) Program(ir::IrContext::Instance()); })
      .def("__str__",
           [](const std::shared_ptr<Program> &self) {
             std::ostringstream print_stream;
             self->Print(print_stream);
             return print_stream.str();
           })
      .def("parameters_num",
           [](const std::shared_ptr<Program> &self) {
             return self->parameters_num();
           })
      .def(
          "block",
          [](std::shared_ptr<Program> self) { return self->block(); },
          return_value_policy::reference)
      .def(
          "block",
          [](const std::shared_ptr<Program> &self) { return self->block(); },
          return_value_policy::reference);
}

void BindBlock(py::module *m) {
  py::class_<Block> block(*m, "Block", R"DOC(
    In IR, a Block has a list of Operation and can represent a sub computational graph.

    Notes:
        The constructor of Block should not be invoked directly. You can
        use `Program.block()` to get a block.
  )DOC");
  block.def("front", &Block::front, return_value_policy::reference)
      .def("get_parent_program",
           [](Block &self) { return self.GetParentOp()->GetParentProgram(); })
      .def_property_readonly(
          "ops",
          [](Block &self) -> py::list {
            py::list op_list;
            for (auto iter = self.begin(); iter != self.end(); iter++) {
              op_list.append(*iter);
            }
            return op_list;
          })
      .def(
          "remove_op",
          [](Block &self, Operation *op) {
            auto op_iter = std::find(self.begin(), self.end(), op);
            self.erase(op_iter);
          },
          R"DOC(
        Remove the specific position operator.

        Args:
            index(int): the position that the operator to insert.

        Returns:
            None

      )DOC");
}

void BindOperation(py::module *m) {
  py::class_<Operation> op(*m, "Operation", R"DOC(
    In IR, all the operation are represented by Operation, and Operation
    is regarded as a build in an instruction of a Block. Users can call
    python api to describe their neural network.

    Notes:
        The constructor of operator should not be invoked directly. Use
        python api, for example: paddle.mean for building mean operation.

  )DOC");
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
      .def("operands", &Operation::operands)
      .def("results", &Operation::results)
      .def("attrs",
           [](Operation &self) -> py::dict {
             py::dict attrs_dict;
             for (auto &pair : self.attributes()) {
               attrs_dict[pair.first.c_str()] =
                   paddle::dialect::GetAttributeData(pair.second);
             }
             return attrs_dict;
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
      .def("get_input_grad_semantics",
           [](Operation &self) -> py::list {
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto inputs_grad_info = std::get<0>(yaml_interface.GetOpInfo());
             for (auto &input_grad_info : inputs_grad_info) {
               op_list.append(input_grad_info.with_grad_semantic);
             }
             return op_list;
           })
      .def("replace_all_uses_with",
           [](Operation &self, const std::vector<OpResult> &op_results) {
             self.ReplaceAllUsesWith(op_results);
           });
}

void BindValue(py::module *m) {
  py::class_<Value> value(*m, "Value", R"DOC(
    Value class represents the SSA value in the IR system. It is a directed edge
    and a base class.

    Notes:
        The constructor of Value should not be invoked directly. Value can be automatically constructed
        when build network.

  )DOC");
  value
      .def("get_defining_op",
           &Value::GetDefiningOp,
           return_value_policy::reference)
      .def("first_use", &Value::first_use, return_value_policy::reference)
      .def("has_one_use", &Value::HasOneUse)
      .def("use_empty", &Value::use_empty)
      .def("__eq__", &Value::operator==)
      .def("__eq__",
           [](Value &self, OpResult &other) {
             return self.impl() == other.value_impl();
           })
      .def("__hash__",
           [](const Value &self) { return std::hash<ir::Value>{}(self); });
}

void BindOpOperand(py::module *m) {
  py::class_<OpOperand> op_operand(*m,
                                   "OpOperand",
                                   R"DOC(
    OpOperand class represents the op_operand (input) of operation.

    Notes:
        The constructor of OpOperand should not be invoked directly. OpOperand can be automatically constructed
        when build network.

  )DOC");
  op_operand
      .def("source",
           [](OpOperand &self) { return self.source().dyn_cast<OpResult>(); })
      .def("set_source",
           [](OpOperand &self, const OpResult &result) {
             self.set_source(result);
           })
      .def("owner", &OpOperand::owner, return_value_policy::reference);
}

bool GetStopGradient(const OpResult &self) {
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
}

void SetStopGradient(const OpResult &self, bool stop_gradient) {
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
  stop_gradients[self.GetResultIndex()] =
      ir::BoolAttribute::get(ir::IrContext::Instance(), stop_gradient);
  defining_op->set_attribute(
      kAttrStopGradients,
      ir::ArrayAttribute::get(ir::IrContext::Instance(), stop_gradients));
}

void BindOpResult(py::module *m) {
  py::class_<OpResult> op_result(*m, "OpResult", R"DOC(
    OpResult class represents the value(output) defined by a result of operation.

    Notes:
        The constructor of OpResult should not be invoked directly. OpResult can be automatically constructed
        when build network.
  )DOC");
  g_ir_opresult_pytype = reinterpret_cast<PyTypeObject *>(op_result.ptr());
  op_result.def("__eq__", &OpResult::operator==)
      .def("__eq__",
           [](OpResult &self, Value &other) {
             return self.value_impl() == other.impl();
           })
      .def("__hash__",
           [](OpResult &self) {
             return std::hash<ir::Value>{}(self.dyn_cast<ir::Value>());
           })
      .def("get_defining_op",
           &OpResult::GetDefiningOp,
           return_value_policy::reference)
      .def("first_use", &OpResult::first_use, return_value_policy::reference)
      .def("has_one_use", &Value::HasOneUse)
      .def("use_empty", &OpResult::use_empty)
      .def("type", &OpResult::type)
      .def_property(
          "stop_gradient",
          [](OpResult &self) { return GetStopGradient(self); },
          [](OpResult &self, bool stop_gradient) {
            SetStopGradient(self, stop_gradient);
          })
      .def_property(
          "shape",
          [](OpResult &self) {
            if (self.type().isa<DenseTensorType>()) {
              return phi::vectorize(
                  self.type().dyn_cast<DenseTensorType>().dims());
            } else {
              PADDLE_THROW(phi::errors::InvalidArgument(
                  "Currently, we can only get shape for dense tensor."));
            }
          },
          [](OpResult &self, const std::vector<int> &shape) {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "can't set shape when building static graph"));
          })
      .def_property(
          "dtype",
          [](OpResult &self) {
            if (self.type().isa<DenseTensorType>()) {
              return paddle::dialect::TransToPhiDataType(
                  self.type().dyn_cast<DenseTensorType>().dtype());
            } else {
              PADDLE_THROW(phi::errors::InvalidArgument(
                  "Currently, we can only get dtype for dense tensor."));
            }
          },
          [](OpResult &self, phi::DataType dtype) {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "can't set dtype when building static graph"));
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
  m->def("set_global_program",
         [](Program *program) { APIBuilder::Instance().SetProgram(program); });
  m->def("set_insertion_point",
         [](Operation *op) { APIBuilder::Instance().SetInsertionPoint(op); });
  m->def("reset_insertion_point_to_start",
         []() { APIBuilder::Instance().ResetInsertionPointToStart(); });
  m->def("reset_insertion_point_to_end",
         []() { APIBuilder::Instance().ResetInsertionPointToEnd(); });
  m->def("register_paddle_dialect", []() {
    ir::IrContext::Instance()
        ->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  });
  m->def(
      "translate_to_new_ir",
      [](const ::paddle::framework::ProgramDesc &legacy_program) {
        std::shared_ptr<Program> ret =
            std::move(paddle::TranslateLegacyProgramToProgram(legacy_program));
        return ret;
      },
      R"DOC(
        Convert Fluid Program to New IR Program.

        Args:

            legacy_program (ProgramDesc): The Fluid Program that will be converted.

        Returns:
            Program: The New IR Program

        Raises:
            PreconditionNotMet: If legacy_program has multi block will raise error.

        Examples:
            .. code-block:: python

                import paddle
                from paddle import ir
                paddle.enable_static()

                x = paddle.randn([4, 4])
                main_program, start_program = (
                    paddle.static.Program(),
                    paddle.static.Program(),
                )
                with paddle.static.program_guard(main_program, start_program):
                    x_s = paddle.static.data('x', [4, 4], x.dtype)
                    x_s.stop_gradient = False
                    y_s = paddle.matmul(x_s, x_s)
                    z_s = paddle.add(y_s, y_s)
                    k_s = paddle.tanh(z_s)
                newir_program = ir.translate_to_new_ir(main_program.desc)

                print(newir_program)

      )DOC");
  m->def(
      "check_unregistered_ops",
      [](const framework::ProgramDesc &legacy_program) {
        ir::IrContext *ctx = ir::IrContext::Instance();
        return paddle::translator::CheckUnregisteredOperation(ctx,
                                                              legacy_program);
      },
      R"DOC(
      Check unregistered operators in paddle dialect.

      Args:
        legacy_program (ProgramDesc): The Fluid Program that need checked.
      Returns:
        list[str] : List of unregistered operators in paddle dialect, the name is expressed by origin op name.
    )DOC");
}

void BindIrPass(pybind11::module *m) {
  py::class_<Pass, std::shared_ptr<Pass>> pass(*m,
                                               "Pass",
                                               R"DOC(
    Pass class.

  )DOC");
  pass.def("name", &Pass::name)
      .def("opt_level",
           [](const Pass &self) { return self.pass_info().opt_level; })
      .def("dependents",
           [](const Pass &self) { return self.pass_info().dependents; });
}

void BindPassManager(pybind11::module *m) {
  py::class_<PassManager, std::shared_ptr<PassManager>> pass_manager(
      *m,
      "PassManager",
      R"DOC(
    A class that manages all passes.

  )DOC");
  pass_manager
      .def(
          "__init__",
          [](PassManager &self, uint8_t opt_level) {
            new (&self) PassManager(ir::IrContext::Instance(), opt_level);
          },
          py::arg("opt_level") = 2)
      .def("add_pass",
           [](PassManager &self, const std::string &pass_name) {
             self.AddPass(
                 std::move(ir::PassRegistry::Instance().Get(pass_name)));
           })
      .def("passes",
           [](PassManager &self) {
             std::vector<std::string> pass_names;
             for (const auto &pass : self.passes()) {
               pass_names.emplace_back(pass->name());
             }
             return pass_names;
           })
      .def("run", [](PassManager &self, Program *p) { self.Run(p); })
      .def("empty", &PassManager::Empty);
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
  BindIrPass(&ir_module);
  BindPassManager(&ir_module);
  auto ops_modules = ir_module.def_submodule("ops");
  BindOpsAPI(&ops_modules);
}

}  // namespace pybind
}  // namespace paddle
