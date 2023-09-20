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
#include "paddle/pir/core/builtin_op.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/inplace_pass.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/type.h"
#include "paddle/pir/core/value.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/transforms/dead_code_elimination_pass.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using paddle::dialect::APIBuilder;
using paddle::dialect::DenseTensorType;
using pir::Block;
using pir::Operation;
using pir::OpOperand;
using pir::OpResult;
using pir::Pass;
using pir::PassManager;
using pir::Program;
using pir::Type;
using pir::Value;
using pybind11::return_value_policy;

USE_PASS(dead_code_elimination);
USE_PASS(inplace);

namespace paddle {
namespace pybind {

PyTypeObject *g_ir_opresult_pytype = nullptr;

void BindOpsAPI(pybind11::module *module);

inline int64_t GetProgramInt64Attr(const std::shared_ptr<Program> &program,
                                   const std::string &attr_name,
                                   int64_t default_value = 0) {
  auto op = program->module_op();
  if (op->HasAttribute(attr_name)) {
    auto val = op->attribute(attr_name).dyn_cast<pir::Int64Attribute>().data();
    return val;
  } else {
    return default_value;
  }
}

inline void SetProgramInt64Attr(std::shared_ptr<Program> program,
                                const std::string &attr_name,
                                int64_t value) {
  auto op = program->module_op();
  op->set_attribute(
      attr_name, pir::Int64Attribute::get(pir::IrContext::Instance(), value));
}

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
      .def("__init__",
           [](Program &self) {
             new (&self) Program(pir::IrContext::Instance());
           })
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
          "global_block",
          [](std::shared_ptr<Program> self) { return self->block(); },
          return_value_policy::reference)
      .def(
          "global_block",
          [](const std::shared_ptr<Program> &self) { return self->block(); },
          return_value_policy::reference)
      .def_property(
          "random_seed",
          [](const std::shared_ptr<Program> &self) {
            return GetProgramInt64Attr(self, "random_seed", 0);
          },
          [](std::shared_ptr<Program> self, int64_t random_seed) {
            SetProgramInt64Attr(self, "random_seed", random_seed);
          });
}

void BindBlock(py::module *m) {
  py::class_<Block> block(*m, "Block", R"DOC(
    In IR, a Block has a list of Operation and can represent a sub computational graph.

    Notes:
        The constructor of Block should not be invoked directly. You can
        use `Program.block()` to get a block.
  )DOC");
  block.def("front", &Block::front, return_value_policy::reference)
      .def_property_readonly(
          "program",
          [](Block &self) { return self.GetParentOp()->GetParentProgram(); },
          return_value_policy::reference)
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

      )DOC")
      .def("all_parameters", [](Block &self) -> py::list {
        py::list param_list;
        for (auto iter = self.begin(); iter != self.end(); iter++) {
          auto op = *iter;
          if (op->HasAttribute(kAttrIsPersisable)) {
            auto attrs = op->attribute(kAttrIsPersisable)
                             .dyn_cast<pir::ArrayAttribute>()
                             .AsVector();
            for (uint32_t i = 0; i < attrs.size(); i++) {
              bool is_persistable =
                  attrs[i].dyn_cast<pir::BoolAttribute>().data();
              if (is_persistable) {
                param_list.append(op->result(i));
              }
            }
          }
        }
        return param_list;
      });
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
           &Operation::GetParent,
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
      .def(
          "get_defining_op",
          [](const Value &self) -> pir::Operation * {
            if (auto op_result = self.dyn_cast<pir::OpResult>()) {
              return op_result.owner();
            }
            return nullptr;
          },
          return_value_policy::reference)
      .def("first_use", &Value::first_use, return_value_policy::reference)
      .def("has_one_use", &Value::HasOneUse)
      .def("use_empty", &Value::use_empty)
      .def("__eq__", &Value::operator==)
      .def("__eq__",
           [](Value &self, OpResult &other) {
             return self.impl() == other.Value::impl();
           })
      .def("__hash__",
           [](const Value &self) { return std::hash<pir::Value>{}(self); });
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

bool GetOpResultBoolAttr(const OpResult &self, const std::string &attr_name) {
  auto *defining_op = self.owner();
  if (defining_op->HasAttribute(attr_name)) {
    auto attrs = defining_op->attribute(attr_name)
                     .dyn_cast<pir::ArrayAttribute>()
                     .AsVector();
    return attrs[self.index()].dyn_cast<pir::BoolAttribute>().data();
  } else {
    return true;
  }
}

void SetOpResultBoolAttr(const OpResult &self,
                         const std::string &attr_name,
                         bool value,
                         bool default_value) {
  auto *defining_op = self.owner();
  std::vector<pir::Attribute> attrs;
  if (defining_op->HasAttribute(attr_name)) {
    attrs = defining_op->attribute(attr_name)
                .dyn_cast<pir::ArrayAttribute>()
                .AsVector();
  } else {
    attrs = std::vector<pir::Attribute>(
        defining_op->num_results(),
        pir::BoolAttribute::get(pir::IrContext::Instance(), default_value));
  }
  attrs[self.index()] =
      pir::BoolAttribute::get(pir::IrContext::Instance(), value);
  defining_op->set_attribute(
      attr_name, pir::ArrayAttribute::get(pir::IrContext::Instance(), attrs));
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
             return self.Value::impl() == other.impl();
           })
      .def("__neg__",
           [](OpResult &self) {
             return paddle::dialect::scale(self, -1.0, 0.0, true);
           })
      .def("__add__",
           [](OpResult &self, OpResult &other) {
             return paddle::dialect::add(self, other);
           })
      .def("__sub__",
           [](OpResult &self, OpResult &other) {
             return paddle::dialect::subtract(self, other);
           })
      .def("__mul__",
           [](OpResult &self, OpResult &other) {
             return paddle::dialect::multiply(self, other);
           })
      .def("__truediv__",
           [](OpResult &self, OpResult &other) {
             return paddle::dialect::divide(self, other);
           })
      .def("__lt__",
           [](OpResult &self, OpResult &other) {
             return paddle::dialect::less_than(self, other);
           })
      .def("__le__",
           [](OpResult &self, OpResult &other) {
             return paddle::dialect::less_equal(self, other);
           })
      .def("__gt__",
           [](OpResult &self, OpResult &other) {
             return paddle::dialect::greater_than(self, other);
           })
      .def("__ge__",
           [](OpResult &self, OpResult &other) {
             return paddle::dialect::greater_equal(self, other);
           })
      .def("__hash__",
           [](OpResult &self) { return std::hash<pir::Value>{}(self); })
      .def(
          "get_defining_op",
          [](const OpResult &self) -> pir::Operation * {
            return self ? self.owner() : nullptr;
          },
          return_value_policy::reference)
      .def_property_readonly(
          "block",
          [](OpResult &self) { return self.owner()->GetParent(); },
          return_value_policy::reference)
      .def_property_readonly(
          "name",
          [](OpResult &self) {
            if (self.owner()->isa<::pir::GetParameterOp>()) {
              auto param_name =
                  self.owner()
                      ->attribute<pir::StrAttribute>("parameter_name")
                      .AsString();
              return param_name;
            } else {
              PADDLE_THROW(phi::errors::InvalidArgument(
                  "Currently, we can only get name of OpResult that is "
                  "persistable"));
            }
          })
      .def("first_use", &OpResult::first_use, return_value_policy::reference)
      .def("has_one_use", &Value::HasOneUse)
      .def("use_empty", &OpResult::use_empty)
      .def("type", &OpResult::type)
      .def_property(
          "stop_gradient",
          [](OpResult &self) {
            return GetOpResultBoolAttr(self, kAttrStopGradients);
          },
          [](OpResult &self, bool stop_gradient) {
            // NOTE(Aurelius84): For other OpResult, set theirs stop_gradient
            // default value as true.
            SetOpResultBoolAttr(self,
                                kAttrStopGradients,
                                stop_gradient,
                                /*default_value=*/true);
          })
      .def_property(
          "is_persistable",
          [](OpResult &self) {
            return GetOpResultBoolAttr(self, kAttrIsPersisable);
          },
          [](OpResult &self, bool is_persistable) {
            // NOTE(Aurelius84): For other OpResult, set theirs is_persistable
            // default value as false.
            SetOpResultBoolAttr(self,
                                kAttrIsPersisable,
                                is_persistable,
                                /*default_value=*/false);
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
    pir::IrContext::Instance()
        ->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
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
        pir::IrContext *ctx = pir::IrContext::Instance();
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
            new (&self) PassManager(pir::IrContext::Instance(), opt_level);
          },
          py::arg("opt_level") = 2)
      .def("add_pass",
           [](PassManager &self, const std::string &pass_name) {
             self.AddPass(
                 std::move(pir::PassRegistry::Instance().Get(pass_name)));
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
