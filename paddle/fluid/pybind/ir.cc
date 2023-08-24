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
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/utils.h"
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
      .def("__repr__",
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
      .def("use_empty", &OpResult::use_empty)
      .def("type", &OpResult::type)
      .def("is_dense_tensor_type",
           [](OpResult &self) {
             if (self.type().isa<DenseTensorType>()) {
               return true;
             } else {
               return false;
             }
           })
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

Operation *BuildOpFrom(
    const Operation *to_copy_op,
    std::unordered_map<ir::Value, ir::Value> &value_map) {  // NOLINT
  ir::OperationArgument to_create_argument(to_copy_op->info());
  to_create_argument.attributes = to_copy_op->attributes();

  auto origin_results = to_copy_op->results();
  std::transform(origin_results.begin(),
                 origin_results.end(),
                 std::back_inserter(to_create_argument.output_types),
                 [](const ir::OpResult &r) {
                   // OpResult -> OpType
                   return r.type();
                 });

  // transform by value_map dict.
  auto origin_operands = to_copy_op->operands();
  std::transform(origin_operands.begin(),
                 origin_operands.end(),
                 std::back_inserter(to_create_argument.inputs),
                 [&value_map](const ir::OpOperand &operand) {
                   // Operand -> OpResult
                   return value_map[operand.source()].impl();
                 });
  auto *cloned_op = Operation::Create(std::move(to_create_argument));

  // update the mapping of value_map. std::transform is a map(func, zip()).
  std::vector<int> tmp;
  std::transform(origin_results.begin(),
                 origin_results.end(),
                 cloned_op->results().begin(),
                 std::back_inserter(tmp),  // NOLINT, just a placeholder.
                 [&value_map](const OpResult &a, const OpResult &b) {  // NOLINT
                   value_map[a.value_impl()] = b.value_impl();
                   return 1;
                 });
  return cloned_op;
}

std::shared_ptr<Program> ProgramClone(const Program &program) {
  // Limitation of this function:
  // 1. don't support Parameters.
  // 2. don't support Regions in operator.
  ir::IrContext *ctx = ir::IrContext::Instance();
  auto cloned_program = std::make_shared<Program>(ctx);
  std::unordered_map<ir::Value, ir::Value> value_map;
  for (auto &op : *program.block()) {
    auto *cloned_op = BuildOpFrom(op, value_map);
    cloned_program->block()->push_back(cloned_op);
  }
  return cloned_program;
}

std::list<Operation *>::const_iterator list_offset(const Block *block,
                                                   int start_idx) {
  auto it = block->begin();
  while (start_idx--) ++it;
  return it;
}

template <class F>
void range_block_do(const Block *block, std::vector<int> range, F fn) {
  for (auto it = list_offset(block, range[0]);
       it != list_offset(block, range[1]);
       ++it) {
    fn(*it);
  }
}

std::vector<ir::Value> AnalysisMiddleVariable(
    const Program &program,
    const std::vector<ir::Value> &forward_inputs,
    const std::vector<int> &forward_range,
    const std::vector<int> &backward_range) {
  std::vector<ir::Value> middle_values;

  std::unordered_set<ir::Value> backward_inputs;
  std::unordered_set<ir::Value> x_or_param(forward_inputs.begin(),
                                           forward_inputs.end());
  range_block_do(
      program.block(), backward_range, [&backward_inputs](Operation *op) {
        for (auto &t : op->operands()) {
          backward_inputs.insert(t.source());
        }
      });

  range_block_do(
      program.block(),
      forward_range,
      [&middle_values, &backward_inputs, &x_or_param](Operation *op) {
        for (auto &t : op->results()) {
          auto v = Value(t.value_impl());
          if (backward_inputs.count(v) && !x_or_param.count(v))
            middle_values.push_back(v);
        }
      });
  return middle_values;
}

void mapping_value(const std::vector<ir::Value> &origin,
                   const std::unordered_map<ir::Value, ir::Value> &value_map,
                   std::vector<ir::Value> &out) {  // NOLINT
  std::transform(origin.begin(),
                 origin.end(),
                 std::back_inserter(out),
                 [&value_map](const ir::Value &v) { return value_map.at(v); });
}

using SplitedProgram = std::vector<std::shared_ptr<Program>>;
using SplitedAttribute = std::map<std::string, std::vector<ir::Value>>;
using SplitedResult = std::pair<SplitedProgram, SplitedAttribute>;

SplitedResult ForwardBackwardSplit(
    const Program &program,
    const std::vector<ir::OpResult> &op_result_forward_inputs,
    const std::vector<ir::OpResult> &op_result_forward_outputs,
    const std::vector<ir::OpResult> &op_result_forward_inputs_grads,
    const std::vector<ir::OpResult> &op_result_forward_outputs_grads,
    const std::vector<int> &forward_range,
    const std::vector<int> &backward_range) {
  // transform opresult -> value
  VLOG(1) << "Start Prepare data structures.";
  std::vector<ir::Value> forward_inputs, forward_outputs, forward_inputs_grads,
      forward_outputs_grads;
  auto op_result_to_value = [](const ir::OpResult &r) {
    return Value(r.value_impl());
  };
  std::transform(op_result_forward_inputs.begin(),
                 op_result_forward_inputs.end(),
                 std::back_inserter(forward_inputs),
                 op_result_to_value);
  std::transform(op_result_forward_outputs.begin(),
                 op_result_forward_outputs.end(),
                 std::back_inserter(forward_outputs),
                 op_result_to_value);
  std::transform(op_result_forward_inputs_grads.begin(),
                 op_result_forward_inputs_grads.end(),
                 std::back_inserter(forward_inputs_grads),
                 op_result_to_value);
  std::transform(op_result_forward_outputs_grads.begin(),
                 op_result_forward_outputs_grads.end(),
                 std::back_inserter(forward_outputs_grads),
                 op_result_to_value);

  std::vector<ir::Value> fx, fp, fm, fo, bx, bp, bm, bo_g, bx_g, bp_g, bo;
  ir::IrContext *ctx = ir::IrContext::Instance();
  auto forward_program = std::make_shared<Program>(ctx);
  auto backward_program = std::make_shared<Program>(ctx);
  auto middle_values = AnalysisMiddleVariable(
      program, forward_inputs, forward_range, backward_range);
  std::unordered_map<ir::Value, ir::Value> forward_value_map;
  std::unordered_map<ir::Value, ir::Value> backward_value_map;
  ir::Builder backward_builder = ir::Builder(ctx, backward_program->block());

  // forward program construct.
  VLOG(1) << "Before Forward Construct.";
  range_block_do(program.block(),
                 forward_range,
                 [&forward_value_map, &forward_program](Operation *op) {
                   auto *cloned_op = BuildOpFrom(op, forward_value_map);
                   forward_program->block()->push_back(cloned_op);
                 });
  VLOG(1) << "After Forward Construct.";

  // backward program construc.
  // Step1. insert data op for inputs_values and middle_values
  int counter = 0;
  auto create_data_fn = [&backward_builder, &backward_value_map, &counter](
                            const ir::Value &v) {
    auto value_type = v.type().dyn_cast<DenseTensorType>();
    auto dtype = paddle::dialect::TransToPhiDataType(value_type.dtype());
    auto shape = phi::vectorize(value_type.dims());
    auto place = phi::CPUPlace();  // TODO(xiongkun): how to get default places.

    paddle::dialect::DataOp op =
        backward_builder.Build<paddle::dialect::DataOp>(
            std::string("input_") + std::to_string(counter),
            shape,
            dtype,
            place);
    counter += 1;
    backward_value_map[v] = op->results()[0].value_impl();
  };
  std::for_each(forward_inputs.begin(), forward_inputs.end(), create_data_fn);
  std::for_each(forward_outputs.begin(), forward_outputs.end(), create_data_fn);
  std::for_each(middle_values.begin(), middle_values.end(), create_data_fn);
  std::for_each(forward_outputs_grads.begin(),
                forward_outputs_grads.end(),
                create_data_fn);
  VLOG(1) << "After call create_data_fn";

  // Step2. copy backward ops .
  range_block_do(program.block(),
                 backward_range,
                 [&backward_value_map, &backward_program](Operation *op) {
                   auto *cloned_op = BuildOpFrom(op, backward_value_map);
                   backward_program->block()->push_back(cloned_op);
                 });
  VLOG(1) << "After call backward copy";

  VLOG(1) << "forward_value_map.size() is " << forward_value_map.size();
  VLOG(1) << "backward_value_map.size() is " << backward_value_map.size();

  std::ostringstream print_stream;
  print_stream << "ForwardProgram is :\n";
  forward_program->Print(print_stream);
  print_stream << "BackwardProgram is:\n";
  backward_program->Print(print_stream);
  VLOG(1) << "Splited Program (fwd | bwd): \n" << print_stream.str();

  // construct all attributes we needed.
  mapping_value(middle_values, forward_value_map, fm);  // write 'fm'
  VLOG(1) << "XKXKXK";
  mapping_value(middle_values, backward_value_map, bm);  // write 'bm'
  VLOG(1) << "XKXKXK";
  mapping_value(forward_inputs, forward_value_map, fx);  // write 'bm'
  VLOG(1) << "XKXKXK";
  mapping_value(forward_inputs, backward_value_map, bx);  // write 'bm'
  VLOG(1) << "XKXKXK";
  mapping_value(forward_outputs, forward_value_map, fo);  // write 'bm'
  VLOG(1) << "XKXKXK";
  mapping_value(forward_inputs_grads, backward_value_map, bx_g);  // write 'bm'
  VLOG(1) << "XKXKXK";
  mapping_value(forward_outputs_grads, backward_value_map, bo_g);  // write 'bm'
  VLOG(1) << "XKXKXK";
  mapping_value(forward_outputs, backward_value_map, bo);  // write 'bm'
  VLOG(1) << "After mapping values.";

  std::map<std::string, std::vector<ir::Value>> attr = {{"fx", fx},
                                                        {"fp", fp},
                                                        {"fm", fm},
                                                        {"fo", fo},
                                                        {"bx", bx},
                                                        {"bp", bp},
                                                        {"bm", bm},
                                                        {"bo_g", bo_g},
                                                        {"bx_g", bx_g},
                                                        {"bp_g", bp_g},
                                                        {"bo", bo}};
  std::vector<std::shared_ptr<Program>> programs = {forward_program,
                                                    backward_program};
  return std::make_pair(programs, attr);
}

void BindUtils(pybind11::module *m) {
  m->def("program_clone", ProgramClone);
  m->def("program_split", ForwardBackwardSplit);
  m->def("set_global_program",
         [](Program *program) { APIBuilder::Instance().SetProgram(program); });
  m->def("set_insertion_point",
         [](Operation *op) { APIBuilder::Instance().SetInsertionPoint(op); });
  m->def("reset_insertion_point_to_start",
         []() { APIBuilder::Instance().ResetInsertionPointToStart(); });
  m->def("reset_insertion_point_to_end",
         []() { APIBuilder::Instance().ResetInsertionPointToEnd(); });
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
