// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/control_flow_api.h"

#include <Python.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <unordered_set>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

#include "paddle/fluid/pybind/python_callable_registry.h"

namespace py = pybind11;
using paddle::dialect::ApiBuilder;
using paddle::dialect::AssertOp;
using paddle::dialect::HasElementsOp;
using paddle::dialect::IfOp;
using paddle::dialect::PyLayerOp;
using paddle::dialect::WhileOp;
using paddle::pybind::PyIfOp;
using paddle::pybind::PyWhileOp;
using pir::Block;
using pir::Builder;
using pir::CombineOp;
using pir::Operation;
using pir::Program;
using pir::StackCreateOp;
using pir::TuplePopOp;
using pir::TuplePushOp;
using pir::Type;
using pir::Value;
using pir::YieldOp;
using pybind11::return_value_policy;
namespace {

void BindIfOp(py::module* m) {
  m->def("build_if_op", [](Value cond) {
    return PyIfOp(ApiBuilder::Instance().GetBuilder()->Build<IfOp>(
        cond, std::vector<Type>{}));
  });
  m->def("build_if_op", [](const std::vector<Value>& cond) {
    auto& builder = ApiBuilder::Instance().GetBuilder();
    auto new_cond = builder->Build<CombineOp>(cond).out();
    return PyIfOp(builder->Build<IfOp>(new_cond, std::vector<Type>{}));
  });
  py::class_<PyIfOp> if_op(*m, "IfOp", R"DOC(
    The PyIfOp is a encapsulation of IfOp. Compared with ifOp, it provides an additional 'update_output' interface.
    The 'update_output' interface will construct a new IfOp operation to replace its underlying IfOp. In the process, the original
    IfOp will be destroyed. In order to avoid the risk of memory used in python side, We encapsulate PyIfOp to python api.
  )DOC");
  if_op.def("true_block", &PyIfOp::true_block, return_value_policy::reference)
      .def("false_block", &PyIfOp::false_block, return_value_policy::reference)
      .def("cond", &PyIfOp::cond)
      .def("update_output", &PyIfOp::UpdateOutput)
      .def("as_operation", &PyIfOp::operation, return_value_policy::reference)
      .def("results", [](PyIfOp& self) -> py::list {
        py::list op_list;
        for (uint32_t i = 0; i < self->num_results(); i++) {
          op_list.append(static_cast<pir::Value>(self.result(i)));
        }
        return op_list;
      });
}

void BindPyLayerOp(py::module* m) {
  m->def("build_pylayer_op", [](const std::vector<Value>& inputs) {
    return ApiBuilder::Instance().GetBuilder()->Build<PyLayerOp>(
        inputs, std::vector<Type>{}, -1);
  });
  py::class_<PyLayerOp> pylayer_op(*m, "PyLayerOp", R"DOC(
    TODO(MarioLulab): Add some docs for pd_op.pylayer
  )DOC");
  pylayer_op
      .def("forward_block",
           &PyLayerOp::forward_block,
           return_value_policy::reference)
      .def("update_output", &PyLayerOp::UpdateOutput)
      .def("update_input", &PyLayerOp::UpdateInput)
      .def(
          "as_operation", &PyLayerOp::operation, return_value_policy::reference)
      .def("id",
           [](PyLayerOp& self) -> uint64_t { return self.operation()->id(); })
      .def("results",
           [](PyLayerOp& self) -> py::list {
             py::list op_list;
             for (uint32_t i = 0; i < self->num_results(); i++) {
               op_list.append(self.result(i));
             }
             return op_list;
           })
      .def("register_backward_function", [](PyLayerOp& self, py::object func) {
        uint64_t unique_id = self.operation()->id();
        VLOG(2) << "register backward function for op id: " << unique_id;
        paddle::pybind::PythonCallableRegistrar::GetInstance().Register(
            unique_id, func);
        self.operation()->set_attribute(
            "backward_function_id",
            pir::Int32Attribute::get(pir::IrContext::Instance(), unique_id));
      });
}

void BindWhileOp(py::module* m) {
  m->def("build_while_op", [](Value cond, py::list loop_vars) -> PyWhileOp {
    std::vector<Value> loop_values;
    for (auto var : loop_vars) {
      loop_values.push_back(var.cast<Value>());
    }
    return PyWhileOp(
        ApiBuilder::Instance().GetBuilder()->Build<WhileOp>(cond, loop_values));
  });
  py::class_<PyWhileOp> while_op(*m, "WhileOp", R"DOC(
    WhileOp in python api.
  )DOC");
  while_op.def("body", &PyWhileOp::body, return_value_policy::reference)
      .def(
          "as_operation", &PyWhileOp::operation, return_value_policy::reference)
      .def("block_arguments",
           &WhileOp::block_args,
           return_value_policy::reference)
      .def("optimize_update", &PyWhileOp::OptimizeUpdate)
      .def("add_extra_input", &PyWhileOp::AddExtraInput);
}

void BindAssertOp(py::module* m) {
  m->def("build_assert_op",
         [](Value cond, const std::vector<Value>& data, int64_t summarize) {
           auto data_combine_op =
               ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(data);
           return ApiBuilder::Instance().GetBuilder()->Build<AssertOp>(
               cond, data_combine_op.out(), summarize);
         });
  py::class_<AssertOp> assert_op(*m, "AssertOp", R"DOC(
    AssertOp in python api.
  )DOC");
  assert_op.def(
      "as_operation", &AssertOp::operation, return_value_policy::reference);
}

void BindTuplePopOp(py::module* m) {
  py::class_<TuplePopOp> tuple_pop_op(*m, "TuplePopOp", R"DOC(
    TuplePopOp in python api.
  )DOC");
  tuple_pop_op
      .def("as_operation",
           &TuplePopOp::operation,
           return_value_policy::reference)
      .def("pop_all_values",
           [](TuplePopOp& self) -> py::list {
             py::list res;
             for (size_t i = 0; i < self.num_results(); ++i) {
               res.append(self.result(i));
             }
             return res;
           })
      .def(
          "tuple_size", &TuplePopOp::tuple_size, return_value_policy::reference)
      .def("outlet_element",
           &TuplePopOp::outlet_element,
           return_value_policy::reference);
}

void BuildPipeForPyLayer(Block* block, const std::vector<pir::Value>& values) {
  PADDLE_ENFORCE_NOT_NULL(
      block,
      common::errors::InvalidArgument(
          "The block used to hook local value can't be nullptr"));
  auto& builder = *(ApiBuilder::Instance().GetBuilder());
  Program* program = block->parent_program();
  PADDLE_ENFORCE_NOT_NULL(
      program,
      common::errors::InvalidArgument(
          "The block used to hook local value must belong to a program"));

  auto original_position = builder.insertion_point();

  builder.SetInsertionPointToStart(program->block());
  auto inlet = builder.Build<StackCreateOp>().inlet();
  auto iter = block->end();
  if (!block->empty() && block->back().isa<YieldOp>()) {
    --iter;
  }
  builder.set_insertion_point(block, iter);
  builder.Build<TuplePushOp>(inlet, values);
  builder.set_insertion_point(original_position);
}

Value BuildHasElementsOp(Operation& fwd_op) {  // NOLINT
  PADDLE_ENFORCE(fwd_op.isa<WhileOp>(),
                 common::errors::PreconditionNotMet(
                     "param op of BuildHasElementsOp must be while op."));
  auto fwdop = fwd_op.dyn_cast<WhileOp>();
  TuplePushOp push_op;
  for (auto iter = fwdop.body().rbegin(); iter != fwdop.body().rend(); ++iter) {
    if (iter->isa<TuplePushOp>()) {
      push_op = iter->dyn_cast<TuplePushOp>();
      PADDLE_ENFORCE_EQ(push_op.container().use_empty(),
                        false,
                        common::errors::InvalidArgument(
                            "The last container in forward while op must used "
                            "after construct while_grad op"));
      break;
    }
  }
  auto new_cond = ApiBuilder::Instance()
                      .GetBuilder()
                      ->Build<HasElementsOp>(push_op.container())
                      .out();
  return new_cond;
}

void BuildPipeForBlock(Block* block) {
  PADDLE_ENFORCE_NOT_NULL(
      block,
      common::errors::InvalidArgument(
          "The block used to hook local value can't be nullptr"));
  auto& builder = *(ApiBuilder::Instance().GetBuilder());
  Program* program = block->parent_program();
  PADDLE_ENFORCE_NOT_NULL(
      program,
      common::errors::InvalidArgument(
          "The block used to hook local value must belong to a program"));

  auto original_position = builder.insertion_point();

  builder.SetInsertionPointToStart(program->block());
  auto inlet = builder.Build<StackCreateOp>().inlet();
  auto iter = block->end();
  if (!block->empty() && block->back().isa<YieldOp>()) {
    --iter;
  }
  std::vector<Value> local_values;
  for (auto arg_value : block->args()) {
    local_values.push_back(arg_value);
  }
  for (auto& op : *block) {
    for (auto result_value : op.results()) {
      local_values.push_back(result_value);
    }
  }
  builder.set_insertion_point(block, iter);
  builder.Build<TuplePushOp>(inlet, local_values);
  builder.set_insertion_point(original_position);
}

}  // namespace

namespace paddle::pybind {
PyIfOp::PyIfOp(IfOp if_op) : IfOp(if_op) {
  PADDLE_ENFORCE_NOT_NULL(
      if_op,
      common::errors::InvalidArgument(
          "The if_op used to construct PyIfOp can't be nullptr"));
}

void PyIfOp::UpdateOutput() {
  PADDLE_ENFORCE_NOT_NULL(
      operation_,
      common::errors::InvalidArgument(
          "The if_op in PyIfOp used to update output can't be nullptr"));
  auto block = parent();
  PADDLE_ENFORCE_NOT_NULL(block,
                          common::errors::InvalidArgument(
                              "The parent block of if_op which used to update "
                              "output can't be nullptr"));
  Block::Iterator iter = **this;
  Builder builder(ir_context(), false);
  auto new_if_op = builder.Build<IfOp>(
      cond(), true_region().TakeBack(), false_region().TakeBack());
  block->Assign(iter, new_if_op);
  IfOp::operator=(new_if_op);
  operation_->Verify();
}

PyWhileOp::PyWhileOp(WhileOp while_op) : WhileOp(while_op) {
  PADDLE_ENFORCE_NOT_NULL(
      operation_,
      common::errors::InvalidArgument(
          "The while_op used to construct PyWhileOp can't be nullptr"));
}

void PyWhileOp::AddExtraInput(const pir::Value& value) {
  extra_inputs_.push_back(value);
}

std::vector<Value> PyWhileOp::OptimizeUpdate() {
  PADDLE_ENFORCE_NOT_NULL(operation_,
                          common::errors::InvalidArgument(
                              "The while_op in PyWhileOp used to remove unused "
                              "loop vars can't be nullptr"));
  auto parent_block = parent();
  PADDLE_ENFORCE_NOT_NULL(
      parent_block,
      common::errors::InvalidArgument(
          "The parent block of while_op which used to remove "
          "unused loop vars can't be nullptr"));

  // Skip verify if operation has extra inputs
  if (extra_inputs_.empty()) {
    operation_->Verify();
  }
  auto& body_block = body();
  auto yield_op = body_block.back().dyn_cast<YieldOp>();
  auto operand_num = operation_->num_operands();

  PADDLE_ENFORCE_EQ(
      operand_num - 1 + extra_inputs_.size(),
      body_block.args().size(),
      common::errors::InvalidArgument("The number of operands in while_op and "
                                      "the number of args in body block "
                                      "should be equal."));
  bool no_change = extra_inputs_.empty();
  std::vector<size_t> index_vec;
  std::vector<Value> res, new_input, new_yield_val{yield_op.operand_source(0)};
  for (uint32_t i = 0; i < num_results(); ++i) {
    res.push_back(result(i));
  }
  for (size_t operand_index = 1u, arg_index = 0u; operand_index < operand_num;
       ++operand_index, ++arg_index) {
    if (!body_block.arg(arg_index).type().isa<pir::DenseTensorType>()) {
      continue;
    }

    auto l_type =
        body_block.arg(arg_index).type().dyn_cast<pir::DenseTensorType>();
    auto r_type = yield_op.operand_source(operand_index)
                      .type()
                      .dyn_cast<pir::DenseTensorType>();
    if (l_type.dims().size() == r_type.dims().size() &&
        l_type.dims() != r_type.dims()) {
      VLOG(4) << "while op input " << operand_index
              << " has dynamic shape, origin shape is: " << l_type.dims()
              << "new shape is: " << r_type.dims();
      auto dim = common::ComputeCompatibleDim(l_type.dims(), r_type.dims());
      auto new_type = pir::DenseTensorType::get(operation_->ir_context(),
                                                l_type.dtype(),
                                                dim,
                                                l_type.data_layout(),
                                                l_type.lod(),
                                                l_type.offset());
      body_block.arg(arg_index).set_type(new_type);
      yield_op.operand_source(operand_index).set_type(new_type);
      result(arg_index).set_type(new_type);
      VLOG(4) << "change shape as: " << new_type.dims();
    }
  }

  for (size_t operand_index = 1u, arg_index = 0u; operand_index < operand_num;
       ++operand_index) {
    operand_source(operand_index).set_type(body_block.arg(arg_index).type());
    if (yield_op.operand_source(operand_index) == body_block.arg(arg_index)) {
      body_block.arg(arg_index).ReplaceAllUsesWith(
          operand_source(operand_index));
      body_block.EraseArg(arg_index);
      no_change = false;
      res[operand_index - 1u] = operand_source(operand_index);
    } else {
      new_input.push_back(operand_source(operand_index));
      index_vec.push_back(operand_index - 1u);
      new_yield_val.push_back(yield_op.operand_source(operand_index));
      ++arg_index;
    }
  }
  for (size_t extra_input_idx = 0u; extra_input_idx < extra_inputs_.size();
       ++extra_input_idx) {
    new_input.push_back(extra_inputs_[extra_input_idx]);
    new_yield_val.push_back(
        yield_op.operand_source(operand_num + extra_input_idx));
  }
  if (no_change) return res;
  Block::Iterator iter = **this;
  Builder builder(ir_context(), false);
  auto new_while_op = builder.Build<WhileOp>(cond(), new_input, false);
  new_while_op->region(0).swap(std::move(operation_->region(0)));
  parent_block->Assign(iter, new_while_op);
  WhileOp::operator=(new_while_op);
  body_block.pop_back();
  builder.SetInsertionPointToBlockEnd(&body_block);
  builder.Build<YieldOp>(new_yield_val);
  operation_->Verify();
  for (size_t result_index = 0;
       result_index < num_results() - extra_inputs_.size();
       ++result_index) {
    res[index_vec[result_index]] = result(result_index);
  }
  for (size_t i = 0; i < extra_inputs_.size(); ++i) {
    res.push_back(result(num_results() - extra_inputs_.size() + i));
  }
  return res;
}

void BindControlFlowApi(py::module* m) {
  m->def("get_used_external_value",
         [](const Operation& op) { return pir::GetUsedExternalValue(op); });
  m->def("get_used_external_value",
         [](const Block& block) { return pir::GetUsedExternalValue(block); });
  m->def("build_pipe_for_block", BuildPipeForBlock);
  m->def("build_pipe_for_pylayer", BuildPipeForPyLayer);
  m->def("cf_has_elements", BuildHasElementsOp);
  m->def("cf_yield", [](py::list inputs) {
    std::vector<Value> input_values;
    for (auto input : inputs) {
      input_values.push_back(input.cast<Value>());
    }
    ApiBuilder::Instance().GetBuilder()->Build<YieldOp>(input_values);
  });
  BindIfOp(m);
  BindWhileOp(m);
  BindAssertOp(m);
  BindPyLayerOp(m);
  BindTuplePopOp(m);
}

}  // namespace paddle::pybind
