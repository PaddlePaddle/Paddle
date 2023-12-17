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

#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/op_result.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"

namespace py = pybind11;
using paddle::dialect::ApiBuilder;
using paddle::dialect::IfOp;
using paddle::dialect::WhileOp;
using pir::Block;
using pir::Builder;
using pir::Operation;
using pir::Program;
using pir::Region;
using pir::StackCreateOp;
using pir::TuplePushOp;
using pir::Type;
using pir::Value;
using pir::YieldOp;
using pybind11::return_value_policy;

using paddle::pybind::PyIfOp;
namespace {

void BindIfOp(py::module* m) {
  m->def("build_if_op", [](Value cond) {
    return PyIfOp(ApiBuilder::Instance().GetBuilder()->Build<IfOp>(
        cond, std::vector<Type>{}));
  });
  py::class_<PyIfOp> if_op(*m, "IfOp", R"DOC(
    The PyIfOp is a encapsulation of IfOp. Compared with ifOp, it provides an additional 'update_output' interface.
    The 'update_output' interface will construct a new IfOp operation to replace its underlying IfOp. In the process, the original
    IfOp will be destroyed. In order to avoid the risk of memory used in python side, We encapsulate PyIfOp to python api.
  )DOC");
  if_op.def("true_block", &PyIfOp::true_block, return_value_policy::reference)
      .def("false_block", &PyIfOp::false_block, return_value_policy::reference)
      .def("update_output", &PyIfOp::UpdateOutput)
      .def("as_operation", &PyIfOp::operation, return_value_policy::reference)
      .def("results", [](PyIfOp& self) -> py::list {
        py::list op_list;
        for (uint32_t i = 0; i < self->num_results(); i++) {
          op_list.append(self.result(i));
        }
        return op_list;
      });
}

void BindWhileOp(py::module* m) {
  m->def("build_while_op", [](Value cond, py::list loop_vars) {
    std::vector<Value> loop_values;
    for (auto var : loop_vars) {
      loop_values.push_back(var.cast<Value>());
    }
    return ApiBuilder::Instance().GetBuilder()->Build<WhileOp>(cond,
                                                               loop_values);
  });
  py::class_<WhileOp> while_op(*m, "WhileOp", R"DOC(
    WhileOp in python api.
  )DOC");
  while_op.def("body", &WhileOp::body, return_value_policy::reference)
      .def("as_operation", &WhileOp::operation, return_value_policy::reference);
}

void GetUsedExternalValueImpl(
    std::unordered_set<Value>& defined_values,  // NOLINT
    std::vector<Value>& used_values,            // NOLINT
    const Operation& op) {
  for (size_t index = 0; index < op.num_operands(); ++index) {
    Value value = op.operand_source(index);
    if (defined_values.find(value) == defined_values.end()) {
      used_values.push_back(value);
      defined_values.insert(value);
    }
  }
  for (auto& region : op) {
    for (auto& block : region) {
      for (auto value : block.args()) {
        defined_values.insert(value);
      }
    }
    for (auto& block : region) {
      for (auto& inner_op : block) {
        GetUsedExternalValueImpl(defined_values, used_values, inner_op);
      }
    }
  }
  for (size_t index = 0; index < op.num_results(); ++index) {
    defined_values.insert(op.result(index));
  }
}

std::vector<Value> GetUsedExternalValue(const Operation& op) {
  std::unordered_set<Value> defined_values{nullptr};
  std::vector<Value> used_values;
  GetUsedExternalValueImpl(defined_values, used_values, op);
  return used_values;
}

void BuildPipeForBlock(Block* block) {
  PADDLE_ENFORCE_NOT_NULL(
      block,
      paddle::platform::errors::InvalidArgument(
          "The block used to hook local value can't be nullptr"));
  auto& builder = *(ApiBuilder::Instance().GetBuilder());
  Program* program = block->parent_program();
  PADDLE_ENFORCE_NOT_NULL(
      program,
      paddle::platform::errors::InvalidArgument(
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

namespace paddle {
namespace pybind {
PyIfOp::PyIfOp(IfOp if_op) : IfOp(if_op) {
  PADDLE_ENFORCE_NOT_NULL(
      if_op,
      paddle::platform::errors::InvalidArgument(
          "The if_op used to construct PyIfOp can't be nullptr"));
}

void PyIfOp::UpdateOutput() {
  PADDLE_ENFORCE_NOT_NULL(
      *this,
      paddle::platform::errors::InvalidArgument(
          "The if_op in PyIfOp used to update output can't be nullptr"));
  auto block = parent();
  PADDLE_ENFORCE_NOT_NULL(block,
                          paddle::platform::errors::InvalidArgument(
                              "The parent block of if_op which used to update "
                              "output can't be nullptr"));
  Block::Iterator iter = **this;
  Builder builder(ir_context(), false);
  auto new_if_op = builder.Build<IfOp>(
      cond(), true_region().TakeBack(), false_region().TakeBack());
  block->Assign(iter, new_if_op);
  IfOp::operator=(new_if_op);
  VerifyRegion();
}

void BindControlFlowApi(py::module* m) {
  m->def("get_used_external_value", GetUsedExternalValue);
  m->def("build_pipe_for_block", BuildPipeForBlock);
  m->def("cf_yield", [](py::list inputs) {
    std::vector<Value> input_values;
    for (auto input : inputs) {
      input_values.push_back(input.cast<Value>());
    }
    ApiBuilder::Instance().GetBuilder()->Build<YieldOp>(input_values);
  });

  BindIfOp(m);
  BindWhileOp(m);
}
}  // namespace pybind
}  // namespace paddle
