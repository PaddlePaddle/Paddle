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

#include <ostream>

#include "paddle/ir/core/block.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/op_info.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/region.h"
#include "paddle/ir/core/utils.h"
#include "paddle/ir/core/value_impl.h"

namespace ir {
Operation *Operation::Create(OperationArgument &&argument) {
  Operation *op = Create(argument.inputs,
                         argument.attributes,
                         argument.output_types,
                         argument.info,
                         argument.regions.size());

  for (size_t index = 0; index < argument.regions.size(); ++index) {
    op->region(index).TakeBody(std::move(*argument.regions[index]));
  }
  return op;
}

// Allocate the required memory based on the size and number of inputs, outputs,
// and operators, and construct it in the order of: OpOutlineResult,
// OpInlineResult, Operation, operand.
Operation *Operation::Create(const std::vector<ir::OpResult> &inputs,
                             const AttributeMap &attributes,
                             const std::vector<ir::Type> &output_types,
                             ir::OpInfo op_info,
                             size_t num_regions) {
  // 1. Calculate the required memory size for OpResults + Operation +
  // OpOperands.
  uint32_t num_results = output_types.size();
  uint32_t num_operands = inputs.size();
  uint32_t max_inline_result_num =
      detail::OpResultImpl::GetMaxInlineResultIndex() + 1;
  size_t result_mem_size =
      num_results > max_inline_result_num
          ? sizeof(detail::OpOutlineResultImpl) *
                    (num_results - max_inline_result_num) +
                sizeof(detail::OpInlineResultImpl) * max_inline_result_num
          : sizeof(detail::OpInlineResultImpl) * num_results;
  size_t operand_mem_size = sizeof(detail::OpOperandImpl) * num_operands;
  size_t op_mem_size = sizeof(Operation);
  size_t region_mem_size = num_regions * sizeof(Region);
  size_t base_size =
      result_mem_size + op_mem_size + operand_mem_size + region_mem_size;
  // 2. Malloc memory.
  char *base_ptr = reinterpret_cast<char *>(aligned_malloc(base_size, 8));
  // 3.1. Construct OpResults.
  for (size_t idx = num_results; idx > 0; idx--) {
    if (idx > max_inline_result_num) {
      new (base_ptr)
          detail::OpOutlineResultImpl(output_types[idx - 1], idx - 1);
      base_ptr += sizeof(detail::OpOutlineResultImpl);
    } else {
      new (base_ptr) detail::OpInlineResultImpl(output_types[idx - 1], idx - 1);
      base_ptr += sizeof(detail::OpInlineResultImpl);
    }
  }
  // 3.2. Construct Operation.
  Operation *op = new (base_ptr)
      Operation(attributes, op_info, num_results, num_operands, num_regions);
  base_ptr += sizeof(Operation);
  // 3.3. Construct OpOperands.
  if ((reinterpret_cast<uintptr_t>(base_ptr) & 0x7) != 0) {
    IR_THROW("The address of OpOperandImpl must be divisible by 8.");
  }
  for (size_t idx = 0; idx < num_operands; idx++) {
    new (base_ptr) detail::OpOperandImpl(inputs[idx].impl_, op);
    base_ptr += sizeof(detail::OpOperandImpl);
  }
  // 3.4. Construct Regions
  if (num_regions > 0) {
    op->regions_ = reinterpret_cast<Region *>(base_ptr);
    for (size_t idx = 0; idx < num_regions; idx++) {
      new (base_ptr) Region(op);
      base_ptr += sizeof(Region);
    }
  }

  // 0. Verify
  if (op_info) {
    op_info.Verify(op);
  }
  return op;
}

// Call destructors for Region , OpResults, Operation, and OpOperands in
// sequence, and finally free memory.
void Operation::Destroy() {
  VLOG(6) << "Destroy Operation [" << name() << "] ...";
  // 1. Deconstruct Regions.
  if (num_regions_ > 0) {
    for (size_t idx = 0; idx < num_regions_; idx++) {
      regions_[idx].~Region();
    }
  }

  // 2. Deconstruct Result.
  for (size_t idx = 0; idx < num_results_; ++idx) {
    detail::OpResultImpl *impl = result(idx).impl();
    IR_ENFORCE(impl->use_empty(),
               name() + " operation destroyed but still has uses.");
    if (detail::OpOutlineResultImpl::classof(*impl)) {
      static_cast<detail::OpOutlineResultImpl *>(impl)->~OpOutlineResultImpl();
    } else {
      static_cast<detail::OpInlineResultImpl *>(impl)->~OpInlineResultImpl();
    }
  }

  // 3. Deconstruct Operation.
  this->~Operation();

  // 4. Deconstruct OpOperand.
  for (size_t idx = 0; idx < num_operands_; idx++) {
    operand(idx).impl()->~OpOperandImpl();
  }
  // 5. Free memory.
  uint32_t max_inline_result_num =
      detail::OpResultImpl::GetMaxInlineResultIndex() + 1;
  size_t result_mem_size =
      num_results_ > max_inline_result_num
          ? sizeof(detail::OpOutlineResultImpl) *
                    (num_results_ - max_inline_result_num) +
                sizeof(detail::OpInlineResultImpl) * max_inline_result_num
          : sizeof(detail::OpInlineResultImpl) * num_results_;
  void *aligned_ptr = reinterpret_cast<char *>(this) - result_mem_size;

  VLOG(6) << "Destroy Operation [" << name() << "]: {ptr = " << aligned_ptr
          << ", size = " << result_mem_size << "} done.";
  aligned_free(aligned_ptr);
}

IrContext *Operation::ir_context() const { return info_.ir_context(); }

Dialect *Operation::dialect() const { return info_.dialect(); }

Operation::Operation(const AttributeMap &attributes,
                     ir::OpInfo op_info,
                     uint32_t num_results,
                     uint32_t num_operands,
                     uint32_t num_regions)
    : attributes_(attributes),
      info_(op_info),
      num_results_(num_results),
      num_operands_(num_operands),
      num_regions_(num_regions) {}

ir::OpResult Operation::result(uint32_t index) const {
  if (index >= num_results_) {
    IR_THROW("index exceeds OP output range.");
  }
  uint32_t max_inline_idx = detail::OpResultImpl::GetMaxInlineResultIndex();
  const char *ptr =
      (index > max_inline_idx)
          ? reinterpret_cast<const char *>(this) -
                (max_inline_idx + 1) * sizeof(detail::OpInlineResultImpl) -
                (index - max_inline_idx) * sizeof(detail::OpOutlineResultImpl)
          : reinterpret_cast<const char *>(this) -
                (index + 1) * sizeof(detail::OpInlineResultImpl);
  if (index > max_inline_idx) {
    return ir::OpResult(
        reinterpret_cast<const detail::OpOutlineResultImpl *>(ptr));
  } else {
    return ir::OpResult(
        reinterpret_cast<const detail::OpInlineResultImpl *>(ptr));
  }
}

OpOperand Operation::operand(uint32_t index) const {
  if (index >= num_operands_) {
    IR_THROW("index exceeds OP input range.");
  }
  const char *ptr = reinterpret_cast<const char *>(this) + sizeof(Operation) +
                    (index) * sizeof(detail::OpOperandImpl);
  return OpOperand(reinterpret_cast<const detail::OpOperandImpl *>(ptr));
}

Value Operation::operand_source(uint32_t index) const {
  OpOperand val = operand(index);
  return val ? val.source() : Value();
}

std::string Operation::name() const {
  auto p_name = info_.name();
  return p_name ? p_name : "";
}

Attribute Operation::attribute(const std::string &key) const {
  IR_ENFORCE(HasAttribute(key), "operation(%s): no attribute %s", name(), key);
  return attributes_.at(key);
}

Region *Operation::GetParentRegion() {
  return parent_ ? parent_->GetParent() : nullptr;
}

Operation *Operation::GetParentOp() const {
  return parent_ ? parent_->GetParentOp() : nullptr;
}

const Program *Operation::GetParentProgram() const {
  Operation *op = const_cast<Operation *>(this);
  while (Operation *parent_op = op->GetParentOp()) {
    op = parent_op;
  }
  ModuleOp module_op = op->dyn_cast<ModuleOp>();
  return module_op ? module_op.program() : nullptr;
}

Region &Operation::region(unsigned index) {
  assert(index < num_regions_ && "invalid region index");
  return regions_[index];
}

const Region &Operation::region(unsigned index) const {
  assert(index < num_regions_ && "invalid region index");
  return regions_[index];
}

void Operation::SetParent(Block *parent, const Block::iterator &position) {
  parent_ = parent;
  position_ = position;
}

void Operation::ReplaceAllUsesWith(const std::vector<Value> &values) {
  IR_ENFORCE(num_results_ == values.size(),
             "the num of result should be the same.");
  for (uint32_t i = 0; i < num_results_; ++i) {
    result(i).ReplaceAllUsesWith(values[i]);
  }
}

void Operation::ReplaceAllUsesWith(const std::vector<OpResult> &op_results) {
  IR_ENFORCE(num_results_ == op_results.size(),
             "the num of result should be the same.");
  for (uint32_t i = 0; i < num_results_; ++i) {
    result(i).ReplaceAllUsesWith(op_results[i]);
  }
}

void Operation::Verify() {
  if (info_) {
    info_.Verify(this);
  }
}

}  // namespace ir
