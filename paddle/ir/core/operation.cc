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
    op->GetRegion(index).TakeBody(std::move(*argument.regions[index]));
  }
  return op;
}

// Allocate the required memory based on the size and number of inputs, outputs,
// and operators, and construct it in the order of: OpOutlineResult,
// OpInlineResult, Operation, Operand.
Operation *Operation::Create(const std::vector<ir::OpResult> &inputs,
                             const AttributeMap &attributes,
                             const std::vector<ir::Type> &output_types,
                             ir::OpInfo op_info,
                             size_t num_regions) {
  // 0. Verify
  if (op_info) {
    op_info.Verify(inputs, output_types, attributes);
  }
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
  return op;
}

// Call destructors for OpResults, Operation, and OpOperands in sequence, and
// finally free memory.
void Operation::Destroy() {
  // Deconstruct Regions.
  if (num_regions_ > 0) {
    for (size_t idx = 0; idx < num_regions_; idx++) {
      regions_[idx].~Region();
    }
  }

  // 1. Get aligned_ptr by result_num.
  uint32_t max_inline_result_num =
      detail::OpResultImpl::GetMaxInlineResultIndex() + 1;
  size_t result_mem_size =
      num_results_ > max_inline_result_num
          ? sizeof(detail::OpOutlineResultImpl) *
                    (num_results_ - max_inline_result_num) +
                sizeof(detail::OpInlineResultImpl) * max_inline_result_num
          : sizeof(detail::OpInlineResultImpl) * num_results_;
  char *aligned_ptr = reinterpret_cast<char *>(this) - result_mem_size;
  // 2.1. Deconstruct OpResult.
  char *base_ptr = aligned_ptr;
  for (size_t idx = num_results_; idx > 0; idx--) {
    // release the uses of this result
    detail::OpOperandImpl *first_use =
        reinterpret_cast<detail::OpResultImpl *>(base_ptr)->first_use();
    while (first_use != nullptr) {
      first_use->remove_from_ud_chain();
      first_use =
          reinterpret_cast<detail::OpResultImpl *>(base_ptr)->first_use();
    }
    // destory the result
    if (idx > max_inline_result_num) {
      reinterpret_cast<detail::OpOutlineResultImpl *>(base_ptr)
          ->~OpOutlineResultImpl();
      base_ptr += sizeof(detail::OpOutlineResultImpl);
    } else {
      reinterpret_cast<detail::OpInlineResultImpl *>(base_ptr)
          ->~OpInlineResultImpl();
      base_ptr += sizeof(detail::OpInlineResultImpl);
    }
  }
  // 2.2. Deconstruct Operation.
  if (reinterpret_cast<uintptr_t>(base_ptr) !=
      reinterpret_cast<uintptr_t>(this)) {
    IR_THROW("Operation address error");
  }
  reinterpret_cast<Operation *>(base_ptr)->~Operation();
  base_ptr += sizeof(Operation);
  // 2.3. Deconstruct OpOperand.
  for (size_t idx = 0; idx < num_operands_; idx++) {
    reinterpret_cast<detail::OpOperandImpl *>(base_ptr)->~OpOperandImpl();
    base_ptr += sizeof(detail::OpOperandImpl);
  }
  // 3. Free memory.
  VLOG(4) << "Destroy an Operation: {ptr = "
          << reinterpret_cast<void *>(aligned_ptr)
          << ", size = " << result_mem_size << "}";
  aligned_free(reinterpret_cast<void *>(aligned_ptr));
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

ir::OpResult Operation::GetResultByIndex(uint32_t index) const {
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

ir::OpOperand Operation::GetOperandByIndex(uint32_t index) const {
  if (index >= num_operands_) {
    IR_THROW("index exceeds OP input range.");
  }
  const char *ptr = reinterpret_cast<const char *>(this) + sizeof(Operation) +
                    (index) * sizeof(detail::OpOperandImpl);
  return ir::OpOperand(reinterpret_cast<const detail::OpOperandImpl *>(ptr));
}

std::string Operation::name() const {
  auto p_name = info_.name();
  return p_name ? p_name : "";
}

Region *Operation::GetParentRegion() const {
  return parent_ ? parent_->GetParentRegion() : nullptr;
}

Operation *Operation::GetParentOp() const {
  return parent_ ? parent_->GetParentOp() : nullptr;
}

Program *Operation::GetParentProgram() {
  Operation *op = this;
  while (Operation *parent_op = op->GetParentOp()) {
    op = parent_op;
  }
  ModuleOp module_op = op->dyn_cast<ModuleOp>();
  return module_op ? module_op.program() : nullptr;
}

Region &Operation::GetRegion(unsigned index) {
  assert(index < num_regions_ && "invalid region index");
  return regions_[index];
}

}  // namespace ir
