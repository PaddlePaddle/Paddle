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

#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"

namespace ir {
Operation *Operation::create(const OperationArgument &argument) {
  return create(argument.inputs_,
                argument.output_types_,
                argument.attribute_,
                argument.info_);
}

// Allocate the required memory based on the size and number of inputs, outputs,
// and operators, and construct it in the order of: OpOutlineResult,
// OpInlineResult, Operation, Operand.
Operation *Operation::create(const std::vector<ir::OpResult> &inputs,
                             const std::vector<ir::Type> &output_types,
                             const AttributeMap &attribute,
                             ir::OpInfo op_info) {
  // 0. Verify
  if (op_info) {
    op_info.verify(inputs, output_types, attribute);
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
  size_t base_size = result_mem_size + op_mem_size + operand_mem_size;
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
  Operation *op =
      new (base_ptr) Operation(num_results, num_operands, attribute, op_info);
  base_ptr += sizeof(Operation);
  // 3.3. Construct OpOperands.
  if ((reinterpret_cast<uintptr_t>(base_ptr) & 0x7) != 0) {
    throw("The address of OpOperandImpl must be divisible by 8.");
  }
  for (size_t idx = 0; idx < num_operands; idx++) {
    new (base_ptr) detail::OpOperandImpl(inputs[idx].impl_, op);
    base_ptr += sizeof(detail::OpOperandImpl);
  }

  return op;
}

// Call destructors for OpResults, Operation, and OpOperands in sequence, and
// finally free memory.
void Operation::destroy() {
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
    throw("Operation address error");
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

IrContext *Operation::ir_context() const { return op_info_.ir_context(); }

Operation::Operation(uint32_t num_results,
                     uint32_t num_operands,
                     const AttributeMap &attribute,
                     ir::OpInfo op_info) {
  num_results_ = num_results;
  num_operands_ = num_operands;
  attribute_ = attribute;
  op_info_ = op_info;
}

ir::OpResult Operation::GetResultByIndex(uint32_t index) const {
  if (index >= num_results_) {
    throw("index exceeds OP output range.");
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
    throw("index exceeds OP input range.");
  }
  const char *ptr = reinterpret_cast<const char *>(this) + sizeof(Operation) +
                    (index) * sizeof(detail::OpOperandImpl);
  return ir::OpOperand(reinterpret_cast<const detail::OpOperandImpl *>(ptr));
}

std::string Operation::print() {
  std::stringstream result;
  result << "{ " << num_results_ << " outputs, " << num_operands_
         << " inputs } : ";
  result << "[ ";
  for (size_t idx = num_results_; idx > 0; idx--) {
    result << GetResultByIndex(idx - 1).impl_ << ", ";
  }
  result << "] = ";
  result << this << "( ";
  for (size_t idx = 0; idx < num_operands_; idx++) {
    result << reinterpret_cast<void *>(reinterpret_cast<char *>(this) +
                                       sizeof(Operation) +
                                       idx * sizeof(detail::OpOperandImpl))
           << ", ";
  }
  result << ")";
  return result.str();
}

std::string Operation::op_name() const { return op_info_.name(); }

}  // namespace ir
