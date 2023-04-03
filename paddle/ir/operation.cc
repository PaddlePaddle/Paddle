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

#include "paddle/ir/operation.h"
#include "paddle/ir/utils.h"

namespace ir {

Operation *Operation::create(const std::vector<ir::OpResult> &inputs,
                             const std::vector<ir::Type> &output_types,
                             ir::DictionaryAttribute attibute) {
  // Get num of Results.
  uint32_t num_results = output_types.size();
  uint32_t num_operands = inputs.size();
  uint32_t max_inline_result_num =
      detail::OpResultImpl::GetMaxInlineResultIndex() + 1;
  // Calculate the memory space required for OpResults + Operation + OpOperands
  size_t result_mem_size =
      num_results > max_inline_result_num
          ? sizeof(detail::OpOutlineResultImpl) *
                    (num_results - max_inline_result_num) +
                sizeof(detail::OpInlineResultImpl) * max_inline_result_num
          : sizeof(detail::OpInlineResultImpl) * num_results;
  size_t operand_mem_size = sizeof(detail::OpOperandImpl) * num_operands;
  size_t op_mem_size = sizeof(Operation);
  size_t base_size = result_mem_size + op_mem_size + operand_mem_size;
  // Malloc memory and construct OpResults, Operation, OpOperands.
  char *malloc_memory = reinterpret_cast<char *>(aligned_malloc(base_size, 8));
  char *base_ptr = malloc_memory;
  for (size_t idx = num_results; idx > 0; idx--) {
    if (idx > 6) {
      new (base_ptr)
          detail::OpOutlineResultImpl(output_types[idx - 1], idx - 1);
      base_ptr += sizeof(detail::OpOutlineResultImpl);
    } else {
      new (base_ptr) detail::OpInlineResultImpl(output_types[idx - 1], idx - 1);
      base_ptr += sizeof(detail::OpInlineResultImpl);
    }
  }
  Operation *op = new (base_ptr) Operation(num_results, num_operands, attibute);
  base_ptr += sizeof(Operation);
  assert((reinterpret_cast<uintptr_t>(base_ptr) & 0x7) == 0 &&
         "The address of OpOperandImpl must be divisible by 8.");
  for (size_t idx = 0; idx < num_operands; idx++) {
    new (base_ptr) detail::OpOperandImpl(inputs[idx].impl_, op);
    base_ptr += sizeof(detail::OpOperandImpl);
  }
  VLOG(4) << "Construct an Operation ----------------->";
  VLOG(4) << op->print();
  VLOG(4) << "---------------------------------------->";
  return op;
}

void Operation::destroy() {
  this->~Operation();
  uint32_t max_inline_result_num =
      detail::OpResultImpl::GetMaxInlineResultIndex() + 1;
  size_t result_mem_size =
      num_results_ > max_inline_result_num
          ? sizeof(detail::OpOutlineResultImpl) *
                    (num_results_ - max_inline_result_num) +
                sizeof(detail::OpInlineResultImpl) * max_inline_result_num
          : sizeof(detail::OpInlineResultImpl) * num_results_;
  void *aligned_ptr = reinterpret_cast<void *>(reinterpret_cast<char *>(this) -
                                               result_mem_size);
  VLOG(4) << "Destroy an Operation --------------------->";
  VLOG(4) << "aligned_ptr = " << aligned_ptr << ", size = " << result_mem_size;
  aligned_free(aligned_ptr);
  VLOG(4) << "------------------------------------------>";
}

Operation::Operation(uint32_t num_results,
                     uint32_t num_operands,
                     ir::DictionaryAttribute attibute) {
  assert(attibute && "unexpected null attribute dictionary");
  num_results_ = num_results;
  num_operands_ = num_operands;
  attibute_ = attibute;
}

Operation::~Operation() {}

ir::OpResult Operation::GetResultByIndex(uint32_t index) {
  assert((index < num_results_) && "index exceeds OP output range.");
  uint32_t max_inline_idx = detail::OpResultImpl::GetMaxInlineResultIndex();
  char *ptr = nullptr;
  if (index > max_inline_idx) {
    ptr = reinterpret_cast<char *>(this) -
          (max_inline_idx + 1) * sizeof(detail::OpInlineResultImpl) -
          (index - max_inline_idx) * sizeof(detail::OpOutlineResultImpl);
  } else {
    ptr = reinterpret_cast<char *>(this) -
          (index + 1) * sizeof(detail::OpInlineResultImpl);
  }
  if (index > max_inline_idx) {
    detail::OpOutlineResultImpl *result_impl_ptr =
        reinterpret_cast<detail::OpOutlineResultImpl *>(ptr);
    return ir::OpResult(result_impl_ptr);
  } else {
    detail::OpInlineResultImpl *result_impl_ptr =
        reinterpret_cast<detail::OpInlineResultImpl *>(ptr);
    return ir::OpResult(result_impl_ptr);
  }
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

}  // namespace ir
