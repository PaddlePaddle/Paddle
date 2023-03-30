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
  uint8_t num_results = output_types.size();
  uint8_t num_operands = inputs.size();
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
  return op;
}

void Operation::destroy() {
  uint32_t max_inline_result_num =
      detail::OpResultImpl::GetMaxInlineResultIndex() + 1;
  size_t result_mem_size =
      num_results_ > max_inline_result_num
          ? sizeof(detail::OpOutlineResultImpl) *
                    (num_results_ - max_inline_result_num) +
                sizeof(detail::OpInlineResultImpl) * max_inline_result_num
          : sizeof(detail::OpInlineResultImpl) * num_results_;
  void *aligned_ptr = reinterpret_cast<char *>(this) - result_mem_size;
  aligned_free(aligned_ptr);
}

Operation::Operation(uint8_t num_results,
                     uint8_t num_operands,
                     ir::DictionaryAttribute attibute) {
  assert(attibute && "unexpected null attribute dictionary");
  num_results_ = num_results;
  num_operands_ = num_operands;
  attibute_ = attibute;
}

Operation::~Operation() { destroy(); }

}  // namespace ir
