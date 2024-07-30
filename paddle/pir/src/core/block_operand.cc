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

#include "paddle/pir/include/core/block_operand.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/src/core/block_operand_impl.h"

#include "paddle/common/enforce.h"

namespace pir {

#define CHECK_BLOCK_OPERAND_NULL_IMPL(func_name)             \
  PADDLE_ENFORCE_NOT_NULL(                                   \
      impl_,                                                 \
      common::errors::InvalidArgument(                       \
          "impl_ pointer is null when call func:" #func_name \
          " , in class: BlockOperand."))

BlockOperand &BlockOperand::operator=(const BlockOperand &rhs) {
  if (this == &rhs) return *this;
  impl_ = rhs.impl_;
  return *this;
}

BlockOperand::operator bool() const { return impl_ && impl_->source(); }

BlockOperand BlockOperand::next_use() const {
  CHECK_BLOCK_OPERAND_NULL_IMPL(next_use);
  return impl_->next_use();
}

Block *BlockOperand::source() const {
  CHECK_BLOCK_OPERAND_NULL_IMPL(source);
  return impl_->source();
}

void BlockOperand::set_source(Block *source) {
  CHECK_BLOCK_OPERAND_NULL_IMPL(set_source);
  impl_->set_source(source);
}

Operation *BlockOperand::owner() const {
  CHECK_BLOCK_OPERAND_NULL_IMPL(owner);
  return impl_->owner();
}

void BlockOperand::RemoveFromUdChain() {
  CHECK_BLOCK_OPERAND_NULL_IMPL(RemoveFromUdChain);
  return impl_->RemoveFromUdChain();
}

// details
namespace detail {

Operation *BlockOperandImpl::owner() const { return owner_; }

BlockOperand BlockOperandImpl::next_use() const { return next_use_; }

Block *BlockOperandImpl::source() const { return source_; }

void BlockOperandImpl::set_source(Block *source) {
  RemoveFromUdChain();
  if (!source) {
    return;
  }
  source_ = source;
  InsertToUdChain();
}

BlockOperandImpl::BlockOperandImpl(Block *source, pir::Operation *owner)
    : source_(source), owner_(owner) {
  if (!source) {
    return;
  }
  InsertToUdChain();
}

void BlockOperandImpl::InsertToUdChain() {
  prev_use_addr_ = source_->first_use_addr();
  next_use_ = source_->first_use();
  if (next_use_) {
    next_use_.impl()->prev_use_addr_ = &next_use_;
  }
  source_->set_first_use(this);
}

void BlockOperandImpl::RemoveFromUdChain() {
  if (!source_) return;
  if (!prev_use_addr_) return;
  if (prev_use_addr_ == source_->first_use_addr()) {
    source_->set_first_use(next_use_);
  } else {
    *prev_use_addr_ = next_use_;
  }
  if (next_use_) {
    next_use_.impl()->prev_use_addr_ = prev_use_addr_;
  }
  next_use_ = nullptr;
  prev_use_addr_ = nullptr;
  source_ = nullptr;
}

BlockOperandImpl::~BlockOperandImpl() { RemoveFromUdChain(); }
}  // namespace detail
}  // namespace pir
