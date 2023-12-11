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
#include "paddle/pir/core/block_argument.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/operation_utils.h"
#include "paddle/pir/core/value_impl.h"

#define CHECK_NULL_IMPL(func_name) \
  IR_ENFORCE(impl_, "impl_ is null when called BlockArgument:" #func_name)

#define IMPL_ static_cast<detail::BlockArgumentImpl *>(impl_)

namespace pir {

namespace detail {
///
/// \brief BlockArgumentImpl is the implementation of an block argument.
///
class BlockArgumentImpl : public ValueImpl {
 public:
  static bool classof(const ValueImpl &value) {
    return value.kind() == BLOCK_ARG_IDX;
  }

  ///
  /// \brief attribute related public interfaces
  ///
  Attribute attribute(const std::string &key) const {
    auto iter = attributes_.find(key);
    return iter == attributes_.end() ? nullptr : iter->second;
  }

  void set_attribute(const std::string &key, Attribute value) {
    attributes_[key] = value;
  }

 private:
  BlockArgumentImpl(Type type, Block *owner, uint32_t index)
      : ValueImpl(type, BLOCK_ARG_IDX), owner_(owner), index_(index) {}

  ~BlockArgumentImpl();
  // access construction and owner
  friend BlockArgument;

  AttributeMap attributes_;
  Block *owner_;
  uint32_t index_;
};

BlockArgumentImpl::~BlockArgumentImpl() {
  if (!use_empty()) {
    LOG(FATAL) << "Destoryed a blockargument that is still in use.";
  }
}

}  // namespace detail

BlockArgument::BlockArgument(detail::BlockArgumentImpl *impl) : Value(impl) {}

bool BlockArgument::classof(Value value) {
  return value && detail::BlockArgumentImpl::classof(*value.impl());
}

Block *BlockArgument::owner() const {
  CHECK_NULL_IMPL(owner);
  return IMPL_->owner_;
}

uint32_t BlockArgument::index() const {
  CHECK_NULL_IMPL(index);
  return IMPL_->index_;
}

Attribute BlockArgument::attribute(const std::string &key) const {
  return impl_ ? IMPL_->attribute(key) : nullptr;
}
void BlockArgument::set_attribute(const std::string &key, Attribute value) {
  CHECK_NULL_IMPL(set_attribute);
  return IMPL_->set_attribute(key, value);
}

BlockArgument BlockArgument::Create(Type type, Block *owner, uint32_t index) {
  return new detail::BlockArgumentImpl(type, owner, index);
}
/// Destroy the argument.
void BlockArgument::Destroy() {
  if (impl_) {
    delete IMPL_;
  } else {
    LOG(WARNING) << "Destroying a null block argument.";
  }
}

void BlockArgument::set_index(uint32_t index) {
  CHECK_NULL_IMPL(set_arg_number);
  IMPL_->index_ = index;
}

BlockArgument BlockArgument::dyn_cast_from(Value value) {
  if (classof(value)) {
    return static_cast<detail::BlockArgumentImpl *>(value.impl());
  } else {
    return nullptr;
  }
}

}  // namespace pir
