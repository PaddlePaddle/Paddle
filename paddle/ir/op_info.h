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

#pragma once

#include <functional>

#include "paddle/ir/op_info_impl.h"

namespace ir {
class OpInfo {
 public:
  constexpr OpInfo() = default;

  OpInfo(const OpInfoImpl *impl) : impl_(impl) {}  // NOLINT

  OpInfo(const OpInfo &other) = default;

  OpInfo &operator=(const OpInfo &other) = default;

  bool operator==(OpInfo other) const { return impl_ == other.impl_; }

  bool operator!=(OpInfo other) const { return impl_ != other.impl_; }

  explicit operator bool() const { return impl_; }

  bool operator!() const { return impl_ == nullptr; }

  const OpInfoImpl *impl() const { return impl_; }

  template <typename Trait>
  bool HasTrait() const {
    return impl_->HasTrait<Trait>();
  }

  template <typename Interface>
  bool HasInterface() const {
    return impl_->HasInterface<Interface>();
  }

  friend struct std::hash<OpInfo>;

 private:
  const OpInfoImpl *impl_{nullptr};  // not owned
};

}  // namespace ir

namespace std {
template <>
struct hash<ir::OpInfo> {
  std::size_t operator()(const ir::OpInfo &obj) const {
    return std::hash<const ir::OpInfoImpl *>()(obj.impl_);
  }
};
}  // namespace std
