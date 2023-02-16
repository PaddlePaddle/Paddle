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

namespace ir {
///
/// \brief This macro definition is used to add some necessary functions to the
/// custom Type class.
///
#define REGISTER_TYPE_UTILS(concrete_type, storage_type)                   \
  using Type::Type;                                                        \
  using StorageType = storage_type;                                        \
  StorageType *storage() const {                                           \
    return static_cast<StorageType *>(this->storage_);                     \
  }                                                                        \
  static ir::TypeId type_id() { return ir::TypeId::get<concrete_type>(); } \
  template <typename T>                                                    \
  static bool classof(T val) {                                             \
    return val.type_id() == type_id();                                     \
  }                                                                        \
  template <typename... Args>                                              \
  static concrete_type create(IrContext *ctx, Args... args) {              \
    return ir::TypeUniquer::template get<concrete_type>(ctx, args...);     \
  }

}  // namespace ir
