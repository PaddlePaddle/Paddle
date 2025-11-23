// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/packed_ir_op_declare_data.h"
#include "paddle/ap/include/drr/tags.h"
#include "paddle/ap/include/drr/type.h"

namespace ap::drr {

struct OpPatternCtxImpl;

template <typename NodeT>
struct PackedIrOpDeclareImpl {
  std::string op_name;
  std::weak_ptr<OpPatternCtxImpl> op_pattern_ctx;
  std::optional<std::shared_ptr<PackedIrOpDeclareData>> data{};

  bool operator==(const PackedIrOpDeclareImpl& other) const {
    return this->op_name == other.op_name &&
           this->op_pattern_ctx.lock() == other.op_pattern_ctx.lock();
  }

  template <typename T>
  adt::Result<T*> cast_data() const {
    auto ThisToString = [&]() {
      const void* address = static_cast<const void*>(this);
      std::ostringstream ss;
      ss << address;
      return ss.str();
    };
    ADT_CHECK(data.has_value())
        << adt::errors::ValueError{std::string() + "((PackedIrOpDeclareImpl*)" +
                                   ThisToString() + ")->data is nullopt"};
    ADT_CHECK(data.value().get() != nullptr) << adt::errors::ValueError{
        std::string() + "((PackedIrOpDeclareImpl*)" + ThisToString() +
        ")->data.value() is nullptr"};
    auto* ptr = dynamic_cast<T*>(data.value().get());
    ADT_CHECK(data.value().get() != nullptr) << adt::errors::ValueError{
        std::string() + "((PackedIrOpDeclareImpl*)" + ThisToString() +
        ")->data.value() cast to " + typeid(T).name() + " failed."};
    return ptr;
  }
};

template <typename NodeT>
ADT_DEFINE_RC(PackedIrOpDeclare, PackedIrOpDeclareImpl<NodeT>);

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnPackedIrOpDeclareClass();

template <typename NodeT>
struct Type<drr::tSrcPtn<drr::PackedIrOpDeclare<NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "SrcPtnPackedIrOpDeclare"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetSrcPtnPackedIrOpDeclareClass();
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResPtnPackedIrOpDeclareClass();

template <typename NodeT>
struct Type<drr::tResPtn<drr::PackedIrOpDeclare<NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "ResPtnPackedIrOpDeclare"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetResPtnPackedIrOpDeclareClass();
  }
};

}  // namespace ap::drr
