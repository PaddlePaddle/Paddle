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

#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/native_ir_value.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/tags.h"
#include "paddle/ap/include/drr/unbound_ir_value.h"

namespace ap::drr {

using ResPtnValidOutIrValueImpl =
    std::variant<UnboundIrValue<drr::Node>, tResPtn<NativeIrValue<drr::Node>>>;

struct ResPtnValidOutIrValue : public ResPtnValidOutIrValueImpl {
  using ResPtnValidOutIrValueImpl::ResPtnValidOutIrValueImpl;

  ADT_DEFINE_VARIANT_METHODS(ResPtnValidOutIrValueImpl);

  static adt::Result<ResPtnValidOutIrValue> CastFromAxprValue(
      const axpr::Value& val) {
    if (val.template CastableTo<UnboundIrValue<drr::Node>>()) {
      ADT_LET_CONST_REF(ret, val.template CastTo<UnboundIrValue<drr::Node>>());
      return ret;
    }
    if (val.template CastableTo<tResPtn<NativeIrValue<drr::Node>>>()) {
      ADT_LET_CONST_REF(
          ret, val.template CastTo<tResPtn<NativeIrValue<drr::Node>>>());
      return ret;
    }
    return adt::errors::TypeError{
        "ResPtnValidOutIrValue::CastFromAxprValue() failed"};
  }

  const std::string& name() const {
    return Match([](const tResPtn<NativeIrValue<drr::Node>>& ir_value)
                     -> const std::string& { return ir_value.value()->name; },
                 [](const auto& ir_value) -> const std::string& {
                   return ir_value->name;
                 });
  }
};

}  // namespace ap::drr
