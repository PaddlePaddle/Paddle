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
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/packed_ir_value.h"
#include "paddle/ap/include/drr/tags.h"
#include "paddle/ap/include/drr/unbound_ir_value.h"
#include "paddle/ap/include/drr/unbound_packed_ir_op.h"

namespace ap::drr {

using SrcPtnValidInIrValueImpl = std::variant<PackedIrValue<drr::Node>,
                                              NativeIrValue<drr::Node>,
                                              UnboundIrValue<drr::Node>,
                                              UnboundPackedIrValue<drr::Node>>;

struct SrcPtnValidInIrValue : public SrcPtnValidInIrValueImpl {
  using SrcPtnValidInIrValueImpl::SrcPtnValidInIrValueImpl;

  ADT_DEFINE_VARIANT_METHODS(SrcPtnValidInIrValueImpl);

  const std::string& name() const {
    return Match([](const auto& ir_value) -> const std::string& {
      return ir_value->name;
    });
  }
};

}  // namespace ap::drr
