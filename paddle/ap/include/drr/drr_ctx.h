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
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/drr_pass_type.h"
#include "paddle/ap/include/drr/result_pattern_ctx.h"
#include "paddle/ap/include/drr/source_pattern_ctx.h"
#include "paddle/ap/include/drr/tags.h"
#include "paddle/ap/include/drr/type.h"
#include "paddle/ap/include/memory/circlable_ref_list_base.h"

namespace ap::drr {

struct DrrCtxImpl {
  std::weak_ptr<ap::memory::CirclableRefListBase> circlable_ref_list;
  std::optional<std::string> pass_name{};
  std::optional<SourcePatternCtx> source_pattern_ctx{};
  std::optional<ResultPatternCtx> result_pattern_ctx{};
  std::optional<axpr::Value> constraint_func{};
  std::optional<drr::DrrPassType> drr_pass_type{};

  adt::Result<SourcePatternCtx> GetSourcePatternCtx() const {
    ADT_CHECK(this->source_pattern_ctx.has_value());
    return this->source_pattern_ctx.value();
  }

  adt::Result<ResultPatternCtx> GetResultPatternCtx() const {
    ADT_CHECK(this->result_pattern_ctx.has_value());
    return this->result_pattern_ctx.value();
  }

  bool operator==(const DrrCtxImpl& other) const { return this == &other; }
};

ADT_DEFINE_RC(DrrCtx, DrrCtxImpl);

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetDrrCtxClass();

template <>
struct Type<drr::DrrCtx> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "DrrCtx"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetDrrCtxClass();
  }
};

}  // namespace ap::drr
