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
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/drr/drr_ctx_method_class.h"
#include "paddle/ap/include/drr/drr_value.h"

namespace ap::drr {

template <typename DoEachT>
void VisitEachBuiltinFrameClass(const DoEachT& DoEach) {
  DoEach(drr::Type<DrrCtx>{}.GetClass());
}

template <typename VisitorT>
ap::axpr::AttrMap<axpr::Value> MakeBuiltinFrameAttrMap(
    const VisitorT& Visitor) {
  ap::axpr::AttrMap<axpr::Value> attr_map;
  ap::axpr::VisitEachBuiltinFrameAttr<axpr::Value>(
      [&](const std::string& k, const axpr::Value& v) { attr_map->Set(k, v); });
  auto Insert = [&](const auto& cls) {
    attr_map->Set(cls.Name(), cls);
    attr_map->Set(std::string("__builtin__") + cls.Name(), cls);
  };
  VisitEachBuiltinFrameClass(Insert);
  Visitor(Insert);
  return attr_map;
}

}  // namespace ap::drr
