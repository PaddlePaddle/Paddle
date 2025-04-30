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
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/kernel_dispatch/const_tensor_method_class.h"
#include "paddle/ap/include/kernel_dispatch/dispatch_ctx_method_class.h"
#include "paddle/ap/include/kernel_dispatch/mutable_tensor_method_class.h"

namespace ap::kernel_dispatch {

template <typename ValueT, typename DoEachT>
void VisitEachBuiltinFrameAttr(const DoEachT& DoEach) {
  // Do Nothing.
}

template <typename ValueT>
axpr::AttrMap<ValueT> MakeBuiltinFrameAttrMap() {
  axpr::AttrMap<ValueT> attr_map;
  axpr::VisitEachBuiltinFrameAttr<ValueT>(
      [&](const std::string& k, const ValueT& v) { attr_map->Set(k, v); });
  VisitEachBuiltinFrameAttr<ValueT>([&](const std::string& k, const ValueT& v) {
    attr_map->Set(k, v);
    attr_map->Set(std::string("__builtin__") + k, v);
  });
  return attr_map;
}

}  // namespace ap::kernel_dispatch
