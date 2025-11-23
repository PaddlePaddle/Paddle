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

#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/memory/circlable_ref.h"

namespace ap::axpr {

template <typename ValueT>
struct Frame : public memory::CirclableRef<Frame<ValueT>, AttrMapImpl<ValueT>> {
  using memory::CirclableRef<Frame<ValueT>, AttrMapImpl<ValueT>>::CirclableRef;
};

}  // namespace ap::axpr
