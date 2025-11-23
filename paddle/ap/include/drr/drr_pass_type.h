// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

namespace ap::drr {

struct AbstractDrrPassType : public std::monostate {
  using std::monostate::monostate;
};
struct ReifiedDrrPassType : public std::monostate {
  using std::monostate::monostate;
};
struct AccessTopoDrrPassType : public std::monostate {
  using std::monostate::monostate;
};

using DrrPassTypeImpl = std::
    variant<AbstractDrrPassType, ReifiedDrrPassType, AccessTopoDrrPassType>;

struct DrrPassType : public DrrPassTypeImpl {
  using DrrPassTypeImpl::DrrPassTypeImpl;
  ADT_DEFINE_VARIANT_METHODS(DrrPassTypeImpl);
};

}  // namespace ap::drr
