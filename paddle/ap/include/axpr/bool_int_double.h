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

namespace ap::axpr {

using BoolIntDoubleImpl = std::variant<bool, int64_t, double>;

struct BoolIntDouble : public BoolIntDoubleImpl {
  using BoolIntDoubleImpl::BoolIntDoubleImpl;

  ADT_DEFINE_VARIANT_METHODS(BoolIntDoubleImpl);

  template <typename ValueT>
  static adt::Result<BoolIntDouble> CastFrom(const ValueT& value) {
    using RetT = adt::Result<BoolIntDouble>;
    return value.Match(
        [](bool c) -> RetT { return c; },
        [](int64_t c) -> RetT { return c; },
        [](double c) -> RetT { return c; },
        [](const auto&) -> RetT {
          return adt::errors::ValueError{"BoolIntDouble::CastFrom() failed."};
        });
  }
};

}  // namespace ap::axpr
