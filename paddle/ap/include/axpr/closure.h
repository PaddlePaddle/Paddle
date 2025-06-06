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

#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/atomic.h"
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/axpr/error.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
class Environment;

template <typename ValueT>
struct ClosureImpl {
  Lambda<CoreExpr> lambda;
  std::shared_ptr<Environment<ValueT>> environment;

  bool operator==(const ClosureImpl& other) const {
    return other.lambda == this->lambda &&
           other.environment == this->environment;
  }
};

template <typename ValueT>
ADT_DEFINE_RC(Closure, const ClosureImpl<ValueT>);

template <typename ValueT>
struct TypeImpl<Closure<ValueT>> : public std::monostate {
  using value_type = Closure<ValueT>;

  const char* Name() const { return "closure"; }
};

}  // namespace ap::axpr
