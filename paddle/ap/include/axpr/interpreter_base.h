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
#include "paddle/ap/include/axpr/frame.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/memory/circlable_ref_list_base.h"

namespace ap::axpr {

template <typename ValueT>
class Environment;

struct SerializableValue;

template <typename ValueT>
class InterpreterBase {
 public:
  virtual Result<ValueT> InterpretCall(const ValueT& func,
                                       const std::vector<ValueT>& args) = 0;

  virtual Result<ValueT> InterpretModule(
      const Frame<SerializableValue>& const_global_frame,
      const Lambda<CoreExpr>& lambda) = 0;

  virtual std::weak_ptr<ap::memory::CirclableRefListBase> circlable_ref_list()
      const = 0;

  virtual Result<adt::Ok> InterpretLambdaCall(
      const std::shared_ptr<Environment<ValueT>>& env,
      const ValueT& outer_func,
      const Lambda<CoreExpr>& lambda,
      const std::vector<ValueT>& args,
      ComposedCallImpl<ValueT>* ret_composed_call) = 0;
};

}  // namespace ap::axpr
