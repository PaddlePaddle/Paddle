// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

/*
Custom VJP stands for manually implemented backward rules for composite
operators. CustomVjpTrait will be added for those composite operators that
defines custom vjp rules. Finally, by calling has_custom_vjp(op), users can
check whether an operator has a CustomVjpTrait, and thus check whether a custom
vjp rule is defined for that operator.
*/

#pragma once

#include "paddle/pir/include/core/op_base.h"

namespace paddle {
namespace dialect {
class CustomVjpTrait : public pir::OpTraitBase<CustomVjpTrait> {
 public:
  explicit CustomVjpTrait(const pir::Operation *op)
      : pir::OpTraitBase<CustomVjpTrait>(op) {}
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::CustomVjpTrait)
