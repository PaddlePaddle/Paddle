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
Custom VJP stands for manually implemented backward rules for complex operators.

For example, when custom vjp is not defined, complex operators such as softmax,
gelu, etc., are split into a set of primitive operators (such as add, multiply,
etc.) in the forward pass, and the backward pass consists of the backward
operators (such as add_grad, multiply_grad) of above primitive operators. And
those backward operators also consists of primitive operators, which may lead to
an increase of memory usage.

After implementing custom vjp manually, the backward pass of complex operators
can be split to several primitive operators directly by calling custom vjp
rules.
*/

#pragma once

#include "paddle/ir/core/op_base.h"

namespace paddle {
namespace dialect {
class CustomVjpTrait : public ir::OpTraitBase<CustomVjpTrait> {
 public:
  explicit CustomVjpTrait(ir::Operation *op)
      : ir::OpTraitBase<CustomVjpTrait>(op) {}
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::CustomVjpTrait)
