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

#pragma once

#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/op_base.h"

namespace ir {
class IR_API YieldOp : public Op<YieldOp> {
 public:
  using Op::Op;
  static const char *name() { return "cf.yield"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    std::vector<OpResult> &&inputs);
  void Verify() {}
};
}  // namespace ir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::YieldOp);
