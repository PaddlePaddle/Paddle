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

namespace test {
///
/// \brief TestRegionOp
///
class RegionOp : public ir::Op<RegionOp> {
 public:
  using Op::Op;
  static const char *name() { return "test.region"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(ir::Builder &builder,              // NOLINT
                    ir::OperationArgument &argument);  // NOLINT
  void Verify() const;
};

///
/// \brief TestBranchOp
///
class BranchOp : public ir::Op<BranchOp> {
 public:
  using Op::Op;
  static const char *name() { return "test.branch"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    const std::vector<ir::OpResult> &target_operands,
                    ir::Block *target);
  void Verify() const;
};

}  // namespace test

IR_DECLARE_EXPLICIT_TYPE_ID(test::RegionOp)
IR_DECLARE_EXPLICIT_TYPE_ID(test::BranchOp)
