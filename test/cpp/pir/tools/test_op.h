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

#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/op_base.h"
#include "paddle/pir/core/operation_utils.h"
#include "test/cpp/pir/tools/test_interface.h"
#include "test/cpp/pir/tools/test_trait.h"

namespace test {
///
/// \brief TestRegionOp
///
class RegionOp : public pir::Op<RegionOp, OneRegionTrait> {
 public:
  using Op::Op;
  static const char *name() { return "test.region"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,              // NOLINT
                    pir::OperationArgument &argument);  // NOLINT
  void Verify() const {}
};

///
/// \brief TestBranchOp
///
class BranchOp : public pir::Op<BranchOp> {
 public:
  using Op::Op;
  static const char *name() { return "test.branch"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::OpResult> &target_operands,
                    pir::Block *target);
  void Verify() const;
};

// Define case op1.
class Operation1 : public pir::Op<Operation1> {
 public:
  using Op::Op;
  static const char *name() { return "test.operation1"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];   // NOLINT
  static void Build(pir::Builder &builder,              // NOLINT
                    pir::OperationArgument &argument);  // NOLINT
  void Verify() const;
};

// Define op2.
class Operation2
    : public pir::Op<Operation2, ReadOnlyTrait, InferShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "test.operation2"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;  // NOLINT
  static void Build(pir::Builder &builder,                  // NOLINT
                    pir::OperationArgument &argument) {}    // NOLINT
  void Verify() const {}
  static void InferShape() { VLOG(2) << "This is op2's InferShape interface."; }
};

}  // namespace test

IR_DECLARE_EXPLICIT_TYPE_ID(test::RegionOp)
IR_DECLARE_EXPLICIT_TYPE_ID(test::BranchOp)
IR_DECLARE_EXPLICIT_TYPE_ID(test::Operation1)
IR_DECLARE_EXPLICIT_TYPE_ID(test::Operation2)
