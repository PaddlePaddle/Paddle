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
#include "test/cpp/ir/tools/test_op.h"

namespace test {
void RegionOp::Build(ir::Builder &builder, ir::OperationArgument &argument) {
  argument.num_regions = 1;
}
void RegionOp::Verify() const {
  auto num_regions = (*this)->num_regions();
  IR_ENFORCE(num_regions == 1u,
             "The region's number in Region Op must be 1, but current is %d",
             num_regions);
}

void BranchOp::Build(ir::Builder &builder,  // NOLINT
                     ir::OperationArgument &argument,
                     const std::vector<ir::OpResult> &target_operands,
                     ir::Block *target) {
  argument.AddOperands(target_operands.begin(), target_operands.end());
  argument.AddSuccessor(target);
}

void BranchOp::Verify() const {
  IR_ENFORCE((*this)->num_successors() == 1u,
             "successors number must equal to 1.");
  IR_ENFORCE((*this)->successor(0), "successor[0] can't be nullptr");
}

}  // namespace test

IR_DEFINE_EXPLICIT_TYPE_ID(test::RegionOp)
IR_DEFINE_EXPLICIT_TYPE_ID(test::BranchOp)
