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

#include "paddle/fluid/ir/dialect/pd_kernel_op.h"

namespace paddle {
namespace dialect {

const char *PhiKernelOp::attributes_name[attributes_num] = {
    "base_op", "infermeta_fn", "kernel_fn"};

void PhiKernelOp::Verify(const std::vector<ir::OpResult> &inputs,
                         const std::vector<ir::Type> &outputs,
                         const ir::AttributeMap &attributes) {
  VLOG(4) << "Verifying inputs, outputs and attributes for: SetParameterOp.";
  // Verify inputs type:

  // Verify if attributes contain attribute name in attributes_name:
  //   if (!attributes.at("parameter_name").isa<StrAttribute>()) {
  //     throw("Type of attribute: parameter_name is not right.");
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PhiKernelOp)
