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

#include "paddle/ir/builtin_op.h"
#include "paddle/ir/builtin_attribute.h"

namespace ir {
const char *GetParameterOp::attributes_name[attributes_num] = {
    "parameter_name"};

void GetParameterOp::verify(const std::vector<ir::OpResult> &inputs,
                            const std::vector<ir::Type> &outputs,
                            const ir::AttributeMap &attributes) {
  VLOG(4) << "Verifying inputs, outputs and attributes for: GetParameterOp.";
  // Verify inputs type:
  if (inputs.size() != 0) {
    throw("The size of inputs must be equal to 0.");
  }
  // Verify outputs type:
  if (outputs.size() != 1) {
    throw("The size of outputs must be equal to 1.");
  }
  // Verify if attributes contain attribute name in attributes_name:
  if (!attributes.at("parameter_name").isa<StrAttribute>()) {
    throw("Type of attribute: parameter_name is not right.");
  }
}

const char *SetParameterOp::attributes_name[attributes_num] = {
    "parameter_name"};

void SetParameterOp::verify(const std::vector<ir::OpResult> &inputs,
                            const std::vector<ir::Type> &outputs,
                            const ir::AttributeMap &attributes) {
  VLOG(4) << "Verifying inputs, outputs and attributes for: SetParameterOp.";
  // Verify inputs type:
  if (inputs.size() != 1) {
    throw("The size of inputs must be equal to 1.");
  }
  // Verify outputs type:
  if (outputs.size() != 0) {
    throw("The size of outputs must be equal to 0.");
  }
  // Verify if attributes contain attribute name in attributes_name:
  if (!attributes.at("parameter_name").isa<StrAttribute>()) {
    throw("Type of attribute: parameter_name is not right.");
  }
}

const char **CombineOp::attributes_name = nullptr;
void CombineOp::verify(const std::vector<ir::OpResult> &inputs,
                       const std::vector<ir::Type> &outputs,
                       const ir::AttributeMap &attributes) {}

const char *SliceOp::attributes_name[attributes_num] = {"index"};
void SliceOp::verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const ir::AttributeMap &attributes) {}

}  // namespace ir
