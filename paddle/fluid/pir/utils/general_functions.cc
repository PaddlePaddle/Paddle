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

#include "paddle/fluid/pir/utils/general_functions.h"

#include <unordered_set>

#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/op_operand.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

namespace {

void GetUsedExternalValueImpl(
    std::unordered_set<pir::Value>& defined_values,  // NOLINT
    std::vector<pir::Value>& used_values,            // NOLINT
    const pir::Operation& op) {
  for (size_t index = 0; index < op.num_operands(); ++index) {
    pir::Value value = op.operand_source(index);
    if (defined_values.find(value) == defined_values.end()) {
      used_values.push_back(value);
      defined_values.insert(value);
    }
  }
  for (auto& region : op) {
    for (auto& block : region) {
      for (auto value : block.args()) {
        defined_values.insert(value);
      }
      for (const auto& [_, value] : block.kwargs()) {
        defined_values.insert(value);
      }
    }
    for (auto& block : region) {
      for (auto& inner_op : block) {
        GetUsedExternalValueImpl(defined_values, used_values, inner_op);
      }
    }
  }
  for (size_t index = 0; index < op.num_results(); ++index) {
    defined_values.insert(op.result(index));
  }
}

}  // namespace

namespace pir {

std::string GetParameterNameFromValue(const pir::Value& value) {
  pir::Operation* owner = value.defining_op();
  std::string name;
  if (owner->isa<ParameterOp>()) {
    pir::ParameterOp op = owner->dyn_cast<pir::ParameterOp>();
    name = op.param_name();
  } else if (owner->isa<ConstantTensorOp>()) {
    pir::ConstantTensorOp op = owner->dyn_cast<pir::ConstantTensorOp>();
    name = op.tensor_name();
  } else {
    PADDLE_THROW(
        common::errors::Unimplemented("Value must be a weight from a Parameter "
                                      "or a ConstantTensorOp op."));
  }
  return name;
}

std::vector<int64_t> GetShapeFromValue(const pir::Value& value) {
  if (value.type().isa<paddle::dialect::DenseTensorType>()) {
    return phi::vectorize(
        value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims());
  } else if (value.type().isa<paddle::dialect::SelectedRowsType>()) {
    return phi::vectorize(
        value.type().dyn_cast<paddle::dialect::SelectedRowsType>().dims());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, we can only get shape for dense_tensor or selected_rows."));
  }
}

pir::Type GetDataTypeFromValue(const pir::Value& value) {
  // TODO(dev): Support other types like DenseTensor.
  PADDLE_ENFORCE_EQ(value.type().isa<paddle::dialect::DenseTensorType>(),
                    true,
                    common::errors::InvalidArgument(
                        "Value's type must be a DenseTensorType."));
  return value.type().dyn_cast<paddle::dialect::DenseTensorType>().dtype();
}

Operation* GetDefiningOpForInput(const Operation* op, uint32_t index) {
  PADDLE_ENFORCE_EQ(
      index < op->num_operands() && op->operand_source(index),
      true,
      common::errors::InvalidArgument("Intput operand's index must be valid."));
  return op->operand_source(index).defining_op();
}

std::vector<std::pair<Operation*, int32_t>> GetUseOpsForOutput(
    const Operation* op, uint32_t index) {
  PADDLE_ENFORCE_EQ(index < op->num_results(),
                    true,
                    common::errors::InvalidArgument(
                        "Output op result's index must be valid."));
  auto result = op->result(index);
  std::vector<std::pair<Operation*, int32_t>> use_ops;
  for (auto it = result.use_begin(); it != result.use_end(); ++it) {
    use_ops.emplace_back(it->owner(), it->index());
  }
  return use_ops;
}

std::vector<pir::Value> GetUsedExternalValue(const pir::Operation& op) {
  std::unordered_set<pir::Value> defined_values{nullptr};
  std::vector<pir::Value> used_values;
  GetUsedExternalValueImpl(defined_values, used_values, op);
  return used_values;
}

std::vector<pir::Value> GetUsedExternalValue(const pir::Block& block) {
  auto& args = block.args();
  std::unordered_set<pir::Value> defined_values(args.begin(), args.end());
  std::vector<pir::Value> used_values;
  for (auto& op : block) {
    GetUsedExternalValueImpl(defined_values, used_values, op);
  }
  return used_values;
}

bool ValueIsPersistable(const pir::Value& value) {
  if (!value.defining_op()) {
    return false;
  }
  if (value.defining_op()->num_operands() > 0) {
    for (const auto& source_value : value.defining_op()->operands_source()) {
      if (!ValueIsPersistable(source_value)) {
        return false;
      }
    }
  } else {
    if (!value.defining_op()->isa<pir::ParameterOp>() &&
        !value.defining_op()->isa<paddle::dialect::FullOp>() &&
        !value.defining_op()->isa<paddle::dialect::FullIntArrayOp>()) {
      return false;
    }
  }
  return true;
}

}  // namespace pir
