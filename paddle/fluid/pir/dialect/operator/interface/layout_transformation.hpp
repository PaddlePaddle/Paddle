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

#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/common/layout.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/type_name.h"

namespace paddle {
namespace dialect {

template <typename ConcreteOp>
common::DataLayout PreferLayoutImpl(pir::Operation* op) {
  return common::DataLayout::ALL_LAYOUT;
}

template <typename ConcreteOp>
common::DataLayout CurrentLayoutImpl(pir::Operation* op) {
  return common::DataLayout::UNDEFINED;
}

template <typename ConcreteOp>
void RewriteByLayoutImpl(pir::Operation* op, common::DataLayout new_layout) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Op %s should have a specialized RewriteByLayout function",
      pir::get_type_name<ConcreteOp>()));
}

template <typename ConcreteOp>
std::vector<pir::Value> RelevantInputsImpl(pir::Operation* op) {
  std::vector<pir::Value> relevant_inputs;
  for (auto& operand : op->operands_source()) {
    if (!operand || !operand.type()) continue;
    if (auto operand_type = operand.type().dyn_cast<pir::VectorType>()) {
      if (operand_type.size() == 0) continue;
    }
    relevant_inputs.push_back(operand);
  }
  return relevant_inputs;
}

template <typename ConcreteOp>
std::vector<pir::Value> RelevantOutputsImpl(pir::Operation* op) {
  std::vector<pir::Value> relevant_outputs;
  for (auto& result : op->results()) {
    if (!result || !result.type()) continue;
    if (auto result_type = result.type().dyn_cast<pir::VectorType>()) {
      if (result_type.size() == 0) continue;
    }
    relevant_outputs.push_back(result);
  }
  return relevant_outputs;
}

class FusedConv2dAddActOp;
template <>
common::DataLayout PreferLayoutImpl<FusedConv2dAddActOp>(pir::Operation*);
extern template common::DataLayout PreferLayoutImpl<FusedConv2dAddActOp>(
    pir::Operation*);
template <>
common::DataLayout CurrentLayoutImpl<FusedConv2dAddActOp>(pir::Operation*);
extern template common::DataLayout CurrentLayoutImpl<FusedConv2dAddActOp>(
    pir::Operation*);
template <>
void RewriteByLayoutImpl<FusedConv2dAddActOp>(pir::Operation*,
                                              common::DataLayout);
extern template void RewriteByLayoutImpl<FusedConv2dAddActOp>(
    pir::Operation*, common::DataLayout);

class GroupNormOp;
template <>
void RewriteByLayoutImpl<GroupNormOp>(pir::Operation* op,
                                      common::DataLayout new_layout);
extern template void RewriteByLayoutImpl<GroupNormOp>(
    pir::Operation* op, common::DataLayout new_layout);
template <>
std::vector<pir::Value> RelevantInputsImpl<GroupNormOp>(pir::Operation* op);
extern template std::vector<pir::Value> RelevantInputsImpl<GroupNormOp>(
    pir::Operation* op);
template <>
std::vector<pir::Value> RelevantOutputsImpl<GroupNormOp>(pir::Operation* op);
extern template std::vector<pir::Value> RelevantOutputsImpl<GroupNormOp>(
    pir::Operation* op);

class ReshapeOp;
template <>
void RewriteByLayoutImpl<ReshapeOp>(pir::Operation* op,
                                    common::DataLayout new_layout);
extern template void RewriteByLayoutImpl<ReshapeOp>(
    pir::Operation* op, common::DataLayout new_layout);
template <>
std::vector<pir::Value> RelevantInputsImpl<ReshapeOp>(pir::Operation* op);
extern template std::vector<pir::Value> RelevantInputsImpl<ReshapeOp>(
    pir::Operation* op);
template <>
std::vector<pir::Value> RelevantOutputsImpl<ReshapeOp>(pir::Operation* op);
extern template std::vector<pir::Value> RelevantOutputsImpl<ReshapeOp>(
    pir::Operation* op);

class SqueezeOp;
template <>
void RewriteByLayoutImpl<SqueezeOp>(pir::Operation* op,
                                    common::DataLayout new_layout);
extern template void RewriteByLayoutImpl<SqueezeOp>(
    pir::Operation* op, common::DataLayout new_layout);
template <>
std::vector<pir::Value> RelevantInputsImpl<SqueezeOp>(pir::Operation* op);
extern template std::vector<pir::Value> RelevantInputsImpl<SqueezeOp>(
    pir::Operation* op);
template <>
std::vector<pir::Value> RelevantOutputsImpl<SqueezeOp>(pir::Operation* op);
extern template std::vector<pir::Value> RelevantOutputsImpl<SqueezeOp>(
    pir::Operation* op);

}  // namespace dialect
}  // namespace paddle
