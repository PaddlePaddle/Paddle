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
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/type_name.h"

#define OVERLOAD_PREFER_LAYOUT(op)                          \
  template <>                                               \
  common::DataLayout PreferLayoutImpl<op>(pir::Operation*); \
  extern template common::DataLayout PreferLayoutImpl<op>(pir::Operation*);

#define OVERLOAD_REWRITE_BY_LAYOUT(op)                               \
  template <>                                                        \
  void RewriteByLayoutImpl<op>(pir::Operation*, common::DataLayout); \
  extern template void RewriteByLayoutImpl<op>(pir::Operation*,      \
                                               common::DataLayout);

#define OVERLOAD_RELEVANT_INPUTS(op)                                   \
  template <>                                                          \
  std::vector<pir::Value> RelevantInputsImpl<op>(pir::Operation * op); \
  extern template std::vector<pir::Value> RelevantInputsImpl<op>(      \
      pir::Operation * op);

#define OVERLOAD_RELEVANT_OUTPUTS(op)                                   \
  template <>                                                           \
  std::vector<pir::Value> RelevantOutputsImpl<op>(pir::Operation * op); \
  extern template std::vector<pir::Value> RelevantOutputsImpl<op>(      \
      pir::Operation * op);

#define OVERLOAD_CAN_BE_MODIFIED(op)               \
  template <>                                      \
  bool CanBeModifiedImpl<op>(pir::Operation * op); \
  extern template bool CanBeModifiedImpl<op>(pir::Operation * op);

namespace paddle {
namespace dialect {

template <typename ConcreteOp>
common::DataLayout PreferLayoutImpl(pir::Operation* op) {
  auto data_format_attr = op->attribute<pir::StrAttribute>("data_format");
  if (!data_format_attr) {
    return common::DataLayout::ALL_LAYOUT;
  }
  return common::StringToDataLayout(data_format_attr.AsString());
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

template <typename ConcreteOp>
bool CanBeModifiedImpl(pir::Operation* op) {
  auto data_format_attr = op->attribute<pir::StrAttribute>("data_format");
  if (!data_format_attr) {
    return true;
  }
  auto cur_layout = common::StringToDataLayout(data_format_attr.AsString());
  auto prefer_layout = PreferLayoutImpl<ConcreteOp>(op);
  return cur_layout != prefer_layout;
}

class FusedConv2dAddActOp;
OVERLOAD_PREFER_LAYOUT(FusedConv2dAddActOp);
OVERLOAD_REWRITE_BY_LAYOUT(FusedConv2dAddActOp);
OVERLOAD_CAN_BE_MODIFIED(FusedConv2dAddActOp);

class Conv2dOp;
OVERLOAD_PREFER_LAYOUT(Conv2dOp);
OVERLOAD_CAN_BE_MODIFIED(Conv2dOp);
OVERLOAD_REWRITE_BY_LAYOUT(Conv2dOp);

class Conv2dTransposeOp;
OVERLOAD_PREFER_LAYOUT(Conv2dTransposeOp);

class GroupNormOp;
OVERLOAD_REWRITE_BY_LAYOUT(GroupNormOp);
OVERLOAD_RELEVANT_INPUTS(GroupNormOp);
OVERLOAD_RELEVANT_OUTPUTS(GroupNormOp);

class AddGroupNormSiluOp;
OVERLOAD_REWRITE_BY_LAYOUT(AddGroupNormSiluOp);
OVERLOAD_PREFER_LAYOUT(AddGroupNormSiluOp);
OVERLOAD_RELEVANT_INPUTS(AddGroupNormSiluOp);
OVERLOAD_RELEVANT_OUTPUTS(AddGroupNormSiluOp);

class ReshapeOp;
OVERLOAD_RELEVANT_INPUTS(ReshapeOp);
OVERLOAD_RELEVANT_OUTPUTS(ReshapeOp);
OVERLOAD_CAN_BE_MODIFIED(ReshapeOp);

class SqueezeOp;
OVERLOAD_REWRITE_BY_LAYOUT(SqueezeOp);
OVERLOAD_RELEVANT_INPUTS(SqueezeOp);
OVERLOAD_RELEVANT_OUTPUTS(SqueezeOp);
OVERLOAD_CAN_BE_MODIFIED(SqueezeOp);

class SiluOp;
OVERLOAD_REWRITE_BY_LAYOUT(SiluOp);

class AddOp;
OVERLOAD_REWRITE_BY_LAYOUT(AddOp);
OVERLOAD_CAN_BE_MODIFIED(AddOp);

class CastOp;
OVERLOAD_REWRITE_BY_LAYOUT(CastOp);

class ConcatOp;
OVERLOAD_REWRITE_BY_LAYOUT(ConcatOp);
OVERLOAD_RELEVANT_INPUTS(ConcatOp);

class ArgmaxOp;
OVERLOAD_REWRITE_BY_LAYOUT(ArgmaxOp);

class Pool2dOp;
OVERLOAD_RELEVANT_INPUTS(Pool2dOp);
OVERLOAD_REWRITE_BY_LAYOUT(Pool2dOp);

class MultiplyOp;
OVERLOAD_REWRITE_BY_LAYOUT(MultiplyOp);

class AssignOp;
OVERLOAD_REWRITE_BY_LAYOUT(AssignOp);

class SwishOp;
OVERLOAD_REWRITE_BY_LAYOUT(SwishOp);

}  // namespace dialect
}  // namespace paddle

namespace pir {
class CombineOp;
}

namespace paddle {
namespace dialect {

OVERLOAD_REWRITE_BY_LAYOUT(::pir::CombineOp);
}  // namespace dialect
}  // namespace paddle
