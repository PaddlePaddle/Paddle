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

#include "paddle/common/enforce.h"
#include "paddle/common/layout.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/type_name.h"

namespace paddle {
namespace dialect {

template <typename ConcreteOp>
common::DataLayout PreferLayoutImpl(pir::Operation* op) {
  return common::DataLayout::ALL_LAYOUT;
}

template <typename ConcreteOp>
void RewriteByLayoutImpl(pir::Operation* op, common::DataLayout new_layout) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Op %s should have a specialized RewriteByLayout function",
      pir::get_type_name<ConcreteOp>()));
}

template <typename ConcreteOp>
std::vector<pir::Value> RelevantInputsImpl(pir::Operation* op) {
  return op->operands_source();
}

template <typename ConcreteOp>
std::vector<pir::Value> RelevantOutputsImpl(pir::Operation* op) {
  return op->results();
}

class FusedConv2dAddActOp;
template <>
common::DataLayout PreferLayoutImpl<FusedConv2dAddActOp>(pir::Operation*);
extern template common::DataLayout PreferLayoutImpl<FusedConv2dAddActOp>(
    pir::Operation*);
template <>
void RewriteByLayoutImpl<FusedConv2dAddActOp>(pir::Operation*,
                                              common::DataLayout);
extern template void RewriteByLayoutImpl<FusedConv2dAddActOp>(
    pir::Operation*, common::DataLayout);

}  // namespace dialect
}  // namespace paddle
