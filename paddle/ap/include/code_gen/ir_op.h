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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/builtin_class_instance.h"

namespace ap::code_gen {

template <typename BirNode>
using IrOpImpl = std::variant<typename BirNode::native_op_type,
                              typename BirNode::packed_op_type,
                              typename BirNode::ref_op_type>;

template <typename BirNode>
struct IrOp : public IrOpImpl<BirNode> {
  using IrOpImpl<BirNode>::IrOpImpl;
  ADT_DEFINE_VARIANT_METHODS(IrOpImpl<BirNode>);

  template <typename ValueT>
  static adt::Result<IrOp> CastFrom(const ValueT& val) {
    ADT_LET_CONST_REF(
        instance, val.template CastTo<axpr::BuiltinClassInstance<ValueT>>());
    if (instance.template Has<typename BirNode::native_op_type>()) {
      ADT_LET_CONST_REF(
          ret, instance.template TryGet<typename BirNode::native_op_type>());
      return ret;
    }
    if (instance.template Has<typename BirNode::packed_op_type>()) {
      ADT_LET_CONST_REF(
          ret, instance.template TryGet<typename BirNode::packed_op_type>());
      return ret;
    }
    if (instance.template Has<typename BirNode::ref_op_type>()) {
      ADT_LET_CONST_REF(
          ret, instance.template TryGet<typename BirNode::ref_op_type>());
      return ret;
    }
    return adt::errors::ValueError{"IrOp::CastFrom failed."};
  }
};

}  // namespace ap::code_gen
