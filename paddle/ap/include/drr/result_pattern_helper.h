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

#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/atomic.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/drr/res_ptn_packed_ir_op_declare_data.h"
#include "paddle/ap/include/drr/value.h"

namespace ap::drr {

struct ResultPatternHelper {
  using DrrNode = drr::Node;
  using DrrGraphNode = graph::Node<DrrNode>;

  using DrrNativeIrValue = drr::NativeIrValue<DrrNode>;
  using DrrPackedIrValue = drr::PackedIrValue<DrrNode>;
  using DrrIrValue = drr::IrValue;

  using DrrNativeIrOp = drr::NativeIrOp<DrrNode>;
  using DrrNativeIrOpOperand = drr::NativeIrOpOperand<DrrNode>;
  using DrrNativeIrOpResult = drr::NativeIrOpResult<DrrNode>;
  using DrrPackedIrOp = drr::PackedIrOp<DrrNode>;
  using DrrPackedIrOpOperand = drr::PackedIrOpOperand<DrrNode>;
  using DrrPackedIrOpResult = drr::PackedIrOpResult<DrrNode>;
  using DrrOptPackedIrOp = drr::OptPackedIrOp<DrrNode>;
  using DrrOptPackedIrOpOperand = drr::OptPackedIrOpOperand<DrrNode>;
  using DrrOptPackedIrOpResult = drr::OptPackedIrOpResult<DrrNode>;

  const DrrCtx& drr_ctx;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnInputIrValueByResPtnIrOp(
      const DrrPackedIrOp& res_ptn_ir_op, const DoEachT& DoEach) const {
    return VisitResPtnInputIrValueByResPtnIrOpImpl(res_ptn_ir_op, DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnInputIrValueByResPtnIrOp(
      const DrrNativeIrOp& res_ptn_ir_op, const DoEachT& DoEach) const {
    return VisitResPtnInputIrValueByResPtnIrOpImpl(res_ptn_ir_op, DoEach);
  }

  template <typename IrOpT, typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnInputIrValueByResPtnIrOpImpl(
      const IrOpT& res_ptn_ir_op, const DoEachT& DoEach) const {
    auto VisitOpOperand =
        [&](const DrrGraphNode& op_operand) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(op_operand_downstreams, op_operand.UpstreamNodes());
      ADT_LET_CONST_REF(ir_value_node, op_operand_downstreams.Sole());
      ADT_LET_CONST_REF(ir_value, ir_value_node.Get());
      const auto& opt_drr_ir_value = DrrIrValue::OptCastFrom(ir_value);
      ADT_CHECK(opt_drr_ir_value.has_value());
      const auto& drr_ir_value = opt_drr_ir_value.value();
      return DoEach(drr_ir_value);
    };
    ADT_LET_CONST_REF(upstreams, res_ptn_ir_op->node.UpstreamNodes());
    ADT_RETURN_IF_ERR(upstreams.VisitNodes(VisitOpOperand));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnOutputIrValueByResPtnIrOp(
      const DrrPackedIrOp& res_ptn_ir_op, const DoEachT& DoEach) const {
    return VisitResPtnOutputIrValueByResPtnIrOpImpl(res_ptn_ir_op, DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnOutputIrValueByResPtnIrOp(
      const DrrNativeIrOp& res_ptn_ir_op, const DoEachT& DoEach) const {
    return VisitResPtnOutputIrValueByResPtnIrOpImpl(res_ptn_ir_op, DoEach);
  }

  template <typename IrOpT, typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnOutputIrValueByResPtnIrOpImpl(
      const IrOpT& res_ptn_ir_op, const DoEachT& DoEach) const {
    auto VisitOpResult =
        [&](const DrrGraphNode& op_result) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(op_result_downstreams, op_result.DownstreamNodes());
      ADT_LET_CONST_REF(ir_node, op_result_downstreams.Sole());
      ADT_LET_CONST_REF(drr_ir_node, ir_node.Get());
      const auto& opt_drr_ir_value = DrrIrValue::OptCastFrom(drr_ir_node);
      ADT_CHECK(opt_drr_ir_value.has_value());
      const auto& drr_ir_value = opt_drr_ir_value.value();
      return DoEach(drr_ir_value);
    };
    ADT_LET_CONST_REF(downstreams, res_ptn_ir_op->node.DownstreamNodes());
    ADT_RETURN_IF_ERR(downstreams.VisitNodes(VisitOpResult));
    return adt::Ok{};
  }

  std::optional<DrrIrValue> SrcPtnIrValue4ResPtnIrValue(
      const DrrIrValue& res_ptn_ir_value) const {
    const auto& opt_src_ptn_ctx = drr_ctx->GetSourcePatternCtx();
    if (opt_src_ptn_ctx.HasError()) {
      return std::nullopt;
    }
    const auto& src_ptn_ctx = opt_src_ptn_ctx.GetOkValue();
    const auto& map = src_ptn_ctx->tensor_pattern_ctx->uid2ir_value;
    auto GetSrcPtnIrValue =
        [&](const auto& ir_value) -> std::optional<DrrIrValue> {
      const auto iter = map.find(ir_value->name);
      if (iter == map.end()) {
        return std::nullopt;
      }
      return iter->second;
    };
    return res_ptn_ir_value.Match(
        [&](const DrrNativeIrValue& ir_value) -> std::optional<DrrIrValue> {
          return GetSrcPtnIrValue(ir_value);
        },
        [&](const DrrPackedIrValue& ir_value) -> std::optional<DrrIrValue> {
          return GetSrcPtnIrValue(ir_value);
        });
  }
};

}  // namespace ap::drr
