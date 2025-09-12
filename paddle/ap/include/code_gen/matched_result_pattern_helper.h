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
#include "paddle/ap/include/code_gen/arg_source_ctx.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/drr/res_ptn_packed_ir_op_declare_data.h"
#include "paddle/ap/include/drr/result_pattern_helper.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/graph/graph_helper.h"
#include "paddle/ap/include/ir_match/graph_match_ctx.h"
#include "paddle/ap/include/ir_match/graph_matcher.h"
#include "paddle/ap/include/ir_match/ir_match_ctx.h"

namespace ap::code_gen {

template <typename BirNode /* backend ir node*/>
struct MatchedResultPatternHelper {
  using DrrNode = drr::Node;

  using DrrCtx = drr::DrrCtx;

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

  using DrrIrOpImpl = std::variant<DrrNativeIrOp, DrrPackedIrOp>;

  using IrMatchCtx = ir_match::IrMatchCtx<BirNode>;

  using GraphMatchCtx = ir_match::GraphMatchCtx<BirNode>;

  const GraphMatchCtx& match_ctx_;
  const DrrCtx& drr_ctx_;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMatchedBirInputOfRestPtnPackedIrOp(
      const DrrPackedIrOp& res_ptn_ir_op, const DoEachT& DoEach) const {
    auto CollectInput =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      return VisitMatchedBirValueOfResPtnIrValue(drr_ir_value, DoEach);
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, CollectInput));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMatchedBirOutputOfRestPtnPackedIrOp(
      const DrrPackedIrOp& res_ptn_ir_op, const DoEachT& DoEach) const {
    auto DoEachDrrIrValue =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      return VisitMatchedBirValueOfResPtnIrValue(drr_ir_value, DoEach);
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, DoEachDrrIrValue));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnInputIrValueByResPtnIrOp(
      const DrrPackedIrOp& res_ptn_ir_op, const DoEachT& DoEach) const {
    drr::ResultPatternHelper helper{drr_ctx_};
    return helper.VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMatchedBirValueOfResPtnIrValue(
      const DrrIrValue& res_ptn_ir_value, const DoEachT& DoEach) const {
    using Ok = adt::Result<adt::Ok>;
    const auto& opt_ir_value = SrcPtnIrValue4ResPtnIrValue(res_ptn_ir_value);
    ADT_CHECK(opt_ir_value.has_value());
    const auto& ir_value = opt_ir_value.value();
    ADT_RETURN_IF_ERR(
        match_ctx_->VisitBigGraphIrValueNode(ir_value.node(), DoEach));
    return adt::Ok{};
  }

  template <typename DoEachIndexT, typename DoEachSliceT>
  adt::Result<adt::Ok> VisitApKernelInputIndexOrSlice(
      const DrrPackedIrOp& res_ptn_ir_op,
      const DoEachIndexT& DoEachIndex,
      const DoEachSliceT& DoEachSlice) const {
    std::size_t start = 0;
    using Ok = adt::Result<adt::Ok>;
    auto DoEachIrValue = [&](const DrrIrValue& drr_ir_value) -> Ok {
      ADT_LET_CONST_REF(num_ir_values, GetResPtnNumBirValues(drr_ir_value));
      ADT_RETURN_IF_ERR(drr_ir_value.Match(
          [&](const DrrNativeIrValue&) -> Ok {
            ADT_RETURN_IF_ERR(DoEachIndex(start));
            return adt::Ok{};
          },
          [&](const DrrPackedIrValue&) -> Ok {
            ADT_RETURN_IF_ERR(DoEachSlice(start, start + num_ir_values));
            return adt::Ok{};
          }));
      start += num_ir_values;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, DoEachIrValue));
    return adt::Ok{};
  }

  template <typename DoEachIndexT, typename DoEachSliceT>
  adt::Result<adt::Ok> VisitApKernelOutputIndexOrSlice(
      const DrrPackedIrOp& res_ptn_ir_op,
      const DoEachIndexT& DoEachIndex,
      const DoEachSliceT& DoEachSlice) const {
    std::size_t start = 0;
    using Ok = adt::Result<adt::Ok>;
    auto DoEachIrValue = [&](const DrrIrValue& drr_ir_value) -> Ok {
      ADT_LET_CONST_REF(num_ir_values, GetResPtnNumBirValues(drr_ir_value));
      ADT_RETURN_IF_ERR(drr_ir_value.Match(
          [&](const DrrNativeIrValue&) -> Ok {
            ADT_RETURN_IF_ERR(DoEachIndex(start));
            return adt::Ok{};
          },
          [&](const DrrPackedIrValue&) -> Ok {
            ADT_RETURN_IF_ERR(DoEachSlice(start, start + num_ir_values));
            return adt::Ok{};
          }));
      start += num_ir_values;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, DoEachIrValue));
    return adt::Ok{};
  }

  adt::Result<std::size_t> GetApKernelNumOutputs(
      const DrrPackedIrOp& res_ptn_ir_op) const {
    std::size_t num_outputs = 0;
    auto AccNumOutputs =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(num_ir_values, GetResPtnNumBirValues(drr_ir_value));
      num_outputs += num_ir_values;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, AccNumOutputs));
    return num_outputs;
  }

  template <typename T, typename IrOpT, typename DoEachT>
  adt::Result<adt::Ok> VisitEachMatchedDrrIrValueAndOutputSlice(
      const std::vector<T>& output_values,
      const IrOpT& res_ptn_ir_op,
      const DoEachT& DoEach) const {
    std::size_t offset = 0;
    auto DoEachSlice =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(num_ir_values, GetResPtnNumBirValues(drr_ir_value));
      ADT_CHECK(offset + num_ir_values <= output_values.size());
      std::vector<T> slice{output_values.begin() + offset,
                           output_values.begin() + offset + num_ir_values};
      ADT_RETURN_IF_ERR(DoEach(drr_ir_value, slice));
      offset += num_ir_values;
      return adt::Ok{};
    };
    return VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, DoEachSlice);
  }

  template <typename IrOpT, typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnOutputIrValueByResPtnIrOp(
      const IrOpT& res_ptn_ir_op, const DoEachT& DoEach) const {
    drr::ResultPatternHelper helper{drr_ctx_};
    return helper.VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, DoEach);
  }

  adt::Result<std::size_t> GetResPtnNumBirValues(
      const DrrIrValue& res_ptn_ir_value) const {
    const auto& opt_src_ptn_ir_value =
        SrcPtnIrValue4ResPtnIrValue(res_ptn_ir_value);
    if (!opt_src_ptn_ir_value.has_value()) {
      // internal ir value in result pattern.
      return 1U;
    }
    return match_ctx_->GetNumBigGraphIrValueNodes(
        opt_src_ptn_ir_value.value().node());
  }

  std::optional<DrrIrValue> SrcPtnIrValue4ResPtnIrValue(
      const DrrIrValue& res_ptn_ir_value) const {
    drr::ResultPatternHelper helper{drr_ctx_};
    return helper.SrcPtnIrValue4ResPtnIrValue(res_ptn_ir_value);
  }

  using BirNativeIrValue = typename BirNode::native_value_type;

  adt::Result<BirNativeIrValue> CastToBirNativeIrValue(
      const BirNode& bir_node) const {
    using RetT = adt::Result<BirNativeIrValue>;
    return bir_node.Match(
        [&](const typename BirNode::native_value_type& bir_value) -> RetT {
          return bir_value;
        },
        [&](const typename BirNode::ref_value_type& ref_value) -> RetT {
          return ref_value.GetOwnerNativeIrValue();
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              "bir_node is not an PirNode::native_value_type or "
              "BirNode::ref_value_type"};
        });
  }
};

}  // namespace ap::code_gen
