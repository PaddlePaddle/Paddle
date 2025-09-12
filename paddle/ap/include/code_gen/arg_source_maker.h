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
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/code_gen/arg_source_ctx.h"
#include "paddle/ap/include/code_gen/matched_result_pattern_helper.h"

namespace ap::code_gen {

template <typename BirNode>
struct ArgSourceMaker {
  const code_gen::MatchedResultPatternHelper<BirNode>& matched_res_ptn_helper;

  using DrrValue = drr::Value;
  using DrrNode = drr::Node;
  using DrrPackedIrOp = drr::PackedIrOp<DrrNode>;

  adt::Result<ArgSourceCtx<BirNode>> MakeArgSourceCtx(
      const DrrPackedIrOp& res_ptn_ir_op) const {
    ADT_LET_CONST_REF(input_and_tensor_source_pairs,
                      MakeInputAndTensorSourcePairs(res_ptn_ir_op));
    ADT_LET_CONST_REF(output_and_tensor_source_pairs,
                      MakeOutputAndTensorSourcePairs(res_ptn_ir_op));
    std::vector<std::pair<symbol::DimExpr, DimSource>>
        dim_expr_and_dim_source_pairs;
    ADT_RETURN_IF_ERR(CollectDimExprAndDimSourcePairs<InTensorSource>(
        &dim_expr_and_dim_source_pairs, input_and_tensor_source_pairs));
    ADT_RETURN_IF_ERR(CollectDimExprAndDimSourcePairs<OutTensorSource>(
        &dim_expr_and_dim_source_pairs, output_and_tensor_source_pairs));
    std::unordered_map<symbol::DimExpr, DimSource> dim_expr2dim_source{
        dim_expr_and_dim_source_pairs.begin(),
        dim_expr_and_dim_source_pairs.end()};
    return ArgSourceCtx<BirNode>{input_and_tensor_source_pairs,
                                 output_and_tensor_source_pairs,
                                 dim_expr_and_dim_source_pairs,
                                 dim_expr2dim_source};
  }

 private:
  adt::Result<std::vector<std::pair<BirNode, InTensorSource>>>
  MakeInputAndTensorSourcePairs(const DrrPackedIrOp& res_ptn_ir_op) const {
    using Ok = adt::Result<adt::Ok>;
    std::vector<BirNode> inputs;
    {
      auto CollectInput = [&](const BirNode& bir_node) -> Ok {
        inputs.emplace_back(bir_node);
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(
          matched_res_ptn_helper.VisitMatchedBirInputOfRestPtnPackedIrOp(
              res_ptn_ir_op, CollectInput));
    }
    using Pair = std::pair<BirNode, InTensorSource>;
    std::vector<Pair> ret;
    ret.reserve(inputs.size());
    {
      std::size_t input_idx = 0;
      auto DoEachIndex = [&](std::size_t index) -> Ok {
        ADT_CHECK(index < inputs.size());
        const auto& bir_node = inputs.at(index);
        NativeIrValueSource native_ir_value_source{input_idx};
        TensorSource tensor_source{native_ir_value_source};
        InTensorSource in_tensor_source{tensor_source};
        Pair pair{bir_node, in_tensor_source};
        ret.emplace_back(pair);
        ++input_idx;
        return adt::Ok{};
      };
      auto DoEachSlice = [&](std::size_t start, std::size_t end) -> Ok {
        for (std::size_t i = start; i < end; ++i) {
          ADT_CHECK(i < inputs.size());
          const auto& bir_node = inputs.at(i);
          PackedIrValueSource packed_ir_value_source{input_idx, i};
          TensorSource tensor_source{packed_ir_value_source};
          InTensorSource in_tensor_source{tensor_source};
          Pair pair{bir_node, in_tensor_source};
          ret.emplace_back(pair);
        }
        ++input_idx;
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(matched_res_ptn_helper.VisitApKernelInputIndexOrSlice(
          res_ptn_ir_op, DoEachIndex, DoEachSlice));
    }
    return ret;
  }

  adt::Result<std::vector<std::pair<BirNode, OutTensorSource>>>
  MakeOutputAndTensorSourcePairs(const DrrPackedIrOp& res_ptn_ir_op) const {
    using Ok = adt::Result<adt::Ok>;
    std::vector<BirNode> outputs;
    {
      auto CollectOutput = [&](const BirNode& bir_node) -> Ok {
        outputs.emplace_back(bir_node);
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(
          matched_res_ptn_helper.VisitMatchedBirOutputOfRestPtnPackedIrOp(
              res_ptn_ir_op, CollectOutput));
    }
    using Pair = std::pair<BirNode, OutTensorSource>;
    std::vector<Pair> ret;
    ret.reserve(outputs.size());
    {
      std::size_t output_idx = 0;
      auto DoEachIndex = [&](std::size_t index) -> Ok {
        ADT_CHECK(index < outputs.size());
        const auto& bir_node = outputs.at(index);
        OutTensorSource out_tensor_source{
            TensorSource{NativeIrValueSource{output_idx}}};
        ret.emplace_back(Pair{bir_node, out_tensor_source});
        ++output_idx;
        return adt::Ok{};
      };
      auto DoEachSlice = [&](std::size_t start, std::size_t end) -> Ok {
        for (std::size_t i = start; i < end; ++i) {
          ADT_CHECK(i < outputs.size());
          const auto& bir_node = outputs.at(i);
          OutTensorSource out_tensor_source{
              TensorSource{PackedIrValueSource{output_idx, i}}};
          ret.emplace_back(Pair{bir_node, out_tensor_source});
        }
        ++output_idx;
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(matched_res_ptn_helper.VisitApKernelOutputIndexOrSlice(
          res_ptn_ir_op, DoEachIndex, DoEachSlice));
    }
    return ret;
  }

  template <typename TensorSourceT>
  adt::Result<adt::Ok> CollectDimExprAndDimSourcePairs(
      std::vector<std::pair<symbol::DimExpr, DimSource>>*
          dim_expr_and_dim_source_pairs,
      const std::vector<std::pair<BirNode, TensorSourceT>>& tensor_and_sources)
      const {
    for (const auto& [bir_node, tensor_source] : tensor_and_sources) {
      ADT_LET_CONST_REF(
          bir_value, matched_res_ptn_helper.CastToBirNativeIrValue(bir_node));
      ADT_LET_CONST_REF(dim_exprs_ptr, bir_value.GetShapeDimExprsPtr());
      for (int i = 0; i < dim_exprs_ptr->size(); ++i) {
        const auto& dim_expr = dim_exprs_ptr->at(i);
        using Pair = std::pair<symbol::DimExpr, DimSource>;
        DimSource dim_source{ShapeDimSource{tensor_source, i}};
        Pair pair{dim_expr, dim_source};
        dim_expr_and_dim_source_pairs->emplace_back(pair);
      }
    }
    return adt::Ok{};
  }
};

}  // namespace ap::code_gen
