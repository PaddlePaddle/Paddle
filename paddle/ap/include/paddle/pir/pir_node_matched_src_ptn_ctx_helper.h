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
#include "paddle/ap/include/drr/drr_ctx.h"
#include "paddle/ap/include/drr/source_pattern_ctx.h"
#include "paddle/ap/include/ir_match/graph_match_ctx.h"
#include "paddle/ap/include/paddle/pir_node.h"
#include "paddle/ap/include/reified_drr/matched_src_ptn_ctx_helper.h"

namespace pir {

class Block;

}

namespace ap::paddle {

struct PirNodeMatchedSrcPtnCtxHelper
    : public reified_drr::MatchedSrcPtnCtxHelper {
  PirNodeMatchedSrcPtnCtxHelper(
      const drr::SourcePatternCtx& src_ptn_ctx,
      const ir_match::GraphMatchCtx<PirNode>& match_ctx)
      : src_ptn_ctx_(src_ptn_ctx), match_ctx_(match_ctx) {}

  virtual drr::SourcePatternCtx src_ptn_ctx() { return src_ptn_ctx_; }

  adt::Result<std::shared_ptr<reified_drr::MatchedSrcPtnCtxHelper>>
  MakeInnerMatchedSrcPtnCtxHelper(
      const drr::PackedIrOp<drr::Node>& drr_packed_ir_op) override;

  adt::Result<adt::Ok> VisitNativeIrOpAttr(
      const drr::NativeIrOp<drr::Node>& drr_native_ir_op,
      const std::function<adt::Result<adt::Ok>(const std::string& attr_name,
                                               const axpr::Value& attr_val)>&
          DoEachAttr) override;

  adt::Result<axpr::Value> GetNativeIrValueType(
      const drr::NativeIrValue<drr::Node>& native_ir_value) override;

 private:
  adt::Result<drr::SourcePatternCtx> ConvertBlockToSrcPtnCtx(
      pir::Block* block, const std::shared_ptr<drr::DrrCtxImpl>& drr_ctx);

  drr::SourcePatternCtx src_ptn_ctx_;
  ir_match::GraphMatchCtx<PirNode> match_ctx_;
};

}  // namespace ap::paddle
