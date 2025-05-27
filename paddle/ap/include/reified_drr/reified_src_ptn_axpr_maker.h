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
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/drr/source_pattern_ctx.h"
#include "paddle/ap/include/reified_drr/drr_node_attr_to_anf_expr_helper.h"
#include "paddle/ap/include/reified_drr/matched_src_ptn_ctx_helper.h"

namespace ap::reified_drr {

class ReifiedSrcPtnAxprMaker {
 public:
  ReifiedSrcPtnAxprMaker(DrrNodeAttrToAnfExprHelper* anf_expr_helper,
                         MatchedSrcPtnCtxHelper* matched_src_ptn_ctx_helper)
      : anf_expr_helper_(anf_expr_helper),
        matched_src_ptn_ctx_helper_(matched_src_ptn_ctx_helper) {}

  adt::Result<adt::Ok> GenAnfExprForSrcPtnCtxOps(axpr::LetVar* op_pattern_ctx);

  adt::Result<adt::Ok> GenAnfExprForSrcPtnCtxValues(
      axpr::LetVar* tensor_pattern_ctx);

  adt::Result<adt::Ok> GenAnfExprForSrcPtnCtxOpValueConnections(
      axpr::LetVar* op_pattern_ctx, axpr::LetVar* tensor_pattern_ctx);

 private:
  DrrNodeAttrToAnfExprHelper* anf_expr_helper_;
  MatchedSrcPtnCtxHelper* matched_src_ptn_ctx_helper_;
};

}  // namespace ap::reified_drr
