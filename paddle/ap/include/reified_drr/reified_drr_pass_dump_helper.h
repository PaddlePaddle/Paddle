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
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/code_gen/code_gen_result.h"
#include "paddle/ap/include/code_module/code_module.h"
#include "paddle/ap/include/drr/drr_ctx.h"
#include "paddle/ap/include/reified_drr/drr_node_attr_to_anf_expr_helper.h"
#include "paddle/ap/include/reified_drr/matched_src_ptn_ctx_helper.h"

namespace ap::reified_drr {

struct ReifiedDrrPassDumpHelper {
  bool DumpEnabled();

  // Returns reified drr_pass_class lambda
  adt::Result<axpr::AnfExpr> Dump(
      const drr::DrrCtx& abstract_drr_ctx,
      DrrNodeAttrToAnfExprHelper* attr2axpr_helper,
      MatchedSrcPtnCtxHelper* src_ptn_ctx_helper,
      const std::function<adt::Result<code_gen::CodeGenResult<axpr::Value>>(
          const std::string&)>& CodeGenResult4FusedOpName,
      int64_t nice) const;
};

}  // namespace ap::reified_drr
