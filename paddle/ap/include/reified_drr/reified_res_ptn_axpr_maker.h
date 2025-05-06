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
#include "paddle/ap/include/drr/result_pattern_ctx.h"

namespace ap::reified_drr {

class ReifiedResPtnAxprMaker {
  drr::ResultPatternCtx res_ptn_ctx_;
  std::function<adt::Result<code_gen::CodeGenResult<axpr::Value>>(
      const std::string&)>
      CodeGenResult4FusedOpName_;

 public:
  ReifiedResPtnAxprMaker(
      const drr::ResultPatternCtx& res_ptn_ctx,
      const std::function<adt::Result<code_gen::CodeGenResult<axpr::Value>>(
          const std::string&)>& CodeGenResult4FusedOpName)
      : res_ptn_ctx_(res_ptn_ctx),
        CodeGenResult4FusedOpName_(CodeGenResult4FusedOpName) {}

  adt::Result<adt::Ok> GenAnfExprForResPtnCtxOps(axpr::LetVar* op_pattern_ctx);

  adt::Result<adt::Ok> GenAnfExprForResPtnCtxOpValueConnections(
      axpr::LetVar* op_pattern_ctx, axpr::LetVar* tensor_pattern_ctx);
};

}  // namespace ap::reified_drr
