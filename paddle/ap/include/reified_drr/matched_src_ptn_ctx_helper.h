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
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/drr_ctx.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/source_pattern_ctx.h"

namespace ap::reified_drr {

struct MatchedSrcPtnCtxHelper {
  virtual ~MatchedSrcPtnCtxHelper() {}

  virtual drr::SourcePatternCtx src_ptn_ctx() = 0;

  virtual adt::Result<std::shared_ptr<MatchedSrcPtnCtxHelper>>
  MakeInnerMatchedSrcPtnCtxHelper(
      const drr::PackedIrOp<drr::Node>& packed_ir_op) = 0;

  virtual adt::Result<adt::Ok> VisitNativeIrOpAttr(
      const drr::NativeIrOp<drr::Node>& native_ir_op,
      const std::function<adt::Result<adt::Ok>(const std::string& attr_name,
                                               const axpr::Value& attr_val)>&
          DoEachAttr) = 0;

  virtual adt::Result<axpr::Value> GetNativeIrValueType(
      const drr::NativeIrValue<drr::Node>& native_ir_value) = 0;
};

}  // namespace ap::reified_drr
