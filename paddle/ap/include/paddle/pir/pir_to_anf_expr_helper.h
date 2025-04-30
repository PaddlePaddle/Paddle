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
#include "paddle/ap/include/paddle/pir/attr_adt_type_id.h"
#include "paddle/ap/include/paddle/pir/type_adt_type_id.h"
#include "paddle/ap/include/reified_drr/drr_node_attr_to_anf_expr_helper.h"

namespace ap::paddle {

struct PirToAnfExprHelper : public reified_drr::DrrNodeAttrToAnfExprHelper {
  adt::Result<axpr::AnfExpr> ConvertTypeToAnfExpr(axpr::LetContext* ctx,
                                                  axpr::Value type) override;
  adt::Result<axpr::AnfExpr> ConvertAttrToAnfExpr(axpr::LetContext* ctx,
                                                  axpr::Value attr) override;

  adt::Result<axpr::AnfExpr> ConvertPirTypeToAnfExpr(axpr::LetContext* ctx,
                                                     pir::Type type);
  adt::Result<axpr::AnfExpr> ConvertPirAttrToAnfExpr(axpr::LetContext* ctx,
                                                     pir::Attribute attr);
};

}  // namespace ap::paddle
