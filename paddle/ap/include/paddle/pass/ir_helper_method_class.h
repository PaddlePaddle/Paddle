// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/drr/drr_value_helper.h"
#include "paddle/ap/include/paddle/pass/ap_drr_helper.h"
#include "paddle/ap/include/paddle/pass/ap_generic_drr_pass.h"
#include "paddle/ap/include/paddle/pass/ir_helper.h"
#include "paddle/ap/include/paddle/pir/op_dialect.h"
#include "paddle/ap/include/paddle/pir/packed_ir_op_inner_source_pattern_helper.h"
#include "paddle/ap/include/paddle/pir/pass_manager_method_class.h"
#include "paddle/ap/include/paddle/pir/pass_method_class.h"
#include "paddle/ap/include/paddle/pir/program_method_class.h"
#include "paddle/ap/include/paddle/pir_node_helper.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace ap::paddle {

void ForceLinkIrTools();

}  // namespace ap::paddle
