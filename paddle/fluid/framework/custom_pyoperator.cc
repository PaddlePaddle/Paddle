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

#include <utility>  // for std::move

#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/framework/custom_pyoperator.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/phi/api/ext/op_meta_info.h"

namespace paddle::framework {

void RegisterCustomPyOp(
    const std::string& op_name,
    std::vector<std::string>&& op_inputs,
    std::vector<std::string>&& op_outputs,
    std::vector<std::string>&& op_attrs,
    std::unordered_map<std::string, std::string>&& op_inplace_map,
    WrapPythonFunction&& pyop_func,
    WrapInferMetaPythonFunction&& pyop_func_infer_meta) {
  ::paddle::OpMetaInfoBuilder op_meta_info_builder =
      ::paddle::OpMetaInfoBuilder(std::string(op_name), 0);
  op_meta_info_builder.Inputs(std::move(op_inputs))
      .Outputs(std::move(op_outputs))
      .Attrs(std::move(op_attrs))
      .SetInplaceMap(std::move(op_inplace_map))
      .SetPyCustomPyOpFunction(pyop_func)
      .SetPyCustomPyOpInferMetaFunction(pyop_func_infer_meta);

  const std::vector<paddle::OpMetaInfo>& op_meta_info_vector =
      OpMetaInfoMap::Instance()[op_name];

  PADDLE_ENFORCE_EQ(op_meta_info_vector.size(),
                    1,
                    common::errors::OutOfRange(
                        "Current op_name(%s) must not be registered more "
                        "than one, because it don't support gradient op."));

  const auto& op_meta_info = op_meta_info_vector.back();

  auto& inplace_map = OpMetaInfoHelper::GetInplaceMap(op_meta_info);
  const auto postfix = inplace_map.empty() ? "" : "_";

  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  auto* custom_pyop_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::CustomPyOpDialect>();

  if (custom_pyop_dialect->HasRegistered(
          paddle::framework::kCustomPyDialectPrefix + op_name + postfix)) {
    return;
  }
  custom_pyop_dialect->RegisterCustomPyOp(op_meta_info);
}

}  // namespace paddle::framework
