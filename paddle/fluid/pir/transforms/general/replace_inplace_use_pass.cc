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

#include "paddle/fluid/pir/transforms/general/replace_inplace_use_pass.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

class ReplaceInplaceUsePass : public pir::Pass {
 public:
  ReplaceInplaceUsePass() : pir::Pass("replace_inplace_use_pass", 1) {}

  void Run(pir::Operation* module_op) override {
    int64_t num_rewrites_{0};
    std::unordered_map<pir::Value, pir::Value> replace_map;
    std::unordered_map<pir::Value, pir::Value> reverse_replace_map;

    for (size_t i = 0; i < module_op->num_regions(); ++i) {
      auto& region = module_op->region(i);
      for (auto& block : region) {
        for (auto& op : block) {
          for (size_t i = 0; i < op.num_operands(); ++i) {
            if (replace_map.count(op.operand_source(i))) {
              op.operand(i).set_source(replace_map[op.operand_source(i)]);
              // replace_map[op.operand_source(i)] =
              // inplace_map(op.result(inplace_id_map(i)));
            }
          }

          if (op.attributes().count("is_inplace") != 0 &&
              op.attributes()
                  .at("is_inplace")
                  .dyn_cast<pir::BoolAttribute>()
                  .data()) {
            if (op.num_results() < 1) return;
            pir::IrContext* ctx = pir::IrContext::Instance();
            std::string op_name = op.name();
            if (op.attributes().count("op_name")) {
              op_name = op.attributes()
                            .at("op_name")
                            .dyn_cast<pir::StrAttribute>()
                            .AsString();
            }

            pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
            paddle::dialect::OpYamlInfoParser yaml_parser(
                op_info
                    .GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
                    ->get_op_info_(op_name),
                paddle::dialect::IsLegacyOp(op_name));

            for (size_t i = 0; i < op.num_results(); ++i) {
              pir::Value value = op.result(i);
              // if (!IsInvalid(value)) {
              //   VLOG(8) << "Number " << i << " result of " << op_name
              //           << " is not invalid, so skip build a variable.";
              //   continue;
              // }
              std::string value_name = yaml_parser.OutputNames()[i];
              if (yaml_parser.HasInplace(value_name)) {
                const std::string& inplace_name =
                    yaml_parser.InplaceName(value_name);
                pir::Value inplace_value = op.operand_source(
                    yaml_parser.InputName2Id().at(inplace_name));
                if (reverse_replace_map.count(inplace_value)) {
                  replace_map[reverse_replace_map[inplace_value]] = value;
                  reverse_replace_map[value] =
                      reverse_replace_map[inplace_value];
                } else {
                  replace_map[inplace_value] = value;
                  reverse_replace_map[value] = inplace_value;
                }
              }
            }
          }
        }
      }
    }
    AddStatistics(num_rewrites_);
  }
};

namespace pir {

std::unique_ptr<pir::Pass> CreateReplaceInplaceUsePass() {
  return std::make_unique<ReplaceInplaceUsePass>();
}

}  // namespace pir

REGISTER_IR_PASS(replace_inplace_use_pass, ReplaceInplaceUsePass);
