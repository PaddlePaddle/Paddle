// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/params_sync_among_devices_pass.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

#include "paddle/phi/core/errors.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/pass/pass.h"

namespace {

class ParamsSyncAmongDevicesPass : public pir::Pass {
 public:
  ParamsSyncAmongDevicesPass(const phi::Place& place,
                             paddle::framework::Scope* scope)
      : pir::Pass("params_sync_among_devices_pass", 0),
        place_(place),
        scope_(scope) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        phi::errors::PreconditionNotMet(
            "params_sync_among_devices_pass should run on module op."));
    auto* block = module_op.block();
    for (auto& op : *block) {
      if (op->attributes().count("op_name") == 0) {
        continue;
      }
      auto op_name = op->attributes()
                         .at("op_name")
                         .dyn_cast<pir::StrAttribute>()
                         .AsString();
      if (op_name == pir::GetParameterOp::name()) {
        auto use_op = pir::GetUseOpsForOutput(op, 0).front();
        phi::KernelKey kernel_key;
        if (use_op->attributes().count("kernel_key")) {
          kernel_key = use_op->attributes()
                           .at("kernel_key")
                           .dyn_cast<paddle::dialect::KernelAttribute>()
                           .data();
        }
        // TODO(liuyuanle): When the kernel_key doesn't exist？
        if (use_op->attributes().count("kernel_key") &&
            kernel_key.backend() != phi::Backend::CPU) {
          std::string param_name = op->attributes()
                                       .at("parameter_name")
                                       .dyn_cast<pir::StrAttribute>()
                                       .AsString();
          auto* param_var = scope_->FindVar(param_name);
          if (param_var->IsType<phi::DenseTensor>()) {
            auto* param_tensor = param_var->GetMutable<phi::DenseTensor>();
            paddle::platform::CPUPlace cpu_place;
            phi::DenseTensor temp_tensor;
            temp_tensor.Resize(param_tensor->dims());
            paddle::framework::TensorCopySync(
                *param_tensor, cpu_place, &temp_tensor);
            param_tensor->clear();
            paddle::framework::TensorCopySync(
                temp_tensor, place_, param_tensor);
          }
        }
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<::pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  phi::Place place_;
  paddle::framework::Scope* scope_{nullptr};
};

}  // namespace

namespace pir {

std::unique_ptr<pir::Pass> CreateParamsSyncAmongDevicesPass(
    const phi::Place& place, paddle::framework::Scope* scope) {
  return std::make_unique<ParamsSyncAmongDevicesPass>(place, scope);
}

}  // namespace pir
