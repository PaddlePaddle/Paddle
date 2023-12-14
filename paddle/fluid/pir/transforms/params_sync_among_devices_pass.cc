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
#include "paddle/fluid/platform/place.h"

#include "paddle/common/errors.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

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
        scope_(scope) {
    PADDLE_ENFORCE(
        paddle::platform::is_gpu_place(place_) ||
            paddle::platform::is_cpu_place(place_),
        phi::errors::PreconditionNotMet(
            "params_sync_among_devices_pass should run on cpu or gpu."));
  }

  void Run(pir::Operation* op) override {
    VLOG(6) << "apply ParamsSyncAmongDevicesPass";
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        phi::errors::PreconditionNotMet(
            "params_sync_among_devices_pass should run on module op."));
    auto& block = module_op.block();
    int64_t num_rewrites_{0};
    for (auto& inner_op : block) {
      if (inner_op.isa<pir::ParameterOp>()) {
        std::string param_name = inner_op.attributes()
                                     .at("parameter_name")
                                     .dyn_cast<pir::StrAttribute>()
                                     .AsString();
        auto* param_var = scope_->FindVar(param_name);
        PADDLE_ENFORCE_NOT_NULL(
            param_var,
            phi::errors::InvalidArgument("Parameter var [%s] not in scope.",
                                         param_name));
        if (param_var->IsType<phi::DenseTensor>()) {
          auto* param_tensor = param_var->GetMutable<phi::DenseTensor>();
          paddle::platform::CPUPlace cpu_place;
          phi::DenseTensor temp_tensor;
          temp_tensor.Resize(param_tensor->dims());
          paddle::framework::TensorCopySync(
              *param_tensor, cpu_place, &temp_tensor);
          param_tensor->clear();
          paddle::framework::TensorCopySync(temp_tensor, place_, param_tensor);
          num_rewrites_++;
        }
      }
    }
    PrintStatistics(num_rewrites_);
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
