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

#include "paddle/fluid/pir/transforms/general/params_sync_among_devices_pass.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"

namespace {

class ParamsSyncAmongDevicesPass : public pir::Pass {
 public:
  ParamsSyncAmongDevicesPass()
      : pir::Pass("params_sync_among_devices_pass", 0) {}

  bool Initialize(pir::IrContext* context) override {
    PADDLE_ENFORCE_EQ(
        Has(pir::Pass::kPlaceAttr),
        true,
        common::errors::InvalidArgument(
            "Pass initialize failed."
            "When using ConstantFoldingPass, place attribute is required!"
            "Use Set method to set the place attribute."));
    PADDLE_ENFORCE_EQ(
        Has(pir::Pass::kParamScopeAttr),
        true,
        common::errors::InvalidArgument(
            "Pass initialize failed."
            "When using ConstantFoldingPass, scope attribute is required!"
            "Use Set method to set the scope attribute."));

    place_ = Get<phi::Place>(pir::Pass::kPlaceAttr);
    scope_ = &Get<paddle::framework::Scope>(pir::Pass::kParamScopeAttr);
    return true;
  }

  void Run(pir::Operation* op) override {
    VLOG(6) << "apply params_sync_among_devices_pass";
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        common::errors::PreconditionNotMet(
            "params_sync_among_devices_pass should run on module op."));
    auto& block = module_op.block();
    int64_t num_rewrites_{0};

    std::vector<phi::DenseTensor*> dense_tensors;
    for (auto& inner_op : block) {
      if (inner_op.template isa<pir::ParameterOp>() &&
          inner_op.num_results() > 0) {
        auto var = inner_op.result(0);
        auto bool_attr =
            var.template attribute<::pir::BoolAttribute>(kAttrIsPersistable);
        if (!bool_attr || !bool_attr.data()) {
          continue;
        }
        std::string param_name = inner_op.attributes()
                                     .at("parameter_name")
                                     .template dyn_cast<pir::StrAttribute>()
                                     .AsString();
        auto* param_var = scope_->FindVar(param_name);
        PADDLE_ENFORCE_NOT_NULL(
            param_var,
            common::errors::InvalidArgument("Parameter var [%s] not in scope.",
                                            param_name));

        if (param_var->IsType<phi::DenseTensor>()) {
          dense_tensors.push_back(param_var->GetMutable<phi::DenseTensor>());
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "params_sync_among_devices_pass only support DenseTensor type of "
              "parameter var."));
        }
      }
    }
    num_rewrites_ = dense_tensors.size();

    size_t num_threads = 8;
    const size_t chunk_size =
        std::max(static_cast<size_t>(1), dense_tensors.size() / num_threads);
    num_threads = std::min(num_threads, dense_tensors.size() / chunk_size);
    size_t remain_size = dense_tensors.size() % num_threads;

    auto sync_handler = [&](const std::vector<phi::DenseTensor*>& tensors) {
      for (auto* tensor : tensors) {
        paddle::framework::TensorCopySync(*tensor, place_, tensor);
      }
    };

    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_threads; ++i) {
      auto start_it = dense_tensors.begin() + i * chunk_size;
      auto end_it = start_it + chunk_size;

      futures.push_back(
          std::async(std::launch::async,
                     sync_handler,
                     std::vector<phi::DenseTensor*>(start_it, end_it)));
    }
    if (remain_size > 0) {
      futures.push_back(std::async(
          std::launch::async,
          sync_handler,
          std::vector<phi::DenseTensor*>(
              dense_tensors.rbegin(), dense_tensors.rbegin() + remain_size)));
    }

    for (auto& future : futures) {
      future.wait();
    }
    AddStatistics(num_rewrites_);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    PADDLE_ENFORCE_NOT_NULL(
        scope_, common::errors::InvalidArgument("scope can not be nullptr"));
#ifdef PADDLE_WITH_XPU
    PADDLE_ENFORCE(phi::is_xpu_place(place_) || phi::is_cpu_place(place_),
                   common::errors::PreconditionNotMet(
                       "The Place attr in params_sync_among_devices_pass "
                       "should be cpu or xpu."));
#endif
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE(phi::is_gpu_place(place_) || phi::is_cpu_place(place_),
                   common::errors::PreconditionNotMet(
                       "The Place attr in params_sync_among_devices_pass "
                       "should be cpu or gpu."));
#endif
    if (phi::is_cpu_place(place_)) {
      return false;
    }
    return op->isa<::pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  phi::Place place_{phi::CPUPlace{}};
  paddle::framework::Scope* scope_{nullptr};
};

}  // namespace

namespace pir {

std::unique_ptr<pir::Pass> CreateParamsSyncAmongDevicesPass() {
  return std::make_unique<ParamsSyncAmongDevicesPass>();
}

}  // namespace pir
