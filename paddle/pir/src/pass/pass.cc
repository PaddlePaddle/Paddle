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

#include <glog/logging.h>

#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/verify.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_instrumentation.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/src/pass/pass_adaptor.h"

#include "paddle/common/enforce.h"

namespace pir {

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
Pass::~Pass() {
  for (const auto& attr : attrs_) {
    if (attr_dels_.find(attr.first) != attr_dels_.end()) {
      attr_dels_[attr.first]();
    }
  }
  attrs_.clear();
  attr_dels_.clear();
}

bool Pass::CanApplyOn(Operation* op) const { return op->num_regions() > 0; }

std::optional<detail::PassExecutionState>& Pass::pass_state() {
  return pass_state_;
}

void Pass::SignalPassFailure() {
  PADDLE_ENFORCE_EQ(pass_state_.has_value(),
                    true,
                    common::errors::InvalidArgument("pass state has no value"));
  pass_state_->pass_failed = true;
}

AnalysisManager Pass::analysis_manager() {
  PADDLE_ENFORCE_EQ(pass_state_.has_value(),
                    true,
                    common::errors::InvalidArgument("pass state has no value"));
  return pass_state_->am;
}
//===----------------------------------------------------------------------===//
// PatternRewritePass
//===----------------------------------------------------------------------===//
bool PatternRewritePass::Initialize(IrContext* context) {
  RewritePatternSet ps = InitializePatterns(context);
  PADDLE_ENFORCE_EQ(
      ps.Empty(),
      false,
      common::errors::InvalidArgument(
          "Pass creation failed."
          "When using PatternRewritePass to create a Pass, the number of "
          "customized Patterns is required to be greater than zero."
          "Suggested fix: Check whether Pattern is added to the "
          "InitializePatterns() function of class [%s]",
          name()));
  patterns_ = FrozenRewritePatternSet(std::move(ps));
  return true;
}

GreedyRewriteConfig PatternRewritePass::InitializeConfig() {
  GreedyRewriteConfig config;
  config.use_top_down_traversal = true;
  config.max_iterations = 10;
  return config;
}

void PatternRewritePass::Run(Operation* op) {
  VLOG(4) << "Run PatternRewritePass: " << name();
  GreedyRewriteConfig config = InitializeConfig();
  if (Has(kValueReplaceHookAttr)) {
    config.value_replaced_hook =
        Get<VALUE_REPLACED_HOOK_FUNC>(kValueReplaceHookAttr);
  }
  auto [_, num_rewrites] = ApplyPatternsGreedily(op, patterns_, config);
  AddStatistics(num_rewrites);
}

//----------------------------------------------------------------------------------------------//
// PassAdaptor
//----------------------------------------------------------------------------------------------//
void detail::PassAdaptor::Run(Operation* op, uint8_t opt_level, bool verify) {
  RunImpl(op, opt_level, verify);
}

void detail::PassAdaptor::RunImpl(Operation* op,
                                  uint8_t opt_level,
                                  bool verify) {
  auto last_am = analysis_manager();

  for (size_t i = 0; i < op->num_regions(); ++i) {
    auto& region = op->region(i);
    for (auto& block : region) {
      for (auto& op : block) {
        AnalysisManagerHolder am(&op, last_am.GetPassInstrumentor());
        if (!RunPipeline(*pm_, &op, am, opt_level, verify))
          return SignalPassFailure();
      }
    }
  }
  return;
}

bool detail::PassAdaptor::RunPipeline(const PassManager& pm,
                                      Operation* op,
                                      AnalysisManager am,
                                      uint8_t opt_level,
                                      bool verify) {
  auto* instrumentor = am.GetPassInstrumentor();
  if (instrumentor) {
    instrumentor->RunBeforePipeline(op);
  }

  for (auto& pass : pm.passes()) {
    if (pass->CanApplyOn(op)) {
      if (!RunPass(pass.get(), op, am, opt_level, verify)) {
        return false;
      }
    }
  }

  if (instrumentor) {
    instrumentor->RunAfterPipeline(op);
  }

  // Apply pass manager on all nested ir.
  if (!RunPass(pm.pass_adaptor_.get(), op, am, opt_level, verify)) {
    return false;
  }

  return true;
}

bool detail::PassAdaptor::RunPass(Pass* pass,
                                  Operation* op,
                                  AnalysisManager am,
                                  uint8_t opt_level,
                                  bool verify) {
  if (opt_level < pass->pass_info().opt_level) return true;

  pass->pass_state_ = PassExecutionState(op, am);

  PassInstrumentor* instrumentor = am.GetPassInstrumentor();

  if (auto* adaptor = dynamic_cast<PassAdaptor*>(pass)) {
    adaptor->Run(op, opt_level, verify);
  } else {
    if (instrumentor) instrumentor->RunBeforePass(pass, op);
    pass->Run(op);
    if (instrumentor) instrumentor->RunAfterPass(pass, op);
  }
  bool pass_failed = pass->pass_state()->pass_failed;

  if (!pass_failed && verify) {
    bool verify_recursively = !dynamic_cast<PassAdaptor*>(pass);
    pir::Verify(op, verify_recursively);
  }

  return !pass_failed;
}

//----------------------------------------------------------------------------------------------//
// PassManager
//----------------------------------------------------------------------------------------------//
PassManager::PassManager(IrContext* context, uint8_t opt_level)
    : context_(context), opt_level_(opt_level) {
  pass_adaptor_ = std::make_unique<detail::PassAdaptor>(this);
}

bool PassManager::Run(Program* program) {
  if (!Initialize(context_)) {
    return false;
  }
  return Run(program->module_op());
}

bool PassManager::Run(Operation* op) {
  // Construct a analysis manager for the pipeline.
  AnalysisManagerHolder am(op, instrumentor_.get());

  return detail::PassAdaptor::RunPipeline(*this, op, am, opt_level_, verify_);
}

bool PassManager::Initialize(IrContext* context) {
  for (auto& pass : passes()) {
    if (!pass->Initialize(context)) return false;
    if (value_replaced_hook_) {
      pass->SetNotOwned<VALUE_REPLACED_HOOK_FUNC>(Pass::kValueReplaceHookAttr,
                                                  &value_replaced_hook_);
    }
  }

  return true;
}

void PassManager::AddInstrumentation(std::unique_ptr<PassInstrumentation> pi) {
  if (!instrumentor_) instrumentor_ = std::make_unique<PassInstrumentor>();

  instrumentor_->AddInstrumentation(std::move(pi));
}

//----------------------------------------------------------------------------------------------//
// PassInstrumentor
//----------------------------------------------------------------------------------------------//
namespace detail {
struct PassInstrumentorImpl {
  // TODO(wilber): Support multi-thread.
  std::vector<std::unique_ptr<PassInstrumentation>> instrumentations;
};
}  // namespace detail

PassInstrumentor::PassInstrumentor()
    : impl_(new detail::PassInstrumentorImpl{}) {}

PassInstrumentor::~PassInstrumentor() = default;

void PassInstrumentor::RunBeforePipeline(Operation* op) {
  if (op->num_regions() == 0) return;
  for (auto& instr : impl_->instrumentations) {
    instr->RunBeforePipeline(op);
  }
}

void PassInstrumentor::RunAfterPipeline(Operation* op) {
  if (op->num_regions() == 0) return;
  for (auto it = impl_->instrumentations.rbegin();
       it != impl_->instrumentations.rend();
       ++it) {
    (*it)->RunAfterPipeline(op);
  }
}

void PassInstrumentor::RunBeforePass(Pass* pass, Operation* op) {
  if (op->num_regions() == 0) return;
  for (auto& instr : impl_->instrumentations) {
    instr->RunBeforePass(pass, op);
  }
}

void PassInstrumentor::RunAfterPass(Pass* pass, Operation* op) {
  if (op->num_regions() == 0) return;
  for (auto it = impl_->instrumentations.rbegin();
       it != impl_->instrumentations.rend();
       ++it) {
    (*it)->RunAfterPass(pass, op);
  }
}

void PassInstrumentor::RunBeforeAnalysis(const std::string& name,
                                         TypeId id,
                                         Operation* op) {
  if (op->num_regions() == 0) return;
  for (auto& instr : impl_->instrumentations) {
    instr->RunBeforeAnalysis(name, id, op);
  }
}

void PassInstrumentor::RunAfterAnalysis(const std::string& name,
                                        TypeId id,
                                        Operation* op) {
  if (op->num_regions() == 0) return;
  for (auto it = impl_->instrumentations.rbegin();
       it != impl_->instrumentations.rend();
       ++it) {
    (*it)->RunBeforeAnalysis(name, id, op);
  }
}

void PassInstrumentor::AddInstrumentation(
    std::unique_ptr<PassInstrumentation> pi) {
  impl_->instrumentations.emplace_back(std::move(pi));
}

}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::detail::PreservedAnalyses::AllAnalysesType)
