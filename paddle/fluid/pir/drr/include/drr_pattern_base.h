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

#pragma once

#include <memory>
#include <string>

#include "paddle/pir/include/pass/pass.h"
#include "paddle/fluid/pir/drr/include/drr_match_context.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/include/drr_rewrite_pattern.h"

namespace pir {
class IrContext;
}

namespace paddle {
namespace drr {

class DrrPatternBase : public std::enable_shared_from_this<DrrPatternBase> {
 public:
  bool should_create_ctx_ = false;
  TEST_API static std::unique_ptr<DrrRewritePattern> Build(
      pir::IrContext* ir_context,
      const std::shared_ptr<DrrPatternBase>& drr_pattern);

  virtual ~DrrPatternBase() = default;

  // Define get_python_drr_context
  virtual DrrPatternContext GetPythonDrrContext() const {
    return DrrPatternContext();
  };

  // Define the drr pattern.
  virtual void operator()(drr::DrrPatternContext* ctx) const = 0;

  // Give the drr pattern name.
  virtual std::string name() const = 0;

  // Give the drr pattern benefit.
  // If you want to control the application order of multiple patterns within a
  // pass, you need to specify it. The larger the value, the earlier it is
  // applied, usually set to the number of operators in the source pattern, with
  // a default of 1.
  virtual uint32_t benefit() const { return 1; }
};

template <typename T, typename... Args>
auto Create(pir::IrContext* ir_context, Args&&... args) {
  return T::Build(ir_context, std::make_shared<T>(std::forward<Args>(args)...));
}

class AutoDrrPattern : public DrrPatternBase {
 private:
  const std::string name_;
  DrrPatternContext drr_pattern_context_;
 public:
  AutoDrrPattern(const char* name,
                 DrrPatternContext& drr_pattern_context)
    : name_(std::string(name)), drr_pattern_context_(drr_pattern_context) {
      should_create_ctx_ = true;
    }
  std::string name() const override { return name_; }

  virtual void operator()(drr::DrrPatternContext* ctx) const override{
      return;
  }

  DrrPatternContext GetPythonDrrContext() const override {
      return drr_pattern_context_;
  }
};

template<typename AutoDrrPattern>
class AutoDrrPass : public pir::PatternRewritePass {
 public:
  const char* name_;
  DrrPatternContext pattern_context_;

  AutoDrrPass(const char* name,
              DrrPatternContext& pattern_context)
      : pir::PatternRewritePass(name, 2), name_(name), pattern_context_(pattern_context) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<AutoDrrPattern, const char*, DrrPatternContext&>(context, std::move(name_), pattern_context_));
    return ps;
  }
};

}  // namespace drr
}  // namespace paddle
