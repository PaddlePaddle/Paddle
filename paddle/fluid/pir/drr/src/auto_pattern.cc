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

class AutoDrrPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string name_;
  DrrPatternContext drr_pattern_context_
 public:
  AutoDrrPattern(std::string name,
                 DrrPatternContext drr_pattern_context,
                 bool should_create_ctx)
    : name_(name), drr_pattern_context_(drr_pattern_context), should_create_ctx_(should_create_ctx) {}
  std::string name() const override { return name_; }

  std::string get_python_drr_context() {
      return drr_pattern_context_;
  }
};

template<typename AutoDrrPattern>
class AutoDrrPass : public pir::PatternRewritePass {
 public:
  AutoDrrPass()
      : pir::PatternRewritePass("auto_drr_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<AutoDrrPattern>(context));
    return ps;
  }
};