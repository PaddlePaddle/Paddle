/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/inference/analysis/analysis_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Transform data from GPU to CPU
 */
class IrDataTransformPass : public AnalysisPass {
 public:
  void RunImpl(Argument *argument) override;

  std::string repr() const override { return "ir-data-transform-pass"; }

 private:
  bool NeedTransform(const framework::BlockDesc *block,
                     const framework::OpDesc *op, const std::string &var_name,
                     size_t index) const;

  bool IsSpecialOp(const std::vector<std::string> &special_ops,
                   std::string type) const;

  void InsertDataTransformOp(framework::BlockDesc *block, framework::OpDesc *op,
                             const std::string &input_name,
                             const std::string &var_name, size_t index);
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
