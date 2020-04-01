// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/fluid/imperative/backward_strategy.h"
#include "paddle/fluid/imperative/engine.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace imperative {

class VarBase;

class PartialGradEngine : public Engine {
 public:
  PartialGradEngine(const std::vector<std::shared_ptr<VarBase>> &input_targets,
                    const std::vector<std::shared_ptr<VarBase>> &output_targets,
                    const std::vector<std::shared_ptr<VarBase>> &output_grads,
                    const std::vector<std::shared_ptr<VarBase>> &no_grad_vars,
                    const platform::Place &place,
                    const detail::BackwardStrategy &strategy, bool create_graph,
                    bool retain_graph, bool allow_unused, bool only_inputs);

  void Execute() override;

  std::vector<std::shared_ptr<VarBase>> GetResult() const;

 private:
  void Clear();

 private:
  std::vector<std::shared_ptr<VarBase>> input_targets_;
  std::vector<std::shared_ptr<VarBase>> output_targets_;
  std::vector<std::shared_ptr<VarBase>> output_grads_;
  std::vector<std::shared_ptr<VarBase>> no_grad_vars_;
  platform::Place place_;
  detail::BackwardStrategy strategy_;
  bool create_graph_;
  bool retain_graph_;
  bool allow_unused_;
  bool only_inputs_;

  std::vector<std::shared_ptr<VarBase>> results_;
};

}  // namespace imperative
}  // namespace paddle
