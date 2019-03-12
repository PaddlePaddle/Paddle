// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/op_lite/op_lite.h"
#include "paddle/fluid/platform/device_context.h"

DECLARE_bool(global_with_gpu);
DECLARE_bool(global_use_lite_op);

namespace paddle {
namespace framework {

/*
 * Simple, intuitive and effective. Only single thread is supported, and
 * currently designed for inference.
 */
class NaiveExecutor {
 public:
  // Either one should be set, and will be executed.
  struct Gear {
    std::unique_ptr<OperatorBase> op;
    std::unique_ptr<inference::op_lite::OpLite> lite_op;
    std::string name;
    double timer{0.}; // ms
    size_t run_times{0};
  };

  explicit NaiveExecutor(const platform::Place& place) : place_(place) {}

  ~NaiveExecutor() {
      for (auto& gear : gears_) {
          LOG(INFO) << "gear " << gear.name << " ave latency " << gear.timer/gear.run_times;
      }
  }

  // Create child scope.
  // Create variables.
  // @with_feed_fetch_ops: whether to work with the feed and fetch operators.
  void Prepare(Scope* scope, const ProgramDesc& program_desc, int block_id,
               bool with_feed_fetch_ops);

  // Create variables before head.
  // Create parameters if persistable is ture, or create the temporary variables
  // instead.
  void CreateVariables(const ProgramDesc& desc, int block_id, bool persistable,
                       Scope* scope);

  // Run all the operators.
  void Run();

  // Get an tensor to operating directly, without the need for feed_ops.
  LoDTensor* FindTensor(const std::string& name);

  Scope* scope() { return scope_; }

 protected:
  void CreateOps(const ProgramDesc& desc, int block_id,
                 bool with_feed_fetch_ops, framework::Scope* scope = nullptr);

 private:
  bool shape_checked_{false};
  const platform::Place place_;
  // Catch the required resource to avoid recreate.
  std::vector<Gear> gears_;
  // The original implementation.
  std::vector<std::unique_ptr<OperatorBase>> ops_;
  Scope* scope_;
};

}  // namespace framework
}  // namespace paddle
