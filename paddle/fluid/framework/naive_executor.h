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

<<<<<<< HEAD
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
=======
#include <memory>
#include <string>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

<<<<<<< HEAD
=======
namespace phi {
class DenseTensor;
}  // namespace phi

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
namespace paddle {
namespace framework {

/*
 * Simple, intuitive and effective. Only single thread is supported, and
 * currently designed for inference.
 */
class ProgramDesc;
class Scope;

class NaiveExecutor {
 public:
<<<<<<< HEAD
  using HookFunc = std::function<void(OperatorBase*)>;

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  explicit NaiveExecutor(const platform::Place& place) : place_(place) {}

  ~NaiveExecutor();

  // Create child scope.
  // Create variables.
  // @with_feed_fetch_ops: whether to work with the feed and fetch operators.
  void Prepare(Scope* scope,
               const ProgramDesc& program_desc,
               int block_id,
               bool with_feed_fetch_ops);

  // Create variables before head.
<<<<<<< HEAD
  // Create parameters if persistable is true, or create the temporary variables
=======
  // Create parameters if persistable is ture, or create the temporary variables
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  // instead.
  void CreateVariables(const ProgramDesc& desc,
                       int block_id,
                       bool persistable,
                       Scope* scope);

  // Run all the operators.
  void Run();

  // Get an tensor to operating directly, without the need for feed_ops.
<<<<<<< HEAD
  phi::DenseTensor* FindTensor(const std::string& name);

  Scope* GetScope() { return scope_; }

  void MakeReusePlan(
      const std::unordered_map<std::string, std::string>& reuse_table);

  void ResetTrtOps(int num);

  void RegisterOutputHook(const HookFunc& hookfunc);

 private:
=======
  LoDTensor* FindTensor(const std::string& name);

  Scope* scope() { return scope_; }

  void CleanFeedFetchOps();

  void ResetTrtOps(int num);

 protected:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  void CreateOps(const ProgramDesc& desc,
                 int block_id,
                 bool with_feed_fetch_ops);

 private:
  const platform::Place place_;
  // Catch the required resource to avoid recreate.
  std::vector<std::unique_ptr<OperatorBase>> ops_;
<<<<<<< HEAD
  Scope* scope_{nullptr};

  std::vector<HookFunc> hookfunc_;

  // Record information that tensor_a should ShareBufferWith tensor_b.
  std::unordered_map<OperatorBase*, std::unordered_map<phi::DenseTensor*, int>>
      reuse_cache_;
  std::vector<phi::DenseTensor*> cluster_buffer_;
=======
  Scope* scope_;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
};

}  // namespace framework
}  // namespace paddle
