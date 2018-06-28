/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
extern void InitializeVariable(Variable* var, proto::VarType::Type var_type);

struct ExecutorPrepareContext {
  ExecutorPrepareContext(const framework::ProgramDesc& prog, size_t block_id);
  ~ExecutorPrepareContext();

  const framework::ProgramDesc& prog_;
  size_t block_id_;
  std::vector<std::unique_ptr<OperatorBase>> ops_;
};

class Executor {
 public:
  // TODO(dzhwinter) : Do not rely on this function, it will be removed
  explicit Executor(const platform::DeviceContext& device)
      : Executor(device.GetPlace()) {}

  explicit Executor(const platform::Place& place);

#ifdef PADDLE_WITH_DISTRIBUTE
  /*
   * Sending signal to pserver to mark current trainer stop.
   */
  void Complete();
#endif

  /* @Brief
   * Runtime evaluation of the given ProgramDesc under certain Scope
   *
   * @param
   *  ProgramDesc
   *  Scope
   */
  void Run(const ProgramDesc& prog, Scope* scope, int block_id,
           bool create_local_scope = true, bool create_vars = true);

  void Run(const ProgramDesc& program, Scope* scope,
           std::map<std::string, const LoDTensor*>* feed_targets,
           std::map<std::string, LoDTensor*>* fetch_targets,
           bool create_local_scope = true, bool create_vars = true,
           const std::string& feed_holder_name = "feed",
           const std::string& fetch_holder_name = "fetch");

  static std::unique_ptr<ExecutorPrepareContext> Prepare(
      const ProgramDesc& program, int block_id);

  static std::vector<std::shared_ptr<ExecutorPrepareContext>> Prepare(
      const ProgramDesc& program, const std::vector<int>& block_ids);

  void CreateVariables(const ProgramDesc& pdesc, Scope* scope, int block_id);

  void RunPreparedContext(ExecutorPrepareContext* ctx, Scope* scope,
                          bool create_local_scope = true,
                          bool create_vars = true, bool keep_kids = false);

  void RunPreparedContext(ExecutorPrepareContext* ctx, Scope* scope,
                          std::map<std::string, const LoDTensor*>* feed_targets,
                          std::map<std::string, LoDTensor*>* fetch_targets,
                          bool create_local_scope = true,
                          bool create_vars = true,
                          const std::string& feed_holder_name = "feed",
                          const std::string& fetch_holder_name = "fetch");

  void EnableMKLDNN(const ProgramDesc& program);

 private:
  const platform::Place place_;
};

}  // namespace framework
}  // namespace paddle
