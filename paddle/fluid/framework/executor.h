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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

class Dataset;
class ProgramDesc;
class Scope;
class TrainerBase;

struct ExecutorPrepareContext {
  ExecutorPrepareContext(const framework::ProgramDesc& prog, size_t block_id);

  ~ExecutorPrepareContext();

  void PrepareUnusedVars(const std::vector<std::string>& keep_vars,
                         bool force_disable_gc = false);

  const framework::ProgramDesc& prog_;
  const size_t block_id_;

  std::vector<std::unique_ptr<OperatorBase>> ops_;

  std::unordered_map<const OperatorBase*, std::vector<std::string>>
      unused_vars_;
  bool force_disable_gc_{false};
};

class Executor {
 public:
  // TODO(dzhwinter) : Do not rely on this function, it will be removed
  explicit Executor(const platform::DeviceContext& device)
      : Executor(device.GetPlace()) {}

  explicit Executor(const platform::Place& place);

  ~Executor();
  /*
   * Close this Executor.
   * Calling this method will send complete messages to all pserver instances.
   */
  void Close();

  /* @Brief
   * Runtime evaluation of the given ProgramDesc under certain Scope
   *
   * @param
   *  ProgramDesc
   *  Scope
   *  block_id
   *  create_local_scope
   *  create_vars
   *  skip_ref_cnt_vars
   *  force_disable_gc
   *  keep_kid_scopes
   */
  void Run(const ProgramDesc& prog, Scope* scope, int block_id,
           bool create_local_scope = true, bool create_vars = true,
           const std::vector<std::string>& skip_ref_cnt_vars =
               std::vector<std::string>(),
           bool force_disable_gc = false, bool keep_kid_scopes = false);

  // This API is very slow.
  void Run(const ProgramDesc& program, Scope* scope,
           std::map<std::string, const LoDTensor*>* feed_targets,
           std::map<std::string, FetchType*>* fetch_targets,
           bool create_local_scope = true, bool create_vars = true,
           const std::string& feed_holder_name = "feed",
           const std::string& fetch_holder_name = "fetch");

  // This API is very slow.
  void RunPreparedContext(ExecutorPrepareContext* ctx, Scope* scope,
                          std::map<std::string, const LoDTensor*>* feed_targets,
                          std::map<std::string, FetchType*>* fetch_targets,
                          bool create_local_scope = true,
                          bool create_vars = true,
                          const std::string& feed_holder_name = "feed",
                          const std::string& fetch_holder_name = "fetch");

  static std::unique_ptr<ExecutorPrepareContext> Prepare(
      const ProgramDesc& program, int block_id,
      const std::vector<std::string>& skip_ref_cnt_vars =
          std::vector<std::string>(),
      bool force_disable_gc = false);

  static std::vector<std::shared_ptr<ExecutorPrepareContext>> Prepare(
      const ProgramDesc& program, const std::vector<int>& block_ids,
      const std::vector<std::vector<std::string>>& skip_ref_cnt_vars =
          std::vector<std::vector<std::string>>(),
      bool force_disable_gc = false);

  void CreateVariables(const ProgramDesc& pdesc, Scope* scope, int block_id);

  void RunPartialPreparedContext(ExecutorPrepareContext* ctx, Scope* scope,
                                 int64_t start_op_index, int64_t end_op_index,
                                 bool create_local_scope = true,
                                 bool create_vars = true,
                                 bool keep_kids = false);

  void RunPreparedContext(ExecutorPrepareContext* ctx, Scope* scope,
                          bool create_local_scope = true,
                          bool create_vars = true, bool keep_kids = false);

  void EnableMKLDNN(const ProgramDesc& program);

  std::shared_ptr<TrainerBase> InitForDataset(
      const ProgramDesc& main_program, const std::string& trainer_desc_str,
      Scope* scope, Dataset* dataset);
  void RunFromDataset(std::shared_ptr<TrainerBase> trainer);

  void ReleaseTrainer(std::shared_ptr<TrainerBase> trainer);

  const platform::Place GetPlace() const { return place_; }

 private:
  const platform::Place place_;
};

}  // namespace framework
}  // namespace paddle
