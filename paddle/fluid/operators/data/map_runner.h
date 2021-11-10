/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
#include <mutex>
#include <vector>
#include "ThreadPool.h"

#include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/operators/data/data_scope.h"

namespace paddle {
namespace operators {

using BlockDesc = framework::BlockDesc;
using Scope = framework::Scope;
using ParallelExecutor = framework::ParallelExecutor;

using Variable = framework::Variable;
using LoDTensor = framework::LoDTensor;
using LoDTensorBlockingQueue = operators::reader::LoDTensorBlockingQueue;

namespace data {

class MapRunner {
 public:
  MapRunner(const std::shared_ptr<BlockDesc> global_block,
           const platform::Place &place, int64_t start_op_index,
           int64_t end_op_index, int64_t program_id,
           const std::vector<std::string> &input_var_names,
           const std::vector<std::string> &output_var_names,
           const std::vector<std::shared_ptr<LoDTensorBlockingQueue>> input_queues,
           const std::vector<std::shared_ptr<LoDTensorBlockingQueue>> output_queues);

  // ~MapRunner() {
  //   VLOG(1) << "~MapRunner";
  //   Close();
  // }

  void Shutdown();

  inline bool IsRunning() { return running_; }


 private:
  void copy_tensor(const framework::LoDTensor &lod_tensor,
                   framework::LoDTensor *out) const {
    if (lod_tensor.numel() == 0) return;
    auto &out_tensor = *out;
    TensorCopy(lod_tensor, lod_tensor.place(), &out_tensor);
    out_tensor.set_lod(lod_tensor.lod());
  }

  bool ShareInputsIntoScope();
  void StartMapThread(std::shared_ptr<ParallelExecutor> executor,
                           const std::vector<std::string> &skip_vars);

  void CheckInputVarStatus(const Variable &var, const std::string &var_name);
  void CheckOutputVarStatus(const Variable &var, const std::string &var_name);

  ThreadPool thread_pool_;
  std::atomic<bool> running_;

  Scope scope_;
  std::shared_ptr<BlockDesc> global_block_;
  platform::Place place_;
  int64_t start_op_index_;
  int64_t end_op_index_;
  int64_t program_id_;

  std::vector<std::string> input_var_names_;
  std::vector<std::string> output_var_names_;
  std::vector<std::shared_ptr<LoDTensorBlockingQueue>> input_queues_;
  std::vector<std::shared_ptr<LoDTensorBlockingQueue>> output_queues_;
};

class MapRunnerManager {
  // MapRunnerManager is a signleton manager for MapRunner, we
  // create single MapRunner for a program id
 private:
  DISABLE_COPY_AND_ASSIGN(MapRunnerManager);

  static MapRunnerManager *pm_instance_ptr_;
  static std::mutex m_;

  std::map<int64_t, std::unique_ptr<MapRunner>> prog_id_to_runner_;

 public:
  static MapRunnerManager *Instance() {
    if (pm_instance_ptr_ == nullptr) {
      std::lock_guard<std::mutex> lk(m_);
      if (pm_instance_ptr_ == nullptr) {
        pm_instance_ptr_ = new MapRunnerManager;
      }
    }
    return pm_instance_ptr_;
  }

  void StartMapRunner(
      int64_t program_id, BlockDesc *global_block, const platform::Place &place,
      int64_t start_op_index, int64_t end_op_index,
      const std::vector<std::string> &input_var_names,
      const std::vector<std::string> &output_var_names,
      const std::vector<std::shared_ptr<LoDTensorBlockingQueue>> &input_queues,
      const std::vector<std::shared_ptr<LoDTensorBlockingQueue>> &output_queues) {
    auto iter = prog_id_to_runner_.find(program_id);
    if (iter == prog_id_to_runner_.end()) {
      prog_id_to_runner_[program_id] = std::unique_ptr<MapRunner>(new MapRunner(
          std::shared_ptr<BlockDesc>(global_block), place, start_op_index,
          end_op_index, program_id, input_var_names, output_var_names,
          input_queues, output_queues));
      }
  }

  void ShutdownMapRunner(int program_id) {
    auto iter = prog_id_to_runner_.find(program_id);
    if (iter != prog_id_to_runner_.end()) {
      iter->second.get()->Shutdown();
      prog_id_to_runner_.erase(iter);
    }
  }

  MapRunnerManager() { VLOG(1) << "MapRunnerManager init"; }

  ~MapRunnerManager() {
    VLOG(1) << "~MapRunnerManager";
    prog_id_to_runner_.clear();
  }
};

}  // data
}  // namespace operators
}  // namespace paddle
