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

namespace paddle {
namespace operators {

using BlockDesc = framework::BlockDesc;
using Scope = framework::Scope;
using ParallelExecutor = framework::ParallelExecutor;

using Variable = framework::Variable;
using LoDTensor = framework::LoDTensor;
using LoDTensorBlockingQueue = operators::reader::LoDTensorBlockingQueue;
using LoDTensorBlockingQueueHolder = operators::reader::LoDTensorBlockingQueueHolder;

namespace data {

class Pipeline {
 public:
  Pipeline(const std::shared_ptr<BlockDesc> global_block,
           const platform::Place &place, int64_t start_op_index,
           int64_t end_op_index, int64_t program_id,
           const std::vector<std::string> &output_var_names);
           // size_t prefetch_queue_size);

  ~Pipeline() { VLOG(1) << "~Pipeline"; }

  void ReadNext(std::vector<Variable *> &out_vars);

 private:

  void CheckOutputVarStatus(const Variable &var, const std::string &var_name);

  void copy_tensor(const framework::LoDTensor& lod_tensor,
                   framework::LoDTensor* out) const {
    if (lod_tensor.numel() == 0) return;
    auto& out_tensor = *out;
    TensorCopy(lod_tensor, lod_tensor.place(), &out_tensor);
    out_tensor.set_lod(lod_tensor.lod());
  }

  Scope scope_;
  std::shared_ptr<BlockDesc> global_block_;
  platform::Place place_;
  int64_t start_op_index_;
  int64_t end_op_index_;
  int64_t program_id_;

  std::vector<std::string> output_var_names_;

};

class PipelineManager {
  // PipelineManager is a signleton manager for Pipeline, we
  // create single Pipeline for a program id
 private:
  DISABLE_COPY_AND_ASSIGN(PipelineManager);

  static PipelineManager *pm_instance_ptr_;
  static std::mutex m_;

  std::map<int64_t, std::unique_ptr<Pipeline>> prog_id_to_pipeline_;

 public:
  static PipelineManager *Instance() {
    if (pm_instance_ptr_ == nullptr) {
      std::lock_guard<std::mutex> lk(m_);
      if (pm_instance_ptr_ == nullptr) {
        pm_instance_ptr_ = new PipelineManager;
      }
    }
    return pm_instance_ptr_;
  }

  Pipeline* GetPipeline(
      int64_t program_id, BlockDesc *global_block, const platform::Place &place,
      int64_t start_op_index, int64_t end_op_index,
      const std::vector<std::string> &output_var_names) {
    auto iter = prog_id_to_pipeline_.find(program_id);
    if (iter == prog_id_to_pipeline_.end()) {
      prog_id_to_pipeline_[program_id] = std::unique_ptr<Pipeline>(new Pipeline(
          std::shared_ptr<BlockDesc>(global_block), place, start_op_index,
          end_op_index, program_id, output_var_names));
      return prog_id_to_pipeline_[program_id].get();
    } else {
      return iter->second.get();
    }
  }

  void ShutDown() {
    prog_id_to_pipeline_.clear();
  }

  PipelineManager() { VLOG(1) << "PipelineManager init"; }

  ~PipelineManager() {
    VLOG(1) << "~PipelineManager";
    ShutDown();
  }
};

}  // data
}  // namespace operators
}  // namespace paddle
