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
#include <vector>
#include "ThreadPool.h"

#include "paddle/fluid/operators/data/pipeline.h"

namespace paddle {
namespace operators {

using BlockDesc = framework::BlockDesc;
using Scope = framework::Scope;

using LoDTensor = framework::LoDTensor;
using LoDTensorBlockingQueue = paddle::operators::reader::LoDTensorBlockingQueue

namespace data {

class Pipeline {
  public:
    Pipeline(const BlockDesc &global_block, int64_t start_op_index,
             int64_t end_op_index, int64_t program_id,
             const std::vector<std::string> &output_var_names,
             size_t prefetch_queue_size = 2);

  private:
    inline size_t PrefetchCap() { return prefetch_queue_.Cap(); }

    inline size_t PrefetchSize() { return prefetch_queue_.Size(); }

    inline void Pipeline::Close() {
      VLOD(1) << "Pipeline close";
      prefetch_queue_.Close();
      closed_ = true;
    }

    inline bool IsClosed() { return closed_; }

    bool Reset();

		void copy_tensor(const framework::LoDTensor &lod_tensor,
										 framework::LoDTensor *out) const {
      if (lod_tensor.numel() == 0) return;
      auto &out_tensor = *out;
      TensorCopy(lod_tensor, lod_tensor.place(), &out_tensor);
      out_tensor.set_lod(lod_tensor.lod());
    }

    void CheckOutputVarStatus(const Variable &var, const std::string &var_name);

    void ReadNext(std::vector<Variable *> &out_vars);

    std::shared_ptr<BlockDesc> global_block_;
    Scope scope_;
    int64_t start_op_index_;
    int64_t end_op_index_;
    int64_t program_id_;

    std::vector<std::string> output_var_names_;

    platform::Place place_;

    ThreadPool thread_pool_;
    const size_t prefetch_queue_size_;
    const std::shared_ptr<LoDTensorBlockingQueue> prefetch_queue_;
    bool closed_{false};

};

class PipelineManager {
  // PipelineManager is a signleton manager for Pipeline, we
  // create single Pipeline for a program id
  private:
    DISABLE_COPY_AND_ASSIGN(PipelineManager);

    static PipelineManager* pm_instance_ptr_{nullptr};
    std::map<int64_t, Pipeline> prog_id_to_pipeline_;

  public:
    static PipelineManager& Instance() {
      if (pm_instance_ptr_ == nullptr) {
        pm_instance_ptr_ = new PipelineManager();
      }
      return *pm_instance_ptr_;
    }

    std::shared_ptr<Pipeline> GetPipeline(
        int64_t program_id, const BlockDesc &global_block,
        const platform::Place &place, int64_t start_op_index,
        int64_t end_op_index,
        const std::vector<std::string> &output_var_names,
        size_t prefetch_queue_size = 2) {
      auto iter = prog_id_to_pipeline_.find(program_id);
      if (iter != prog_id_to_pipeline_.end()) {
        auto* pipeline = new Pipeline(global_block, place, 
                                      start_op_index,
                                      end_op_index,
                                      program_id,
                                      output_var_names,
                                      prefetch_queue_size);
        prog_id_to_pipeline_.insert(std::pair<int64_t, Pipeline>(program_id, *pipeline));
        return std::make_shared<Pipeline>(pipeline);
      } else {
        reutrn std::make_shared<Pipeline>(&iter.second);
      }

    }
};


}  // data
}  // namespace operators
}  // namespace paddle
