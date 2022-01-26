// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <random>
#include <utility>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace operators {
namespace data {

using Scope = framework::Scope;
using Variable = framework::Variable;
using BlockDesc = framework::BlockDesc;
using LoDTensor = framework::LoDTensor;
using LoDTensorArray = framework::LoDTensorArray;
using LoDTensorBlockingQueue = operators::reader::LoDTensorBlockingQueue;
using LoDTensorBlockingQueueHolder = operators::reader::LoDTensorBlockingQueueHolder;

class Sampler {
  public:
    explicit Sampler(const int64_t batch_size, const int64_t num_samples,
                     const bool shuffle, const bool drop_last,
                     const int rank, const int world_size)
                     : current_iter_(0),
                       batch_size_(batch_size),
                       num_samples_(num_samples),
                       drop_last_(drop_last),
                       rank_(rank),
                       world_size_(world_size) {
      sample_ids_.reserve(num_samples);
      for (int64_t i = 0; i < num_samples; i++) {
        sample_ids_.emplace_back(i);
      }
      if (shuffle) {
        rnd_.seed(time(0));
        std::shuffle(sample_ids_.begin(), sample_ids_.end(), rnd_);
      }
    }

    void GetNextIndices(std::vector<int64_t>* indices) {
      int64_t start_idx =
          batch_size_ * world_size_ * current_iter_ + rank_ * batch_size_;
      current_iter_++;

      if (start_idx >= num_samples_) return;
      if (drop_last_ && start_idx + batch_size_ >= num_samples_) return;

      int64_t batch_len = std::min(batch_size_, num_samples_ - start_idx);
      indices->reserve(batch_len);
      for (int64_t i = 0; i < batch_len; i++) {
        indices->emplace_back(sample_ids_[start_idx + i]);
      }
    }

  private:
    int64_t current_iter_;
    const int64_t batch_size_;
    const int64_t num_samples_;
    const bool drop_last_;
    const int rank_;
    const int world_size_;

    std::mt19937 rnd_;
    std::vector<int64_t> sample_ids_;
};

class DataReader {
 public:
  explicit DataReader(BlockDesc* reader_block,
                      const Scope* scope,
                      const platform::Place place,
                      const std::string &indices_var_name,
                      const std::vector<std::string> &output_var_names,
                      const std::vector<std::shared_ptr<LoDTensorBlockingQueue>> output_queues,
                      const int batch_size,
                      const int num_samples,
                      const bool shuffle,
                      const bool drop_last,
                      const int rank,
                      const int world_size)
                      : running_(true),
                        reader_block_(reader_block),
                        place_(place),
                        indices_var_name_(indices_var_name),
                        output_var_names_(output_var_names),
                        output_queues_(output_queues),
                        batch_size_(batch_size),
                        sampler_(batch_size, num_samples, shuffle,
                                 drop_last, rank, world_size) {
    StartReaderThread(scope);
  }

  void StartReaderThread(const Scope* scope) {
    if (reader_thread_.joinable()) {
      return;
    }

    reader_thread_ = std::thread([this, scope] {
      auto& scope_ = scope->NewScope();
      framework::Executor executor(place_);
      while (running_.load()) {
        std::vector<int64_t> indices;
        sampler_.GetNextIndices(&indices);
        // shutdown reader if indices drained
        if (indices.size() == 0) {
          for(auto& queue: output_queues_) {
            while (queue->Size()) sleep(0.5);
            queue->Close();
          }

          running_.store(false);
          return;
        }

        ShareIndicesIntoScope(&scope_, indices);

        try {
          executor.Run(*reader_block_->Program(), &scope_,
                       static_cast<int>(reader_block_->ID()),
                       false, true, {}, false, true);
        } catch (...) {
          break;
        }

        for (size_t i = 0; i < output_var_names_.size(); i++) {
          auto *out_var = scope_.FindVar(output_var_names_[i]);
          PADDLE_ENFORCE_NOT_NULL(
              out_var, platform::errors::NotFound(
                "The output variable %s is not found in DataReader "
                "program's internal scope", output_var_names_[i]));
          // CheckOutputVarStatus(*out_var, output_var_names_[i]);

          if (out_var->IsType<LoDTensor>()) {
            framework::LoDTensorArray t_arr(1);
            copy_tensor(out_var->Get<LoDTensor>(), &t_arr[0]);
            output_queues_[i]->Push(t_arr);
          } else {
            auto out_arr = out_var->Get<LoDTensorArray>();
            framework::LoDTensorArray t_arr(out_arr.size());
            for (size_t i = 0; i < out_arr.size(); i++) {
              copy_tensor(out_arr[i], &t_arr[i]);
            }
            output_queues_[i]->Push(t_arr);
          }
        }
      }
      scope->DeleteScope(&scope_);
    });
  }

  void ShutDown() {
    for(auto& queue: output_queues_) {
      while (queue->Size()) sleep(0.5);
      queue->Close();
    }

    running_.store(false);
    if (reader_thread_.joinable()) reader_thread_.join();
  }

  void ShareIndicesIntoScope(Scope* scope,
                             std::vector<int64_t> indices) {
    auto* var = scope->Var(indices_var_name_);

    auto* indices_tensor = var->GetMutable<LoDTensor>();
    indices_tensor->Resize(framework::make_ddim({static_cast<int64_t>(indices.size())}));
    auto* indices_data = indices_tensor->mutable_data<int64_t>(platform::CPUPlace());
    
    for (size_t i = 0; i < indices.size(); i++) {
      indices_data[i] = indices[i];
    }
  }

 private:
  std::atomic<bool> running_;
  std::thread reader_thread_;

  BlockDesc* reader_block_;
  platform::Place place_;

  std::string indices_var_name_;
  std::vector<std::string> output_var_names_;
  std::vector<std::shared_ptr<LoDTensorBlockingQueue>> output_queues_;

  const int64_t batch_size_;
  Sampler sampler_;

  void copy_tensor(const framework::LoDTensor& lod_tensor,
                   framework::LoDTensor* out) const {
    if (lod_tensor.numel() == 0) return;
    auto& out_tensor = *out;
    TensorCopy(lod_tensor, lod_tensor.place(), &out_tensor);
    out_tensor.set_lod(lod_tensor.lod());
  }
};


class ReaderManager {
 private:
  DISABLE_COPY_AND_ASSIGN(ReaderManager);

  static ReaderManager *rm_instance_ptr_;
  static std::mutex m_;

  std::map<int64_t, std::unique_ptr<DataReader>> id_to_reader_;

 public:
  static ReaderManager *Instance() {
    if (rm_instance_ptr_ == nullptr) {
      std::lock_guard<std::mutex> lk(m_);
      if (rm_instance_ptr_ == nullptr) {
        rm_instance_ptr_ = new ReaderManager;
      }
    }
    return rm_instance_ptr_;
  }

  void StartDataReader(
      const int64_t reader_id, BlockDesc *reader_block,
      const Scope* scope, const platform::Place place,
      const std::string &indices_var_name,
      const std::vector<std::string> &output_var_names,
      const std::vector<std::shared_ptr<LoDTensorBlockingQueue>> &output_queues,
      const int batch_size, const int num_samples, const bool shuffle,
      const bool drop_last, const int rank, const int world_size) {
    auto iter = id_to_reader_.find(reader_id);
    if (iter == id_to_reader_.end()) {
      id_to_reader_[reader_id] = std::unique_ptr<DataReader>(
          new DataReader(reader_block, scope, place, indices_var_name,
                         output_var_names, output_queues, batch_size,
                         num_samples, shuffle, drop_last, rank, world_size));
    }
  }

  void ShutDownReader(const int64_t reader_id) {
    auto iter = id_to_reader_.find(reader_id);
    if (iter != id_to_reader_.end()) {
      iter->second->ShutDown();
      id_to_reader_.erase(reader_id);
    }
  }
  void ShutDown() {
    auto iter = id_to_reader_.begin();
    while (iter != id_to_reader_.end()){
      if(iter->second.get()){
        iter->second->ShutDown();
      }
      iter++;
    }
    id_to_reader_.clear();
  }

  ReaderManager() { VLOG(1) << "ReaderManager init"; }

  ~ReaderManager() {
    VLOG(1) << "~ReaderManager";
    ShutDown();
  }
};

static void CheckAndInitOutputQueue(const std::vector<Variable*>& vars, int capacity) {
  for (auto var : vars) {
    if (var->IsInitialized()) {
      PADDLE_ENFORCE_EQ(var->IsType<LoDTensorBlockingQueueHolder>(), true,
          platform::errors::InvalidArgument(
            "Output Variables of DataLoaderOp should hold "
            "LoDTensorBlockingQueueHolder type"));
      auto queue = var->Get<LoDTensorBlockingQueueHolder>().GetQueue();
      if (queue == nullptr) {
        auto* holder = var->template GetMutable<LoDTensorBlockingQueueHolder>();
        holder->InitOnce(capacity);
        VLOG(1) << "DataLoaderOpKernel init queue" << holder->GetQueue();
      }
    } else {
      VLOG(1) << "Initialize Output LoDTensorBlockingQueue capacity " << capacity;
      auto* holder = var->GetMutable<LoDTensorBlockingQueueHolder>();
      holder->InitOnce(capacity);
    }
  }
}

static std::vector<std::shared_ptr<LoDTensorBlockingQueue>> GetQueueVecFromVariableVec(const std::vector<Variable*>& vars) {
  std::vector<std::shared_ptr<LoDTensorBlockingQueue>> queues;
  queues.reserve(vars.size());
  for (size_t i = 0; i < vars.size(); i++) {
    queues.push_back(vars[i]->Get<LoDTensorBlockingQueueHolder>().GetQueue());
  }
  return queues;
}

template <typename T>
class DataReaderCPUKernel: public framework::OpKernel<T> {
 public:
   void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace data
}  // namespace operators
}  // namespace paddle
