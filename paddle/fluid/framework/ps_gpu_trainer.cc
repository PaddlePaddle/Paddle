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

#include <cstdlib>
#include <string>
#include <vector>
#include "io/fs.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/gpu_task.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/framework/trainer.h"
#if (defined PADDLE_WITH_CUDA) && (defined PADDLE_WITH_PSLIB)

#include "paddle/fluid/platform/cuda_device_guard.h"

namespace paddle {
namespace framework {

void PSGPUTrainer::Initialize(const TrainerDesc& trainer_desc,
                                 Dataset* dataset) {
  return;
}

void PSGPUTrainer::DumpWork(int tid) {}

void PSGPUTrainer::RegisterHeterCallback() {
  /*
  auto fleet_ptr = FleetWrapper::GetInstance();
  fleet_ptr->RegisterHeterCallback([this](int worker, int taskid) {
    // workers_[worker]->Schedule(taskid);
  });
  */
}

void PSGPUTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                     const platform::Place& place) {
  return;
}

void PSGPUTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  
  VLOG(3) << "init other env done.";
}

void PSGPUTrainer::Run() {
  
}
void PSGPUTrainer::BuildGPUPSTask(Dataset* dataset, int table_id, int feadim) {
  VLOG(3) << "PSGPUTrainer::BuildGPUPSTask begin";
  int shard_num = dataset->multi_output_channel_.size();
  auto gpu_ps_wrapper = PSGPUWrapper::GetInstance();
  std::vector<std::unordered_set<uint64_t>> local_keys_set(shard_num);
  // read thread
  std::vector<std::thread> threads(shard_num);
  dataset->consume_task_pool_.resize(shard_num);
  for (size_t i = 0; i < consume_task_pool_.size(); i++) {
    consume_task_pool_[i].reset(new ::ThreadPool(1));
  }
  auto consume_func = [&local_keys_set](int shard_id, int feadim,
                                          std::vector<uint64_t>& keys) {
    for (auto k : keys) {
      if (local_keys_set.find(k) == local_keys_set.end()) {
        local_keys_set.insert(k);
      }
    }
  };

  if (dataset->input_channel_->Size() == 0) {
    // output_channel_ should hold one pass instances now
    uint64_t output_channels_data_size = 0;
    for (size_t i = 0; i < dataset->multi_output_channel_.size(); i++) {
      int cur_channel_size = multi_output_channel_[i]->Size();
      output_channels_data_size += cur_channel_size;
      local_keys.reserve(cur_channel_size);
    }
    CHECK(output_channels_data_size > 0);
    auto gen_func = [&dataset, &shard_num, &feadim,
                   &consume_func](int i) {
      std::vector<Record> vec_data;
      std::vector<std::vector<uint64_t>> task_keys(shard_num);
      std::vector<std::future<void>> task_futures;
      dataset->multi_output_channel_[i]->Close();
      dataset->multi_output_channel_[i]->ReadAll(vec_data);
      for (size_t j = 0; j < vec_data.size(); j++) {
        for (auto& feature : vec_data[j].uint64_feasigns_) {
          int shard = feature.sign().uint64_feasign_ % shard_num;
          task_keys[shard].push_back(feature.sign().uint64_feasign_);
        }
      }

      for (int shard_id = 0; shard_id < shard_num; shard_id++) {
        task_futures.emplace_back(consume_task_pool_[shard_id]->enqueue(
            consume_func, shard_id, feadim, task_keys[shard_id]));
      }

      dataset->multi_output_channel_[i]->Open();
      dataset->multi_output_channel_[i]->Write(std::move(vec_data));
      vec_data.clear();
      vec_data.shrink_to_fit();
      
      for (auto& tf : task_futures) {
        tf.wait();
      }
      for (auto& tk : task_keys) {
        tk.clear();
        std::vector<uint64_t>().swap(tk);
      }
      task_keys.clear();
      std::vector<std::vector<uint64_t>>().swap(task_keys);
    };
  } else {
    int input_channel_size = dataset->input_channel_->Size();
    CHECK(input_channel_size > 0);
    CHECK(shard_num > 0);
    
    std::vector<Record> vec_data;
    dataset->input_channel_->Close();
    dataset->input_channel_->ReadAll(vec_data);
    auto gen_func = [&dataset, &vec_data, &shard_num, &input_channel_size, &feadim,
                   &consume_func](int i) {
      std::vector<std::vector<uint64_t>> task_keys(shard_num);
      std::vector<std::future<void>> task_futures;
      size_t per_shard_num = input_channel_size / shard_num + 1;
      size_t total_size = vec_data.size();
      size_t start_index = i * per_shard_num;
      size_t end_index = std::min(start_index+per_shard_num-1, total_size-1);
      for (size_t j = start_index; j <= end_index ; j++) {
        for (auto& feature : vec_data[j].uint64_feasigns_) {
          int shard = feature.sign().uint64_feasign_ % shard_num;
          task_keys[shard].push_back(feature.sign().uint64_feasign_);
        }
      }

      for (int shard_id = 0; shard_id < shard_num; shard_id++) {
        task_futures.emplace_back(consume_task_pool_[shard_id]->enqueue(
            consume_func, shard_id, feadim, task_keys[shard_id]));
      }

      for (auto& tf : task_futures) {
        tf.wait();
      }
      for (auto& tk : task_keys) {
        tk.clear();
        std::vector<uint64_t>().swap(tk);
      }
      task_keys.clear();
      std::vector<std::vector<uint64_t>>().swap(task_keys);
    };
    for (size_t i = 0; i < threads.size(); i++) {
      threads[i] = std::thread(gen_func, i);
    }
    for (std::thread& t : threads) {
      t.join();
    }
    dataset->input_channel_->Open();
    dataset->input_channel_->Write(std::move(vec_data));
  }
  for (size_t i = 0; i < dataset->consume_task_pool_.size(); i++) {
    dataset->consume_task_pool_[i].reset();
  }
  dataset->consume_task_pool_.clear();
  std::vector<GpuTask*> gpu_tasks(shard_num);
  
  std::vector<std::vector<uint64_t>> keys_vec(shaed_num);
  for (int i = 0; i < shard_num; i++) {
    keys_vec[i].assign(local_keys_set[i].begin(), local_keys_set[i].end());
  }
  for (int i = 0; i < shard_num, i++) {
    local_keys_set[i].clear();
  }
  local_keys_set.clear();
  gpu_ps_wrapper->BuildGPUPS(table_id, feadim, keys_vec, gpu_tasks);
}

Scope* PSGPUTrainer::GetWorkerScope(int thread_id) { return nullptr; }

template <typename T>
void PSGPUTrainer::MergeToRootScope(LoDTensor* root_tensor,
                                       LoDTensor* tensor) {
  LoDTensor tmp_root;
  TensorCopy(*root_tensor, platform::CPUPlace(), &tmp_root);
  T* tmp_root_data = tmp_root.data<T>();
  LoDTensor tmp_tensor;
  TensorCopy(*tensor, platform::CPUPlace(), &tmp_tensor);
  T* data = tmp_tensor.data<T>();
  for (int i = 0; i < tmp_tensor.numel(); i++) {
    tmp_root_data[i] += data[i];
  }
  TensorCopy(tmp_root, platform::CPUPlace(), root_tensor);
}

void PSGPUTrainer::Finalize() {
  for (auto& th : pull_threads_) {
    th.join();
  }
  for (auto& th : threads_) {
    th.join();
  }
  for (size_t i = 0; i < need_merge_var_names_.size(); i++) {
    Variable* root_var = root_scope_->FindVar(need_merge_var_names_[i]);
    if (root_var == nullptr) {
      continue;
    }
    LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();

    for (size_t j = 0; j < places_.size(); j++) {
      Scope* cur_thread_scope = workers_[j]->GetThreadScope();
      Variable* thread_var =
          cur_thread_scope->FindVar(need_merge_var_names_[i]);
      if (thread_var == nullptr) {
        continue;
      }
      LoDTensor* thread_tensor = thread_var->GetMutable<LoDTensor>();
#define MergeCallback(cpp_type, proto_type)                                    \
  do {                                                                         \
    if (root_tensor->type() == proto_type) {                                   \
      if (thread_tensor->type() != proto_type) {                               \
        VLOG(0) << "Error: thread id=" << j << ", need_merge_var_names_[" << i \
                << "] " << need_merge_var_names_[i]                            \
                << ", root tensor type=" << root_tensor->type()                \
                << ", thread tensor type=" << thread_tensor->type();           \
        exit(-1);                                                              \
      }                                                                        \
      MergeToRootScope<cpp_type>(root_tensor, thread_tensor);                  \
    }                                                                          \
  } while (0)
      _ForEachDataType_(MergeCallback);
    }
  }
  pull_dense_worker_->MergeDenseParam();
  root_scope_->DropKids();
}
}  // namespace framework
}  // namespace paddle
#endif
