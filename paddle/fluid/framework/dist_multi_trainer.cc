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

#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"
#include <string>
#include <vector>

namespace paddle {
namespace framework {

void DistMultiTrainer::Initialize(const TrainerDesc &trainer_desc,
                                  Dataset *dataset) {
  thread_num_ = trainer_desc.thread_num();
  SetDataset(dataset);

  const std::vector<paddle::framework::DataFeed *> readers =
      dataset->GetReaders();

  thread_num_ = readers.size();
  workers_.resize(thread_num_);
  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
       i++) {
    need_merge_var_names_.push_back(
        trainer_desc.downpour_param().stat_var_names(i));
  }

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetDataFeed(readers[i]);
    workers_[i]->Initialize(trainer_desc);
  }

  VLOG(3) << "going to initialize pull dense worker";
  pull_dense_worker_ = PullDenseWorker::GetInstance();
  pull_dense_worker_->Initialize(trainer_desc);
  VLOG(3) << "initialize pull dense worker";
  SetDebug(trainer_desc.debug());
}

void DistMultiTrainer::InitOtherEnv(const ProgramDesc &main_program) {
  pull_dense_worker_->SetRootScope(root_scope_);
  pull_dense_worker_->Start();
  VLOG(3) << "init other env done.";
}

void DistMultiTrainer::Run() {
  for (int thidx = 0; thidx < thread_num_; ++thidx) {
    if (!debug_) {
      threads_.push_back(
          std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
    } else {
      threads_.push_back(std::thread(&DeviceWorker::TrainFilesWithProfiler,
                                     workers_[thidx].get()));
    }
  }
}

void DistMultiTrainer::Finalize() {
  for (auto &th : threads_) {
    th.join();
  }
  for (int i = 0; i < need_merge_var_names_.size(); i++) {
    Variable *root_var = root_scope_->FindVar(need_merge_var_names_[i]);
    if (root_var == nullptr) {
      continue;
    }
    LoDTensor *root_tensor = root_var->GetMutable<LoDTensor>();
    for (int j = 1; j < thread_num_; j++) {
      Scope *cur_thread_scope = workers_[j]->GetThreadScope();
      Variable *thread_var =
          cur_thread_scope->FindVar(need_merge_var_names_[i]);
      LoDTensor *thread_tensor = thread_var->GetMutable<LoDTensor>();
      if (root_tensor->numel() != thread_tensor->numel()) {
        continue;
      }
#define MergeCallback(cpp_type, proto_type)                                    \
  do {                                                                         \
    if (root_tensor->type() == proto_type) {                                   \
      MergeToRootScope<cpp_type>(root_tensor, thread_tensor);                  \
    }                                                                          \
  } while (0)
      _ForEachDataType_(MergeCallback);
    }
  }

  pull_dense_worker_->Stop();
  root_scope_->DropKids();
}

template <typename T>
void DistMultiTrainer::MergeToRootScope(LoDTensor *root_tensor,
                                        LoDTensor *tensor) {
  T *root_data = root_tensor->data<T>();
  T *data = tensor->data<T>();
  for (int i = 0; i < tensor->numel(); i++) {
    root_data[i] += data[i];
  }
}
} // end namespace framework
} // end namespace paddle
