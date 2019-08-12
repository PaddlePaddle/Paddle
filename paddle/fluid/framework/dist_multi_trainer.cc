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

#include <string>
#include <vector>
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"

namespace paddle {
namespace framework {

void DistMultiTrainer::Initialize(const TrainerDesc& trainer_desc,
                                  Dataset* dataset) {
  thread_num_ = trainer_desc.thread_num();
  SetDataset(dataset);

  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();

  thread_num_ = readers.size();
  workers_.resize(thread_num_);
  std::cout << "Trainer Thread num: " << thread_num_ << std::endl;
  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size(); i++) {
    std::cout << " need merge var name: " << trainer_desc.downpour_param().stat_var_names(i);
    need_merge_var_names_.push_back(trainer_desc.downpour_param().stat_var_names(i));
  }
  std::cout << "  >>need merge var names push END<<" << std::endl;

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

void DistMultiTrainer::InitOtherEnv(const ProgramDesc& main_program) {
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
  for (auto& th : threads_) {
    th.join();
  }
  for (int i = 0; i < need_merge_var_names_.size(); i++) {
    auto *root_var = root_scope_->FindVar(need_merge_var_names_[i]);
    if (root_var == nullptr) {
      continue;
    }
    auto *root_tensor = root_var->GetMutable<LoDTensor>();
    std::cout << "Will Merge Var name: " << need_merge_var_names_[i] << std::endl;
    for (int j = 0; j < thread_num_; j++) {
      std::cout << "cp 1" << std::endl;
      auto *cur_thread_scope = workers_[j]->GetThreadScope();
      std::cout << "cp 2" << std::endl;  
      auto *thread_var = cur_thread_scope->FindVar(need_merge_var_names_[i]);
      std::cout << "cp 3" << std::endl;     
      auto *thread_tensor = thread_var->GetMutable<LoDTensor>();
      std::cout << "cp 4" << std::endl;
      if (root_tensor->numel() != thread_tensor->numel()) {
        std::cout << "MERGE PASS" << std::endl;
        continue;    
      }
      for (int k = 0; k < root_tensor->numel(); k++) {
        std::cout << "thread " << j << " debug merge val: " << root_tensor->data<int64_t>()[k] << " | " << thread_tensor->data<int64_t>()[k] << std::endl;    
      }
      MergeToRootScope(root_tensor, thread_tensor);
      std::vector<std::string> delete_vars;
      delete_vars.push_back(need_merge_var_names_[i]);
      //cur_thread_scope->EraseVars(delete_vars);
    }
  }
  for (int i = 0; i < thread_num_; i++) {
    auto *cur_thread_scope = workers_[i]->GetThreadScope();
    cur_thread_scope->DropKids();
  }
  pull_dense_worker_->Stop();
  root_scope_->DropKids();
}

void DistMultiTrainer::MergeToRootScope(LoDTensor* root_tensor, LoDTensor* tensor) {
  std::cout << "root tensor numel: " << root_tensor->numel() << " tensor numel: " << tensor->numel() << std::endl;
  std::cout << "cp 5" << std::endl;
  int64_t* root_data = root_tensor->data<int64_t>();
  std::cout << "cp 6" << std::endl;
  int64_t* data = tensor->data<int64_t>();
  std::cout << "cp 7" << std::endl;
  for (int i = 0; i < tensor->numel(); i++){
    root_data[i] += data[i];
  }
}
}  // end namespace framework
}  // end namespace paddle
