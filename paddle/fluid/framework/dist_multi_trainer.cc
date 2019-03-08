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
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"

namespace paddle {
namespace framework {

void DistMultiTrainer::Initialize(const TrainerDesc& trainer_desc,
                                  const Dataset& data_set) {
  thread_num_ = trainer_desc.thread_num();
  workers_.resize(thread_num_);
  readers_.resize(thread_num_);

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    readers_[i] =
        DataFeedFactory::CreateDataFeed(trainer_desc.data_desc().name());
    workers_[i]->SetDeviceIndex(i);
    readers_[i]->Init(trainer_desc.data_desc());
    workers_[i]->SetDataFeed(readers_[i]);
    workers_[i]->Initialize(trainer_desc);
  }

  std::vector<std::string> filelist_vec;
  for (unsigned i = 0; i < trainer_desc.filelist_size(); ++i) {
    filelist_vec.push_back(trainer_desc.filelist(i));
  }

  readers_[0]->SetFileList(filelist_vec);

  fleet_ptr_ = FleetWrapper::GetInstance();
  pull_dense_worker_ = PullDenseWorker::GetInstance();
  pull_dense_worker_->Initialize(trainer_desc);
  VLOG(3) << "initialize pull dense worker";
}

void DistMultiTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  pull_dense_worker_->SetRootScope(root_scope_);
  pull_dense_worker_->Start();
  VLOG(3) << "init other env done.";
}

void DistMultiTrainer::Finalize() {
  for (auto& th : threads_) {
    th.join();
  }
  pull_dense_worker_->Stop();
}

}  // end namespace framework
}  // end namespace paddle
