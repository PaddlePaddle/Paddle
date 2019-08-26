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
#include "io/fs.h"
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

  dump_fields_path_ = trainer_desc.dump_fields_path();
  dump_converter_ = trainer_desc.dump_converter();
  need_dump_field_ = false;
  if (trainer_desc.dump_fields_size() != 0 && dump_fields_path_ != "") {
    need_dump_field_ = true;
  }
  if (need_dump_field_) {
    auto& file_list = dataset->GetFileList();
    if (file_list.size() == 0) {
      need_dump_field_ = false;
    }
  }
  mpi_rank_ = trainer_desc.mpi_rank() / 2;
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();

  thread_num_ = readers.size();
  workers_.resize(thread_num_);

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetDataFeed(readers[i]);
    workers_[i]->Initialize(trainer_desc);
    workers_[i]->SetNeedDump(need_dump_field_);
  }

  VLOG(3) << "going to initialize pull dense worker";
  pull_dense_worker_ = PullDenseWorker::GetInstance();
  pull_dense_worker_->Initialize(trainer_desc);
  VLOG(3) << "initialize pull dense worker";
  SetDebug(trainer_desc.debug());
}

void DistMultiTrainer::DumpWork() {
#ifdef _LINUX
  while (1) {
    std::string out_str;
    if (!queue_->Get(out_str)) {
      break;
    }
    size_t write_count =
        fwrite_unlocked(out_str.data(), 1, out_str.length(), fp_.get());
    if (write_count != out_str.length()) {
      VLOG(3) << "dump text failed";
      continue;
    }
    write_count = fwrite_unlocked("\n", 1, 1, fp_.get());
    if (write_count != 1) {
      VLOG(3) << "dump text failed";
      continue;
    }
  }
#endif
}

void DistMultiTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  int err_no = 0;
  std::string path = string::format_string(
      "%s/part-%03d", dump_fields_path_.c_str(), mpi_rank_);

  fp_ = fs_open_write(path, &err_no, dump_converter_);
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->SetChannelWriter(queue_.get());
  }
  dump_thread_ = std::thread(&DistMultiTrainer::DumpWork, this);
}

void DistMultiTrainer::FinalizeDumpEnv() {
  queue_->Close();
  dump_thread_.join();
  queue_.reset();
}

void DistMultiTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  if (need_dump_field_) {
    InitDumpEnv();
  }
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
  if (need_dump_field_) {
    FinalizeDumpEnv();
  }
  pull_dense_worker_->Stop();
  root_scope_->DropKids();
}

}  // end namespace framework
}  // end namespace paddle
