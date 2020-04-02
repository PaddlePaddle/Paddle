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
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/operators/distributed/distributed.h"

namespace paddle {
namespace framework {

void MultiTrainer::Initialize(const TrainerDesc& trainer_desc,
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
  mpi_rank_ = trainer_desc.mpi_rank();
  mpi_size_ = trainer_desc.mpi_size();
  dump_file_num_ = trainer_desc.dump_file_num();

  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
       i++) {
    need_merge_var_names_.push_back(
        trainer_desc.downpour_param().stat_var_names(i));
  }
  // get filelist from trainer_desc here
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();
  // change thread num to readers num
  thread_num_ = readers.size();
  VLOG(3) << "worker thread num: " << thread_num_;
  workers_.resize(thread_num_);

#ifdef PADDLE_WITH_DISTRIBUTE
  if (trainer_desc.thread_barrier()) {
    operators::distributed::Communicator::GetInstance()->BarrierTriggerReset(
        thread_num_);
  }
#endif

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->Initialize(trainer_desc);
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetDataFeed(readers[i]);
    workers_[i]->SetNeedDump(need_dump_field_);
  }

  // set debug here
  SetDebug(trainer_desc.debug());
}

void MultiTrainer::DumpWork(int tid) {
#ifdef _LINUX
  int err_no = 0;
  std::string path = string::format_string(
      "%s/part-%03d-%05d", dump_fields_path_.c_str(), mpi_rank_, tid);

  std::shared_ptr<FILE> fp = fs_open_write(path, &err_no, dump_converter_);
  while (1) {
    std::string out_str;
    if (!queue_->Get(out_str)) {
      break;
    }
    size_t write_count =
        fwrite_unlocked(out_str.data(), 1, out_str.length(), fp.get());
    if (write_count != out_str.length()) {
      VLOG(3) << "dump text failed";
      continue;
    }
    write_count = fwrite_unlocked("\n", 1, 1, fp.get());
    if (write_count != 1) {
      VLOG(3) << "dump text failed";
      continue;
    }
  }
#endif
}

void MultiTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->SetChannelWriter(queue_.get());
  }
  dump_thread_num_ = 1;
  if (dump_file_num_ > mpi_size_) {
    dump_thread_num_ = dump_file_num_ / mpi_size_;
    if (dump_file_num_ % mpi_size_ > mpi_rank_) {
      dump_thread_num_ += 1;
    }
  }
  for (int i = 0; i < dump_thread_num_; i++) {
    dump_thread_.push_back(
        std::thread(std::bind(&MultiTrainer::DumpWork, this, i)));
  }
}

void MultiTrainer::FinalizeDumpEnv() {
  queue_->Close();
  for (auto& th : dump_thread_) {
    th.join();
  }
  queue_.reset();
}

// call only after all resources are set in current trainer
void MultiTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                  const platform::Place& place) {
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->SetPlace(place);
    workers_[i]->SetReaderPlace(place);
    workers_[i]->SetRootScope(root_scope_);
    workers_[i]->CreateDeviceResource(main_program);  // Program
    workers_[i]->BindingDataFeedMemory();
  }
}

void MultiTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  if (need_dump_field_) {
    InitDumpEnv();
  }
  VLOG(3) << "init other env done.";
}

Scope* MultiTrainer::GetWorkerScope(int thread_id) {
  return workers_[thread_id]->GetThreadScope();
}

void MultiTrainer::Run() {
  VLOG(3) << "Going to run";
  for (int thidx = 0; thidx < thread_num_; ++thidx) {
    if (!debug_) {
      threads_.push_back(
          std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
    } else {
      threads_.push_back(std::thread(&DeviceWorker::TrainFilesWithProfiler,
                                     workers_[thidx].get()));
    }
  }
  for (auto& th : threads_) {
    th.join();
  }
}

void MultiTrainer::Finalize() {
  if (need_dump_field_) {
    FinalizeDumpEnv();
  }
  root_scope_->DropKids();
}

}  // end namespace framework
}  // end namespace paddle
