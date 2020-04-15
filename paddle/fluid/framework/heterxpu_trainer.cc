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
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/trainer.h"

namespace paddle {
namespace framework {

void HeterXpuTrainer::Initialize(const TrainerDesc &trainer_desc,
                                  Dataset *dataset) {
  //thread_num_ = trainer_desc.thread_num();
  SetDataset(dataset);

  //dump_fields_path_ = trainer_desc.dump_fields_path();
  //dump_converter_ = trainer_desc.dump_converter();
  //need_dump_field_ = false;
  //if (trainer_desc.dump_fields_size() != 0 && dump_fields_path_ != "") {
  //  need_dump_field_ = true;
  //}
  //if (need_dump_field_) {
  //  auto &file_list = dataset->GetFileList();
  //  if (file_list.size() == 0) {
  //    need_dump_field_ = false;
  //  }
  //}
  //mpi_rank_ = trainer_desc.mpi_rank();
  //mpi_size_ = trainer_desc.mpi_size();
  //dump_file_num_ = trainer_desc.dump_file_num();
  const std::vector<paddle::framework::DataFeed *> readers =
      dataset->GetReaders();
  //thread_num_ = readers.size();
  //for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
  //     i++) {
  //  need_merge_var_names_.push_back(
  //      trainer_desc.downpour_param().stat_var_names(i));
  //}

  VLOG(3) << "going to initialize pull dense worker";
  pull_dense_worker_ = PullDenseWorker::GetInstance();
  pull_dense_worker_->Initialize(trainer_desc);
  VLOG(3) << "initialize pull dense worker";
  SetDebug(trainer_desc.debug());
  
  fleet_ptr_ = FleetWrapper::GetInstance();
  RegisterServiceHandler();
  CreateXpu2ServerConnection();
  //for (int i = 0; i < trainer_desc.worker_places_size(); ++i) {
  //  int num = trainer_desc.worker_places(i);
  //  platform::CUDAPlace place = platform::CUDAPlace(num);
  //  platform::CUDADeviceGuard guard(place.device);
  //  cudaStream_t stream;
  //  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&stream));
  //  copy_streams_.push_back(stream);
  //  places_.push_back(place);
  //}
        
  server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE);
  brpc::ServerOptions options;
  int start_port = 9000;
  const static int max_port = 65535;
       
  if (server_.Start(butil::my_ip_cstr(), 
      brpc::PortRange(start_port,  max_port), &options) != 0) {
      VLOG(0) << "xpu server start fail";
  }
}

void HeterXpuTrainer::CreateXpu2ServerConnection() {
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connection_type = "single";
  auto& server_list = fleet_ptr_->GetServerList();
  
  server_channels_.resize(server_list.size());
  for (size_t i = 0; i < server_list.size(); ++i) {
    server_channels_[i].reset(new brpc::Channel());
    if (server_channels_[i]->Init(server_list[i].c_str(), "", &options) != 0) {
      VLOG(0) << "server channel init fail";
    }
  }
}

void HeterXpuTrainer::DumpWork(int tid) {
}

void HeterXpuTrainer::InitTrainerEnv(const ProgramDesc &main_program,
                                      const platform::Place &place) {
}

void HeterXpuTrainer::InitOtherEnv(const ProgramDesc &main_program) {
  pull_dense_worker_->SetRootScope(root_scope_);
  pull_dense_worker_->Start();
  VLOG(3) << "init other env done.";
}

void HeterXpuTrainer::Run() {
  //for (int thidx = 0; thidx < thread_num_; ++thidx) {
  //  if (!debug_) {
  //    threads_.push_back(
  //        std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
  //  } else {
  //    threads_.push_back(std::thread(&DeviceWorker::TrainFilesWithProfiler,
  //                                   workers_[thidx].get()));
  //  }
  //}
}

int HeterXpuTrainer::RunTask(const std::string& data) {
  std::cout << data << std::endl;
  return 0;
}

void HeterXpuTrainer::RegisterServiceHandler() {
  service_.RegisterServiceHandler(
    [this](const std::string& data) -> int {
      return this->RunTask(data);
    });
}

Scope* HeterXpuTrainer::GetWorkerScope(int thread_id) {
  return nullptr;
}

void HeterXpuTrainer::Finalize() {
  //for (auto &th : threads_) {
  //  th.join();
  //}

  pull_dense_worker_->Stop();
  //root_scope_->DropKids();
}

}  // namespace framework
}  // namespace paddle
