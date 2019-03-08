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

#include "paddle/fluid/framework/async_executor.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/executor_thread_worker.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/framework/trainer_factory.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/pybind.h"

namespace paddle {
namespace framework {
AsyncExecutor::AsyncExecutor(Scope* scope, const platform::Place& place)
    : root_scope_(scope), place_(place) {}

void AsyncExecutor::InitServer(const std::string& dist_desc, int index) {
  fleet_ptr_ = FleetWrapper::GetInstance();
  fleet_ptr_->InitServer(dist_desc, index);
}

void AsyncExecutor::InitWorker(const std::string& dist_desc,
                               const std::vector<uint64_t>& host_sign_list,
                               int node_num, int index) {
  fleet_ptr_ = FleetWrapper::GetInstance();
  fleet_ptr_->InitWorker(dist_desc, host_sign_list, node_num, index);
}

uint64_t AsyncExecutor::StartServer() { return fleet_ptr_->RunServer(); }

void AsyncExecutor::StopServer() { fleet_ptr_->StopServer(); }

void AsyncExecutor::GatherServers(const std::vector<uint64_t>& host_sign_list,
                                  int node_num) {
  fleet_ptr_->GatherServers(host_sign_list, node_num);
}

void AsyncExecutor::RunFromFile(const ProgramDesc& main_program,
                                const std::string& trainer_desc_str,
                                const bool debug) {
  VLOG(3) << "start to run from files in async_executor";
  TrainerDesc trainer_desc;
  google::protobuf::TextFormat::ParseFromString(trainer_desc_str,
                                                &trainer_desc);
  VLOG(3) << "Going to create trainer, trainer class is "
          << trainer_desc.class_name();
  std::shared_ptr<TrainerBase> trainer;
  trainer = TrainerFactory::CreateTrainer(trainer_desc.class_name());
  // initialize trainer
  VLOG(3) << "Going to initialize trainer";
  // trainer->Initialize(trainer_desc, NULL);
  VLOG(3) << "Set root scope here";
  trainer->SetScope(root_scope_);
  VLOG(3) << "Going to set debug";
  trainer->SetDebug(debug);
  // prepare training environment and helper environment
  VLOG(3) << "Try to init train environment";
  trainer->InitTrainerEnv(main_program, place_);
  VLOG(3) << "Try to init other environment";
  trainer->InitOtherEnv(main_program);
  // training and finalize training
  VLOG(3) << "Trainer starts to run";
  trainer->Run();
  VLOG(3) << "Trainer going to finalize";
  trainer->Finalize();
  VLOG(3) << "Drop current scope kids";
  root_scope_->DropKids();
  return;
}

}  // end namespace framework
}  // end namespace paddle
