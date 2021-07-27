// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"

namespace paddle {
namespace framework {

void PipelineTrainer::Initialize(const TrainerDesc& trainer_desc,
                                 Dataset* dataset) {
  const auto& section_params = trainer_desc.section_param();
  const int num_pipeline_stages_ = section_params.num_pipeline_stages();
  const int pipeline_stage_ = section_params.pipeline_stage();
  const int schedule_mode_ = section_params.schedule_mode();
  num_microbatches_ = section_params.num_microbatches();
  VLOG(3) << "Number of microbatches per minibatch: " << num_microbatches_;
  trainer_desc_ = trainer_desc;

  ParseDumpConfig(trainer_desc);
  const auto& section_config = section_params.section_config();
  int place_id = section_config.place_id();
#if (defined PADDLE_WITH_NCCL) || (defined PADDLE_WITH_RCCL)
  place_ = platform::CUDAPlace(place_id);
#elif (defined PADDLE_WITH_ASCEND_CL)  // NOLINT
  place_ = platform::NPUPlace(place_id);
#endif
  worker_ = DeviceWorkerFactory::CreateDeviceWorker(
      trainer_desc.device_worker_name());
  auto this_worker =
      std::dynamic_pointer_cast<paddle::framework::SectionWorker>(worker_);
  this_worker->SetPlace(place_);
  this_worker->SetMicrobatchNum(num_microbatches_);
  this_worker->SetPipelineStageNum(num_pipeline_stages_);
  this_worker->SetPipelineStage(pipeline_stage_);
  this_worker->SetScheduleMode(schedule_mode_);
  this_worker->Initialize(trainer_desc);
}

void PipelineTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  if (need_dump_field_) {
    InitDumpEnv();
  }
}

std::string PipelineTrainer::GetDumpPath(int tid) {
  return string::format_string("%s/part-%05d", dump_fields_path_.c_str(), tid);
}

void PipelineTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  // TODO(sandyhouse): should make it as a config
  dump_thread_num_ = 1;
  for (int i = 0; i < dump_thread_num_; i++) {
    dump_thread_.push_back(
        std::thread(std::bind(&TrainerBase::DumpWork, this, i)));
  }
}

void PipelineTrainer::CopyParameters(int microbatch_id,
                                     const ProgramDesc& program,
                                     const platform::Place& place) {
  auto& global_block = program.Block(0);

  for (auto& var : global_block.AllVars()) {
    if (var->Persistable() && microbatch_id == 0) {
      auto* ptr = root_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(5) << "Create persistable var: " << var->Name()
              << ", which pointer is " << ptr;
    } else if (!var->Persistable()) {
      auto* ptr = microbatch_scopes_[microbatch_id]->Var(var->Name());
      VLOG(5) << "Create variable " << var->Name() << " for microbatch "
              << microbatch_id << ", which pointer is " << ptr;
      InitializeVariable(ptr, var->GetType());
    }
  }
}

void PipelineTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                     const platform::Place& place) {
  PADDLE_ENFORCE_NOT_NULL(root_scope_, platform::errors::InvalidArgument(
                                           "root_scope_ can not be nullptr"));
  microbatch_scopes_.resize(num_microbatches_);

  VLOG(3) << "Create minibatch and microbatch scopes...";
  minibatch_scope_ = &root_scope_->NewScope();
  std::shared_ptr<framework::ProgramDesc> program;
  program.reset(new ProgramDesc(
      trainer_desc_.section_param().section_config().program_desc()));
  for (int j = 0; j < num_microbatches_; ++j) {
    microbatch_scopes_[j] = &minibatch_scope_->NewScope();
    CopyParameters(j, *program, place_);
  }

  auto this_worker =
      std::dynamic_pointer_cast<paddle::framework::SectionWorker>(worker_);
  this_worker->SetRootScope(root_scope_);
  this_worker->SetMinibatchScope(minibatch_scope_);
  this_worker->SetMicrobatchScopes(microbatch_scopes_);
  this_worker->PrepareUnusedVar();
}

void PipelineTrainer::Run() {
  VLOG(5) << "Going to run PipelineTrainer::Run()";
  try {
    worker_->TrainFiles();
  } catch (platform::EOFException& e) {
    std::rethrow_exception(std::current_exception());
  }
  for (auto* micro_scop : microbatch_scopes_) {
    // By default, we should delete all kid scopes after run executor because
    // some operators may create local scope when running, such as while_op.
    // But when while_op also create a local executor to run it's sub block,
    // the sub scopes it created should not be dropped immediately, because
    // while_grad_op will use some variables created during while_op run, so
    // we need to keep the kids and wait for the outer executor to drop them.
    micro_scop->DropKids();
  }
}

void PipelineTrainer::Finalize() {
  if (need_dump_field_) {
    FinalizeDumpEnv();
  }
  root_scope_->DropKids();
}

Scope* PipelineTrainer::GetWorkerScope(int thread_id) {
  return microbatch_scopes_[0];
}

}  // end namespace framework
}  // end namespace paddle
#endif
