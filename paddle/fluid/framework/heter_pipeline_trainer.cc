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

#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"

namespace paddle {
namespace framework {

class Variable;

void HeterPipelineTrainer::ResetDataset(Dataset* dataset) {
  SetDataset(dataset);
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();
  auto this_worker =
      std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(worker_);

  if (pipeline_stage_ == 0) {
    this_worker->SetDataFeed(readers[0]);
    this_worker->SetReaderPlace(place_);
  }
}

void HeterPipelineTrainer::Initialize(const TrainerDesc& trainer_desc,
                                      Dataset* dataset) {
  const auto& section_params = trainer_desc.section_param();
  num_pipeline_stages_ = section_params.num_pipeline_stages();
  pipeline_stage_ = section_params.pipeline_stage();
  num_microbatches_ = section_params.num_microbatches();
  VLOG(3) << "Number of microbatches per minibatch: " << num_microbatches_;
  trainer_desc_ = trainer_desc;

  SetDataset(dataset);
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();

  ParseDumpConfig(trainer_desc);

  trainer_id_ = trainer_desc.trainer_id();
  trainers_ = trainer_desc.trainers();
  worker_ = DeviceWorkerFactory::CreateDeviceWorker(
      trainer_desc.device_worker_name());
  auto this_worker =
      std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(worker_);
  this_worker->SetMicrobatchNum(num_microbatches_);
  this_worker->SetPipelineStageNum(num_pipeline_stages_);
  this_worker->SetPipelineStage(pipeline_stage_);
  this_worker->SetTrainerId(trainer_id_);
  this_worker->SetTrainers(trainers_);
  this_worker->Initialize(trainer_desc);
  if (pipeline_stage_ == 0) this_worker->SetDataFeed(readers[0]);
}

void HeterPipelineTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  if (need_dump_field_) {
    InitDumpEnv();
  }
}

std::string HeterPipelineTrainer::GetDumpPath(int tid) {
  return string::format_string("%s/part-%05d", dump_fields_path_.c_str(), tid);
}

void HeterPipelineTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  // TODO(sandyhouse): should make it as a config
  dump_thread_num_ = 1;
  for (int i = 0; i < dump_thread_num_; i++) {
    dump_thread_.push_back(
        std::thread(std::bind(&TrainerBase::DumpWork, this, i)));
  }
}

void HeterPipelineTrainer::CopyParameters(int microbatch_id,
                                          const ProgramDesc& program,
                                          const platform::Place& place) {
  auto& global_block = program.Block(0);
  auto var_list = global_block.AllVars();
  if (program.Size() > 1) {
    auto& heter_block = program.Block(1);
    auto heter_var_list = heter_block.AllVars();
    var_list.insert(var_list.end(), heter_var_list.begin(),
                    heter_var_list.end());
  }
  if (program.Size() > 2) {
    auto& heter_block = program.Block(2);
    auto heter_var_list = heter_block.AllVars();
    var_list.insert(var_list.end(), heter_var_list.begin(),
                    heter_var_list.end());
  }

  // create microbatch_id variable
  // and set micro id value
  auto* ptr = microbatch_scopes_[microbatch_id]->Var("microbatch_id");

  InitializeVariable(ptr, proto::VarType::LOD_TENSOR);

  framework::Variable* var =
      microbatch_scopes_[microbatch_id]->FindVar("microbatch_id");
  PADDLE_ENFORCE_EQ(var->IsType<framework::LoDTensor>(), 1,
                    platform::errors::InvalidArgument(
                        "the type of microbatch_id  should be LoDTensor"));
  auto* tensor = var->GetMutable<framework::LoDTensor>();

  std::vector<int> dims{1};
  tensor->Resize(framework::make_ddim(dims));

  void* tensor_data =
      tensor->mutable_data(place, framework::proto::VarType::FP32);

  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    char* temp_ptr =
        new char[tensor->numel() * framework::SizeOfType(tensor->type())];
    float* temp_ptr_float = reinterpret_cast<float*>(temp_ptr);
    temp_ptr_float[0] = microbatch_id;
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(*dev_ctx_)
            .stream();
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place), tensor_data,
                 platform::CPUPlace(), reinterpret_cast<void*>(temp_ptr),
                 tensor->numel() * framework::SizeOfType(tensor->type()),
                 stream);
    delete[] temp_ptr;
#endif
  } else {
    float* temp_ptr = reinterpret_cast<float*>(tensor_data);
    temp_ptr[0] = microbatch_id;
  }

  for (auto& var : var_list) {
    if (var->Persistable() && microbatch_id == 0) {
      if (root_scope_->FindVar(var->Name()) != nullptr) continue;
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

void HeterPipelineTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                          const platform::Place& place) {
  PADDLE_ENFORCE_NOT_NULL(root_scope_, platform::errors::InvalidArgument(
                                           "root_scope_ can not be nullptr"));
  place_ = place;
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place);
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
      std::dynamic_pointer_cast<paddle::framework::HeterSectionWorker>(worker_);
  this_worker->SetRootScope(root_scope_);
  this_worker->SetMinibatchScope(minibatch_scope_);
  this_worker->SetMicrobatchScopes(microbatch_scopes_);
  if (pipeline_stage_ == 0) {
    this_worker->SetReaderPlace(place);
  }
  this_worker->SetPlace(place);
}

void HeterPipelineTrainer::Run() {
  VLOG(5) << "Going to run PipelineTrainer::Run()";
  // worker_->TrainFiles();
  try {
    worker_->TrainFiles();
  } catch (platform::EOFException& e) {
    std::rethrow_exception(std::current_exception());
  }
  // for (auto* micro_scop : microbatch_scopes_) {
  // By default, we should delete all kid scopes after run executor because
  // some operators may create local scope when running, such as while_op.
  // But when while_op also create a local executor to run it's sub block,
  // the sub scopes it created should not be dropped immediately, because
  // while_grad_op will use some variables created during while_op run, so
  // we need to keep the kids and wait for the outer executor to drop them.
  // micro_scop->DropKids();
  //}
}

void HeterPipelineTrainer::Finalize() {
  if (need_dump_field_) {
    FinalizeDumpEnv();
  }
  root_scope_->DropKids();
}

Scope* HeterPipelineTrainer::GetWorkerScope(int thread_id) {
  return microbatch_scopes_[0];
}

}  // end namespace framework
}  // end namespace paddle
