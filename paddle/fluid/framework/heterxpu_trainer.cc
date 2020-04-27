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
#include <ctime>
#include <cstdlib>
#include "io/fs.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/trainer.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif

namespace paddle {
namespace framework {

void HeterXpuTrainer::Initialize(const TrainerDesc &trainer_desc,
                                  Dataset *dataset) {
  srand((unsigned)time(NULL));
  param_ = trainer_desc.downpour_param();
  for (int i = 0; i < param_.dense_table_size(); ++i) {
    uint64_t table_id = static_cast<uint64_t>(param_.dense_table(i).table_id());
    auto table = param_.dense_table(i);
    dense_grad_names_[table_id].resize(table.dense_grad_name_size());
    for (int j = 0; j < table.dense_grad_name_size(); ++j) {
      dense_grad_names_[table_id][j] = table.dense_grad_name(j);
    }
  }
  scale_datanorm_ = trainer_desc.scale_datanorm();
  int place_num = trainer_desc.worker_places_size();
  for (int i = 0; i < place_num; ++i) {
    int num = trainer_desc.worker_places(i);
    platform::CUDAPlace place = platform::CUDAPlace(num);
    platform::CUDADeviceGuard guard(place.device);
    cudaStream_t stream;
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&stream));
    copy_streams_.push_back(stream);
    places_.push_back(place);
    cudaEvent_t event;
    PADDLE_ENFORCE(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    events_.push_back(event);
  }
  
  //thread_num_ = trainer_desc.thread_num();
  //SetDataset(dataset);

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
  //const std::vector<paddle::framework::DataFeed *> readers =
  //    dataset->GetReaders();
  //thread_num_ = readers.size();
  //for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
  //     i++) {
  //  need_merge_var_names_.push_back(
  //      trainer_desc.downpour_param().stat_var_names(i));
  //}
  running_ = true;
  VLOG(3) << "going to initialize pull dense worker";
  pull_dense_worker_ = PullDenseWorker::GetInstance();
  pull_dense_worker_->Initialize(trainer_desc);
  VLOG(3) << "initialize pull dense worker";
  SetDebug(trainer_desc.debug());
  
  fleet_ptr_ = FleetWrapper::GetInstance();
  heter_ptr_ = HeterWrapper::GetInstance();
  RegisterServiceHandler();
  //for (int i = 0; i < trainer_desc.worker_places_size(); ++i) {
  //  int num = trainer_desc.worker_places(i);
  //  platform::CUDAPlace place = platform::CUDAPlace(num);
  //  platform::CUDADeviceGuard guard(place.device);
  //  cudaStream_t stream;
  //  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&stream));
  //  copy_streams_.push_back(stream);
  //  places_.push_back(place);
  //}
        
}

void HeterXpuTrainer::CreateThreadParam(const ProgramDesc& program, int num) {
  #ifdef PADDLE_WITH_CUDA
  auto place = places_[num];
  Scope* scope = place_scopes_[num];
  auto stream = copy_streams_[num];
  auto event = events_[num];

  auto dev_id = boost::get<platform::CUDAPlace>(place).device;
  platform::CUDADeviceGuard guard(dev_id);
  auto &block = program.Block(0);
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      auto name = var->Name();
      Variable* root_var = root_scope_->FindVar(name);
      LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();
      auto *ptr = scope->Var(name);
      InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
      LoDTensor* thread_tensor = ptr->GetMutable<LoDTensor>();

#define HeterMemcpyFunc(cpp_type, proto_type)                       \
  do {                                                             \
    if (root_tensor->type() == proto_type) {                       \
      HeterMemCpy<cpp_type>(thread_tensor, root_tensor, place, stream); \
    }                                                              \
  } while (0)
        _ForEachDataType_(HeterMemcpyFunc);

    }
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event, stream));
  cudaEventSynchronize(event);
  #endif
}

#ifdef PADDLE_WITH_CUDA
template <typename T>
void HeterXpuTrainer::HeterMemCpy(LoDTensor *thread_tensor, LoDTensor *root_tensor,
                            const paddle::platform::Place& thread_place, 
                            cudaStream_t stream) {
  T* thread_ptr = thread_tensor->mutable_data<T>(root_tensor->dims(), thread_place);
  T* root_ptr = root_tensor->data<T>();
  if (platform::is_cpu_place(root_tensor->place())) {
    memory::Copy(
        boost::get<platform::CUDAPlace>(thread_place),
        thread_ptr,
        platform::CPUPlace(),
        root_ptr, sizeof(T) * root_tensor->numel(), stream);
  }
  else {
    memory::Copy(
        boost::get<platform::CUDAPlace>(thread_place),
        thread_ptr,
        boost::get<platform::CUDAPlace>(root_tensor->place()),
        root_ptr, sizeof(T) * root_tensor->numel(), stream);
  }
}
#endif
void HeterXpuTrainer::DumpWork(int tid) {
}

void HeterXpuTrainer::InitTrainerEnv(const ProgramDesc &main_program,
                                      const platform::Place &place) {
  CacheProgram(main_program);
  place_ = place;
  auto& profiler = paddle::ps::CostProfiler::instance();
  profiler.register_profiler("xpu_service_run_task");
}

void HeterXpuTrainer::InitOtherEnv(const ProgramDesc &main_program) {
  auto &block = main_program.Block(0);
  
  pull_dense_worker_->SetRootScope(root_scope_);
  pull_dense_worker_->CreatePinVar();
  
  for (size_t i = 0; i < places_.size(); ++i) {
    Scope* scope = &(root_scope_->NewScope());
    //for (auto &var : block.AllVars()) {
    //  if (var->Persistable()) {
    //    auto *ptr = scope->Var(var->Name());
    //    InitializeVariable(ptr, var->GetType());
    //  }
    //}
    place_scopes_.push_back(scope);
    CreateThreadParam(main_program, i);
    pull_dense_worker_->AddThreadScope(scope);
    pull_dense_worker_->AddPlace(places_[i]);
    pull_dense_worker_->AddStream(copy_streams_[i]);
  }

  pull_dense_worker_->Start();
  for (auto& stream : copy_streams_) {
    cudaStreamSynchronize(stream);
  }
  op_names_.clear();
  for (auto &op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    op_names_.push_back(op_desc->Type());
    OperatorBase *local_op_ptr = local_op.release();
    ops_.push_back(local_op_ptr);
    continue;
  }
  
  for (size_t i = 0; i < ops_.size(); ++i) {
    auto& out_map = ops_[i]->Outputs();
    
    {
      auto it = out_map.find("Out");
      if (it != out_map.end()) {
        for (auto& x : it->second) {
          if (x == "concat_1.tmp_0") {
            xpu_begin_op_index_ = i + 1;
          }
        }
      }
    }
    
    {
      auto it = out_map.find("X@GRAD");
      if (it != out_map.end()) {
        for (auto& x : it->second) {
          if (x == "concat_1.tmp_0@GRAD") {
            xpu_end_op_index_ = i;
          }
        }
      }
    }
    
    {
      auto it = out_map.find("Out");
      if (it != out_map.end()) {
        for (auto& x : it->second) {
          if (x == "concat_1.tmp_0@GRAD") {
            xpu_end_op_index_ = i;
          }
        }
      }
    }
  }
  
  VLOG(3) << "xpu begin: " << xpu_begin_op_index_ << " xpu end: " << xpu_end_op_index_;
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

int HeterXpuTrainer::RunTask(const HeterRequest* request, HeterResponse* response) {
  auto timer = std::make_shared<paddle::ps::CostTimer>("xpu_service_run_task");
  std::shared_ptr<HeterServiceContext> context = object_pool_.Get();

  if (!context->scope_) {
    int num = rand() % places_.size();
    context->place_num_ = num;
    auto place = places_[num];
    context->scope_ = &(place_scopes_[num]->NewScope());
    auto &block = program_.Block(0);
    for (auto &var : block.AllVars()) {
      if (!var->Persistable()) {
        auto *ptr = context->scope_->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
      }
    }
    for (auto& v : dense_grad_names_) {
      for (auto& name : v.second) {
        auto *ptr = context->scope_->Var(name + "pin");
        InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
      }
    }
  
    auto dev_id = boost::get<platform::CUDAPlace>(place).device;
    platform::CUDADeviceGuard guard(dev_id);
    PADDLE_ENFORCE(cudaEventCreateWithFlags(&context->event_, cudaEventDisableTiming));
  }

  auto place = places_[context->place_num_];
  for (int i = 0; i < request->vars_size(); ++i) {
    heter_ptr_->DeSerializeToTensor(context->scope_, request->vars(i));
  }
  
  for (int i = xpu_begin_op_index_; i <= xpu_end_op_index_; ++i) {
    auto& op = ops_[i];
    op->Run(*(context->scope_), place);
  }
  auto* dev_ctx = static_cast<platform::CUDADeviceContext *>(
                              platform::DeviceContextPool::Instance().Get(place));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(context->event_, dev_ctx->stream()));
  //cudaEventSynchronize(context->event_);
  while (cudaEventQuery(context->event_) != cudaSuccess) {
    bthread_yield();
  }

  std::string varname = "concat_1.tmp_0@GRAD";

  auto* res_var = response->mutable_vars();
  heter_ptr_->SerializeToReq(varname, context->scope_, res_var);
  
  for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
       ++i) {
    uint64_t tid = static_cast<uint64_t>(
        param_.program_config(0).push_dense_table_id(i));
    fleet_ptr_->PushDenseVarsAsync(
        *(context->scope_), tid, dense_grad_names_[tid], &push_dense_status_,
        scale_datanorm_, request->cur_batch(), places_[context->place_num_],
        copy_streams_[context->place_num_], context->event_);
  }
          
  for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
       ++i) {
    uint64_t tid = static_cast<uint64_t>(
        param_.program_config(0).push_dense_table_id(i));
    pull_dense_worker_->IncreaseThreadVersion(0, tid);
  }
  VLOG(3) << "push dense gradient done.";

  object_pool_.Push(context);
  return 0;
}

void HeterXpuTrainer::RegisterServiceHandler() {
  heter_ptr_->RegisterServiceHandler(
    [this](const HeterRequest* request, HeterResponse* response) -> int {
      return this->RunTask(request, response);
      //return 0;
    });
}

Scope* HeterXpuTrainer::GetWorkerScope(int thread_id) {
  return nullptr;
}

void HeterXpuTrainer::Finalize() {
  //for (auto &th : threads_) {
  //  th.join();
  //}
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this] { return !running_; });

  pull_dense_worker_->Stop();
  root_scope_->DropKids();
}

}  // namespace framework
}  // namespace paddle
