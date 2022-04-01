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

#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#include "io/fs.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_wrapper.h"
#include "paddle/fluid/framework/trainer.h"
#if (defined PADDLE_WITH_CUDA || defined PADDLE_WITH_XPU) && \
    (defined PADDLE_WITH_PSLIB) && (!defined(PADDLE_WITH_HETERPS))
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif
namespace paddle {
namespace framework {

void HeterXpuTrainer::Initialize(const TrainerDesc& trainer_desc,
                                 Dataset* dataset) {
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
#ifdef PADDLE_WITH_CUDA
    platform::CUDAPlace place = platform::CUDAPlace(num);
    platform::CUDADeviceGuard guard(place.device);
    cudaStream_t stream;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream));
    copy_streams_.push_back(stream);
    places_.push_back(place);
    cudaEvent_t event;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    events_.push_back(event);
#endif
#ifdef PADDLE_WITH_XPU
    platform::XPUPlace place = platform::XPUPlace(num);
    places_.push_back(place);
#endif
  }
  // thread_num_ = trainer_desc.thread_num();
  // SetDataset(dataset);

  // dump_fields_path_ = trainer_desc.dump_fields_path();
  // dump_converter_ = trainer_desc.dump_converter();
  // need_dump_field_ = false;
  // if (trainer_desc.dump_fields_size() != 0 && dump_fields_path_ != "") {
  //   need_dump_field_ = true;
  // }
  // if (need_dump_field_) {
  //   auto &file_list = dataset->GetFileList();
  //   if (file_list.size() == 0) {
  //     need_dump_field_ = false;
  //   }
  // }
  // mpi_rank_ = trainer_desc.mpi_rank();
  // mpi_size_ = trainer_desc.mpi_size();
  // dump_file_num_ = trainer_desc.dump_file_num();
  // const std::vector<paddle::framework::DataFeed *> readers =
  //     dataset->GetReaders();
  // thread_num_ = readers.size();
  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
       i++) {
    need_merge_var_names_.push_back(
        trainer_desc.downpour_param().stat_var_names(i));
  }
  running_ = true;
  VLOG(3) << "going to initialize pull dense worker";
  pull_dense_worker_ = PullDenseWorker::GetInstance();
  pull_dense_worker_->Initialize(trainer_desc);
  VLOG(3) << "initialize pull dense worker";
  SetDebug(trainer_desc.debug());
  fleet_ptr_ = FleetWrapper::GetInstance();
  heter_ptr_ = HeterWrapper::GetInstance();
  RegisterServiceHandler();
  // for (int i = 0; i < trainer_desc.worker_places_size(); ++i) {
  //   int num = trainer_desc.worker_places(i);
  //   platform::CUDAPlace place = platform::CUDAPlace(num);
  //   platform::CUDADeviceGuard guard(place.device);
  //   cudaStream_t stream;
  //   PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream));
  //   copy_streams_.push_back(stream);
  //   places_.push_back(place);
  // }
  trainer_desc_ = trainer_desc;
}

void HeterXpuTrainer::CreateThreadParam(const ProgramDesc& program, int num) {
  auto place = places_[num];
  Scope* scope = place_scopes_[num];
#ifdef PADDLE_WITH_CUDA
  auto stream = copy_streams_[num];
  auto event = events_[num];
  auto dev_id = place.device;
  platform::CUDADeviceGuard guard(dev_id);
#endif

#ifdef PADDLE_WITH_XPU
  auto dev_id = place.device;
  platform::XPUDeviceGuard guard(dev_id);
#endif

  auto& block = program.Block(0);
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      auto name = var->Name();
      Variable* root_var = root_scope_->FindVar(name);
      LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();
      auto* ptr = scope->Var(name);
      InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
      LoDTensor* thread_tensor = ptr->GetMutable<LoDTensor>();

#define HeterMemcpyFunc(cpp_type, proto_type)                                 \
  do {                                                                        \
    if (framework::TransToProtoVarType(root_tensor->dtype()) == proto_type) { \
      HeterMemCpy<cpp_type>(thread_tensor, root_tensor, place, stream);       \
    }                                                                         \
  } while (0)

#define HeterMemcpyXpuFunc(cpp_type, proto_type)                              \
  do {                                                                        \
    if (framework::TransToProtoVarType(root_tensor->dtype()) == proto_type) { \
      HeterMemCpy<cpp_type>(thread_tensor, root_tensor, place);               \
    }                                                                         \
  } while (0)
#ifdef PADDLE_WITH_CUDA
      _ForEachDataType_(HeterMemcpyFunc);
#endif
#ifdef PADDLE_WITH_XPU
      _ForEachDataType_(HeterMemcpyXpuFunc);
#endif
    }
  }
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, stream));
  cudaEventSynchronize(event);
#endif
}

#ifdef PADDLE_WITH_CUDA
template <typename T>
void HeterXpuTrainer::HeterMemCpy(LoDTensor* thread_tensor,
                                  LoDTensor* root_tensor,
                                  const paddle::platform::Place& thread_place,
                                  cudaStream_t stream) {
  T* thread_ptr =
      thread_tensor->mutable_data<T>(root_tensor->dims(), thread_place);
  T* root_ptr = root_tensor->data<T>();
  if (platform::is_cpu_place(root_tensor->place())) {
    memory::Copy(thread_place, thread_ptr, platform::CPUPlace(), root_ptr,
                 sizeof(T) * root_tensor->numel(), stream);
  } else {
    memory::Copy(thread_place, thread_ptr, root_tensor->place(), root_ptr,
                 sizeof(T) * root_tensor->numel(), stream);
  }
}
#endif

#ifdef PADDLE_WITH_XPU
template <typename T>
void HeterXpuTrainer::HeterMemCpy(LoDTensor* thread_tensor,
                                  LoDTensor* root_tensor,
                                  const paddle::platform::Place& thread_place) {
  T* thread_ptr =
      thread_tensor->mutable_data<T>(root_tensor->dims(), thread_place);
  T* root_ptr = root_tensor->data<T>();
  if (platform::is_cpu_place(root_tensor->place())) {
    memory::Copy(thread_place, thread_ptr, platform::CPUPlace(), root_ptr,
                 sizeof(T) * root_tensor->numel());
  } else {
    memory::Copy(thread_place, thread_ptr, root_tensor->place(), root_ptr,
                 sizeof(T) * root_tensor->numel());
  }
}
#endif

void HeterXpuTrainer::DumpWork(int tid) {}

void HeterXpuTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                     const platform::Place& place) {
  CacheProgram(main_program);
  place_ = place;
  auto& profiler = paddle::ps::CostProfiler::instance();
  profiler.register_profiler("xpu_service_run_task");
  profiler.register_profiler("xpu_service_deserial");
  profiler.register_profiler("xpu_service_launch_kernel");
  profiler.register_profiler("xpu_service_wait");
}

void HeterXpuTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  auto& block = main_program.Block(0);
  pull_dense_worker_->SetRootScope(root_scope_);
  pull_dense_worker_->CreatePinVar();
  for (size_t i = 0; i < places_.size(); ++i) {
    Scope* scope = &(root_scope_->NewScope());
    // for (auto &var : block.AllVars()) {
    //   if (var->Persistable()) {
    //     auto *ptr = scope->Var(var->Name());
    //     InitializeVariable(ptr, var->GetType());
    //   }
    // }
    place_scopes_.push_back(scope);
    CreateThreadParam(main_program, i);
    pull_dense_worker_->AddThreadScope(scope);
    pull_dense_worker_->AddPlace(places_[i]);
#ifdef PADDLE_WITH_CUDA
    pull_dense_worker_->AddStream(copy_streams_[i]);
#endif
  }
  pull_dense_worker_->Start();
#ifdef PADDLE_WITH_CUDA
  for (auto& stream : copy_streams_) {
    cudaStreamSynchronize(stream);
  }
#endif
  op_names_.clear();
  for (auto& op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    op_names_.push_back(op_desc->Type());
    OperatorBase* local_op_ptr = local_op.release();
    ops_.push_back(local_op_ptr);
    continue;
  }
  xpu_begin_op_index_ = xpu_end_op_index_ = -1;
  xpu_begin_op_index_ = trainer_desc_.xpu_start_idx();
  xpu_end_op_index_ = trainer_desc_.xpu_end_idx();
  VLOG(0) << "xpu begin: " << xpu_begin_op_index_
          << " xpu end: " << xpu_end_op_index_;
  // CHECK(xpu_begin_op_index_ == 0);
  // CHECK(xpu_end_op_index_ = ops_.size() - 1);
  //// init pool
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < places_.size(); ++j) {
      int num = j;
      std::shared_ptr<HeterServiceContext> context =
          std::make_shared<HeterServiceContext>();
      context->place_num_ = num;
      auto place = places_[num];
      context->scope_ = &(place_scopes_[num]->NewScope());
      auto& block = program_.Block(0);
      for (auto& var : block.AllVars()) {
        if (!var->Persistable()) {
          auto* ptr = context->scope_->Var(var->Name());
          InitializeVariable(ptr, var->GetType());
        }
      }
      for (auto& v : dense_grad_names_) {
        for (auto& name : v.second) {
          auto* ptr = context->scope_->Var(name + "pin");
          InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
        }
      }
      for (auto& op_desc : block.AllOps()) {
        std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
        OperatorBase* local_op_ptr = local_op.release();
        (context->ops_).push_back(local_op_ptr);
      }
#ifdef PADDLE_WITH_CUDA
      auto dev_id = place.device;
      platform::CUDADeviceGuard guard(dev_id);
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaEventCreateWithFlags(&context->event_, cudaEventDisableTiming));
#endif
      object_pool_.Push(context);
    }
  }
  VLOG(3) << "init other env done.";
}

void HeterXpuTrainer::Run() {}

int HeterXpuTrainer::EndPass(const HeterRequest* request,
                             HeterResponse* response) {
  // int scope_num = object_pool_.Size();
  for (size_t i = 0; i < need_merge_var_names_.size(); i++) {
    Variable* root_var = root_scope_->FindVar(need_merge_var_names_[i]);
    if (root_var == nullptr) {
      continue;
    }
    LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();

    for (size_t j = 0; j < place_scopes_.size(); j++) {
      Scope* cur_thread_scope = place_scopes_[j];
      Variable* thread_var =
          cur_thread_scope->FindVar(need_merge_var_names_[i]);
      if (thread_var == nullptr) {
        continue;
      }
      LoDTensor* thread_tensor = thread_var->GetMutable<LoDTensor>();
//      if (root_tensor->numel() != thread_tensor->numel()) {
//        continue;
//      }
#define MergeCallback(cpp_type, proto_type)                                    \
  do {                                                                         \
    if (framework::TransToProtoVarType(root_tensor->dtype()) == proto_type) {  \
      if (framework::TransToProtoVarType(thread_tensor->dtype()) !=            \
          proto_type) {                                                        \
        VLOG(0) << "Error: thread id=" << j << ", need_merge_var_names_[" << i \
                << "] " << need_merge_var_names_[i]                            \
                << ", root tensor type=" << root_tensor->dtype()               \
                << ", thread tensor type=" << thread_tensor->dtype();          \
        exit(-1);                                                              \
      }                                                                        \
      MergeToRootScope<cpp_type>(root_tensor, thread_tensor);                  \
    }                                                                          \
  } while (0)
      _ForEachDataType_(MergeCallback);
      if (!platform::is_cpu_place(thread_tensor->place())) {
#ifdef PADDLE_WITH_CUDA
        auto dev_id = thread_tensor->place().device;
        platform::CUDADeviceGuard guard(dev_id);
        cudaMemset(
            thread_tensor->data(), 0,
            thread_tensor->numel() * SizeOfType(framework::TransToProtoVarType(
                                         thread_tensor->dtype())));
#endif
#ifdef PADDLE_WITH_XPU
        auto place = thread_tensor->place();
        auto dev_id = place.device;
        platform::XPUDeviceGuard guard(dev_id);
        platform::DeviceContextPool& pool =
            platform::DeviceContextPool::Instance();
        platform::DeviceContext* dev_ctx = pool.Get(place);
        const platform::XPUDeviceContext* xpu_ctx =
            reinterpret_cast<const platform::XPUDeviceContext*>(dev_ctx);
        xpu::memset(
            xpu_ctx->x_context(), thread_tensor->data(), 0,
            thread_tensor->numel() * SizeOfType(framework::TransToProtoVarType(
                                         thread_tensor->dtype())));
#endif
      } else {
        memset(thread_tensor->data(), 0,
               thread_tensor->numel() *
                   SizeOfType(
                       framework::TransToProtoVarType(thread_tensor->dtype())));
      }
    }
    auto* merge_var = response->add_vars();
    heter_ptr_->SerializeToReq(need_merge_var_names_[i], root_scope_,
                               merge_var);
    if (!platform::is_cpu_place(root_tensor->place())) {
#ifdef PADDLE_WITH_CUDA
      auto dev_id = root_tensor->place().device;
      platform::CUDADeviceGuard guard(dev_id);
      cudaMemset(
          root_tensor->data(), 0,
          root_tensor->numel() *
              SizeOfType(framework::TransToProtoVarType(root_tensor->dtype())));
#endif
#ifdef PADDLE_WITH_XPU
      auto place = root_tensor->place();
      auto dev_id = place.device;
      platform::XPUDeviceGuard guard(dev_id);
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      platform::DeviceContext* dev_ctx = pool.Get(place);
      const platform::XPUDeviceContext* xpu_ctx =
          reinterpret_cast<const platform::XPUDeviceContext*>(dev_ctx);
      xpu::memset(
          xpu_ctx->x_context(), root_tensor->data(), 0,
          root_tensor->numel() *
              SizeOfType(framework::TransToProtoVarType(root_tensor->dtype())));
#endif
    } else {
      memset(root_tensor->data(), 0,
             root_tensor->numel() * SizeOfType(framework::TransToProtoVarType(
                                        root_tensor->dtype())));
    }
  }
  return 0;
}

template <typename T>
void HeterXpuTrainer::MergeToRootScope(LoDTensor* root_tensor,
                                       LoDTensor* tensor) {
  LoDTensor tmp_root;
  TensorCopy(*root_tensor, platform::CPUPlace(), &tmp_root);
  T* tmp_root_data = tmp_root.data<T>();
  LoDTensor tmp_tensor;
  TensorCopy(*tensor, platform::CPUPlace(), &tmp_tensor);
  T* data = tmp_tensor.data<T>();
  for (int i = 0; i < tmp_tensor.numel(); i++) {
    tmp_root_data[i] += data[i];
  }
  TensorCopy(tmp_root, root_tensor->place(), root_tensor);
}

int HeterXpuTrainer::StopService(const HeterRequest* request,
                                 HeterResponse* response) {
  std::unique_lock<std::mutex> lock(mutex_);
  running_ = false;
  cond_.notify_one();
  return 0;
}

int HeterXpuTrainer::RunTask(const HeterRequest* request,
                             HeterResponse* response) {
  auto timer = std::make_shared<paddle::ps::CostTimer>("xpu_service_run_task");
  std::shared_ptr<HeterServiceContext> context = object_pool_.Get();

  if (!context->scope_) {
    int num = rand_r() % places_.size();
    context->place_num_ = num;
    auto place = places_[num];
    context->scope_ = &(place_scopes_[num]->NewScope());
    auto& block = program_.Block(0);
    for (auto& var : block.AllVars()) {
      if (!var->Persistable()) {
        auto* ptr = context->scope_->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
      }
    }
    for (auto& v : dense_grad_names_) {
      for (auto& name : v.second) {
        auto* ptr = context->scope_->Var(name + "pin");
        InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
      }
    }
    for (auto& op_desc : block.AllOps()) {
      std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
      OperatorBase* local_op_ptr = local_op.release();
      (context->ops_).push_back(local_op_ptr);
    }
#ifdef PADDLE_WITH_CUDA
    auto dev_id = place.device;
    platform::CUDADeviceGuard guard(dev_id);
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventCreateWithFlags(&context->event_, cudaEventDisableTiming));
#endif
  }

  context->Reset();
  auto place = places_[context->place_num_];
  {
    auto deserial_timer =
        std::make_shared<paddle::ps::CostTimer>("xpu_service_deserial");
    for (int i = 0; i < request->vars_size(); ++i) {
#ifdef PADDLE_WITH_CUDA
      heter_ptr_->DeSerializeToTensor(context->scope_, request->vars(i), place,
                                      copy_streams_[context->place_num_]);
#endif
#ifdef PADDLE_WITH_XPU
      heter_ptr_->DeSerializeToTensor(context->scope_, request->vars(i), place);
#endif
    }
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventRecord(context->event_, copy_streams_[context->place_num_]));
    while (cudaEventQuery(context->event_) != cudaSuccess) {
      VLOG(3) << "wait for kernel";
      bthread_yield();
    }
#endif
  }

  {
    auto launch_timer =
        std::make_shared<paddle::ps::CostTimer>("xpu_service_launch_kernel");
    for (int i = xpu_begin_op_index_; i <= xpu_end_op_index_; ++i) {
      auto& op = (context->ops_)[i];
      op->Run(*(context->scope_), place);
    }
  }
#ifdef PADDLE_WITH_CUDA
  auto* dev_ctx = static_cast<platform::CUDADeviceContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(context->event_, dev_ctx->stream()));
  // cudaEventSynchronize(context->event_);
  {
    auto wait_timer =
        std::make_shared<paddle::ps::CostTimer>("xpu_service_wait");
    while (cudaEventQuery(context->event_) != cudaSuccess) {
      VLOG(3) << "wait for kernel";
      bthread_yield();
    }
  }
#endif
#ifdef PADDLE_WITH_XPU
  xpu_wait();
#endif

  for (int i = 0; i < trainer_desc_.xpu_send_list_size(); ++i) {
    const std::string& varname = trainer_desc_.xpu_send_list(i);
    // CHECK(varname == "concat_1.tmp_0@GRAD");
    auto* res_var = response->add_vars();
    heter_ptr_->SerializeToReq(varname, context->scope_, res_var);
  }

  // std::string varname = "concat_1.tmp_0@GRAD";
  //
  // auto* res_var = response->add_vars();
  // heter_ptr_->SerializeToReq(varname, context->scope_, res_var);
  for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
       ++i) {
    uint64_t tid =
        static_cast<uint64_t>(param_.program_config(0).push_dense_table_id(i));
#ifdef PADDLE_WITH_CUDA
    fleet_ptr_->PushDenseVarsAsync(
        *(context->scope_), tid, dense_grad_names_[tid],
        &(context->push_dense_status_), scale_datanorm_, request->cur_batch(),
        places_[context->place_num_], copy_streams_[context->place_num_],
        context->event_);
#endif
#ifdef PADDLE_WITH_XPU
    fleet_ptr_->PushDenseVarsAsync(
        *(context->scope_), tid, dense_grad_names_[tid],
        &(context->push_dense_status_), scale_datanorm_, request->cur_batch(),
        places_[context->place_num_]);
#endif
  }
  for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
       ++i) {
    uint64_t tid =
        static_cast<uint64_t>(param_.program_config(0).push_dense_table_id(i));
    pull_dense_worker_->IncreaseThreadVersion(0, tid);
  }
  VLOG(3) << "push dense gradient done.";
  context->scope_->DropKids();
  object_pool_.Push(context);
  VLOG(0) << "pool size " << object_pool_.Size();
  return 0;
}

void HeterXpuTrainer::RegisterServiceHandler() {
  heter_ptr_->RegisterServiceHandler(
      0, [this](const HeterRequest* request, HeterResponse* response) -> int {
        return this->RunTask(request, response);
      });
  heter_ptr_->RegisterServiceHandler(
      1, [this](const HeterRequest* request, HeterResponse* response) -> int {
        return this->EndPass(request, response);
      });
  heter_ptr_->RegisterServiceHandler(
      2, [this](const HeterRequest* request, HeterResponse* response) -> int {
        return this->StopService(request, response);
      });
}

Scope* HeterXpuTrainer::GetWorkerScope(int thread_id) { return nullptr; }

void HeterXpuTrainer::Finalize() {
  // for (auto &th : threads_) {
  //   th.join();
  // }
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this] { return !running_; });
  sleep(3);
  pull_dense_worker_->Stop();
  root_scope_->DropKids();
}
}  // namespace framework
}  // namespace paddle
#endif
