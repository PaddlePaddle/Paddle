/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <google/protobuf/text_format.h>
#include <cstdlib>
#include <string>
#include <vector>

#include "io/fs.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/framework/trainer.h"
#if (defined PADDLE_WITH_NCCL || defined PADDLE_WITH_RCCL || \
     defined PADDLE_WITH_XPU_BKCL) &&                        \
    (defined PADDLE_WITH_PSLIB)
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif
#include "paddle/fluid/platform/timer_manager.h"

namespace paddle {
namespace framework {

void PSGPUTrainer::Initialize(const TrainerDesc& trainer_desc,
                              Dataset* dataset) {
  VLOG(0) << "do PSGPUTrainer::Initialize";
  thread_num_ = trainer_desc.thread_num();
  platform::TimerManager* timer_manager = platform::TimerManager::instance();
  timer_manager->init(platform::TIMER_MAX_ID * thread_num_);
  VLOG(0) << "do TimerManager Initialize " << platform::TIMER_MAX_ID * thread_num_;
  TIMER_SINGLETHREAD_ENTER(platform::TIMER_TRAINER_ALL);
  TIMER_SINGLETHREAD_ENTER(platform::TIMER_TRAINER_INIT);

  SetDataset(dataset);
    param_ = trainer_desc.downpour_param();
  ParseDumpConfig(trainer_desc);
  mpi_rank_ = trainer_desc.mpi_rank();
  mpi_size_ = trainer_desc.mpi_size();
  for (int i = 0; i < param_.dense_table_size(); ++i) {
    uint64_t table_id = static_cast<uint64_t>(param_.dense_table(i).table_id());
    auto table = param_.dense_table(i);
    dense_grad_names_[table_id].resize(table.dense_grad_name_size());
    for (int j = 0; j < table.dense_grad_name_size(); ++j) {
      dense_grad_names_[table_id][j] = table.dense_grad_name(j);
    }
  }
  InitializeGPUServer(trainer_desc);
  scale_datanorm_ = trainer_desc.scale_datanorm();
  int place_num = trainer_desc.worker_places_size();
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  dump_file_num_ = trainer_desc.dump_file_num();
  user_define_dump_filename_ = trainer_desc.user_define_dump_filename();
  std::vector<int> dev_ids;
  for (int i = 0; i < place_num; ++i) {
    int num = trainer_desc.worker_places(i);
#ifdef PADDLE_WITH_CUDA
    platform::CUDAPlace place = platform::CUDAPlace(num);
#endif
#ifdef PADDLE_WITH_XPU_KP
    platform::XPUPlace place = platform::XPUPlace(num);
#endif
    places_.push_back(place);
    dev_ids.push_back(num);
  }
  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
       i++) {
    need_merge_var_names_.push_back(
        trainer_desc.downpour_param().stat_var_names(i));
  }
  VLOG(3) << "going to initialize pull dense worker";
  SetDebug(trainer_desc.debug());
  trainer_desc_ = trainer_desc;
  workers_.resize(place_num);
  for (int i = 0; i < place_num; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetNeedDumpField(need_dump_field_);
    workers_[i]->SetNeedDumpParam(need_dump_param_);
    workers_[i]->SetDumpFieldVector(dump_fields_);
    workers_[i]->SetDumpParamVector(dump_param_);
    workers_[i]->InitRandomDumpConfig(trainer_desc);
    workers_[i]->SetDataFeed(readers[i]);
    workers_[i]->SetPlace(places_[i]);
    workers_[i]->SetReaderPlace(places_[i]);
    workers_[i]->Initialize(trainer_desc);
    workers_[i]->SetWorkerNum(place_num);
  }
  return;
}

void add_sparse_optimizer(
    std::unordered_map<std::string, float>& config,  // NOLINT
    const ::paddle::SparseCommonSGDRuleParameter& sgd_param,
    const std::string& prefix = "") {
  auto optimizer_name = sgd_param.name();
  if (optimizer_name == "naive") {
    config[prefix + "learning_rate"] = sgd_param.naive().learning_rate();
    config[prefix + "initial_range"] = sgd_param.naive().initial_range();
    if (sgd_param.naive().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.naive().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.naive().weight_bounds()[1];
    }
  } else if (optimizer_name == "adagrad") {
    config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
    config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
    config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
    if (sgd_param.adagrad().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
    }
  } else if (optimizer_name == "std_adagrad") {
    config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
    config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
    config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
    if (sgd_param.adagrad().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
    }
  } else if (optimizer_name == "adam") {
    config[prefix + "learning_rate"] = sgd_param.adam().learning_rate();
    config[prefix + "initial_range"] = sgd_param.adam().initial_range();
    if (sgd_param.adam().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.adam().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adam().weight_bounds()[1];
    }
  }
}

void PSGPUTrainer::InitializeGPUServer(const TrainerDesc& trainer_desc) {
  // optimizer config for hbmps
  auto fleet_desc_str = trainer_desc.fleet_desc();
  google::protobuf::TextFormat::ParseFromString(fleet_desc_str, &_ps_param);
  auto sparse_table =
      _ps_param.server_param().downpour_server_param().downpour_table_param(0);
  auto sparse_table_accessor = sparse_table.accessor();
  auto sparse_table_accessor_parameter =
      sparse_table_accessor.downpour_accessor_param();
  auto accessor_class = sparse_table_accessor.accessor_class();
  // NOTE(zhangminxu): gpups' sparse table optimizer config,
  // now only support single sparse table
  // auto sparse_table = param_.sparse_table(0);
  std::unordered_map<std::string, float> config;
  if (accessor_class == "DownpourFeatureValueAccessor" ||
      accessor_class == "DownpourCtrAccessor" ||
      accessor_class == "DownpourCtrDoubleAccessor") {
    config["nonclk_coeff"] = sparse_table_accessor_parameter.nonclk_coeff();
    config["clk_coeff"] = sparse_table_accessor_parameter.click_coeff();
    config["learning_rate"] =
        sparse_table_accessor.sparse_sgd_param().learning_rate();
    config["initial_g2sum"] =
        sparse_table_accessor.sparse_sgd_param().initial_g2sum();
    config["initial_range"] =
        sparse_table_accessor.sparse_sgd_param().initial_range();
    if (sparse_table_accessor.sparse_sgd_param().weight_bounds_size() == 2) {
      config["min_bound"] =
          sparse_table_accessor.sparse_sgd_param().weight_bounds()[0];
      config["max_bound"] =
          sparse_table_accessor.sparse_sgd_param().weight_bounds()[1];
    }
    // NOTE(zhangminxu): for DownpourCtrAccessor & DownpourCtrDoubleAccessor,
    // optimizer config for embed_w & embedx_w is the same
    config["mf_create_thresholds"] = sparse_table_accessor.embedx_threshold();
    config["mf_learning_rate"] = config["learning_rate"];
    config["mf_initial_g2sum"] = config["initial_g2sum"];
    config["mf_initial_range"] = config["initial_range"];
    config["mf_min_bound"] = config["min_bound"];
    config["mf_max_bound"] = config["max_bound"];
  } else if (accessor_class == "DownpourSparseValueAccessor") {
    auto optimizer_name = sparse_table_accessor.sparse_commonsgd_param().name();
    if (optimizer_name == "naive") {
      config["learning_rate"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .naive()
                                    .learning_rate();
      config["initial_range"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .naive()
                                    .initial_range();
      if (sparse_table_accessor.sparse_commonsgd_param()
              .naive()
              .weight_bounds_size() == 2) {
        config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .naive()
                                  .weight_bounds()[0];
        config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .naive()
                                  .weight_bounds()[1];
      }
    } else if (optimizer_name == "adagrad") {
      config["learning_rate"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .learning_rate();
      config["initial_range"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .initial_range();
      config["initial_g2sum"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .initial_g2sum();
      if (sparse_table_accessor.sparse_commonsgd_param()
              .adagrad()
              .weight_bounds_size() == 2) {
        config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adagrad()
                                  .weight_bounds()[0];
        config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adagrad()
                                  .weight_bounds()[1];
      }
    } else if (optimizer_name == "adam") {
      config["learning_rate"] =
          sparse_table_accessor.sparse_commonsgd_param().adam().learning_rate();
      config["initial_range"] =
          sparse_table_accessor.sparse_commonsgd_param().adam().initial_range();
      if (sparse_table_accessor.sparse_commonsgd_param()
              .adam()
              .weight_bounds_size() == 2) {
        config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adam()
                                  .weight_bounds()[0];
        config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adam()
                                  .weight_bounds()[1];
      }
    }
  } else if (accessor_class == "DownpourUnitAccessor" ||
             accessor_class == "DownpourDoubleUnitAccessor") {
    config["nonclk_coeff"] = sparse_table_accessor_parameter.nonclk_coeff();
    config["clk_coeff"] = sparse_table_accessor_parameter.click_coeff();
    config["mf_create_thresholds"] = sparse_table_accessor.embedx_threshold();
    // optimizer config for embed_w and embedx
    add_sparse_optimizer(config, sparse_table_accessor.embed_sgd_param());
    add_sparse_optimizer(config, sparse_table_accessor.embedx_sgd_param(),
                         "mf_");
  }
  auto ps_gpu_wrapper = paddle::framework::PSGPUWrapper::GetInstance();
  ps_gpu_wrapper->InitializeGPUServer(config);
}

std::string PSGPUTrainer::GetDumpPath(int tid) {
  if (user_define_dump_filename_ != "") {
    return string::format_string("%s/part-%s-%05d", dump_fields_path_.c_str(),
                                 user_define_dump_filename_.c_str(), tid);
  }
  return string::format_string("%s/part-%03d-%05d", dump_fields_path_.c_str(),
                               mpi_rank_, tid);
}

void PSGPUTrainer::RegisterHeterCallback() {
  /*
  auto fleet_ptr = FleetWrapper::GetInstance();
  fleet_ptr->RegisterHeterCallback([this](int worker, int taskid) {
    // workers_[worker]->Schedule(taskid);
  });
  */
}

void PSGPUTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                  const platform::Place& place) {
  for (size_t i = 0; i < places_.size(); ++i) {
    workers_[i]->SetRootScope(root_scope_);
    workers_[i]->CreateDeviceResource(main_program);  // Program
    workers_[i]->BindingDataFeedMemory();
  }
  for (size_t num = 0; num < places_.size(); ++num) {
    auto place = places_[num];
    Scope* scope = workers_[num]->GetThreadScope();
    auto& block = main_program.Block(0);
    for (auto& var : block.AllVars()) {
      if (var->Persistable()) {
        auto name = var->Name();
        Variable* root_var = root_scope_->FindVar(name);
        if (!root_var) {
          continue;
        }
        LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();
        auto* ptr = scope->Var(name);
        InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
        LoDTensor* thread_tensor = ptr->GetMutable<LoDTensor>();
        TensorCopy(*root_tensor, place, thread_tensor);
      }
    }
  }
  for (auto& var : main_program.Block(0).AllVars()) {
    if (var->Persistable()) {
      auto it = std::find(need_merge_var_names_.begin(),
                          need_merge_var_names_.end(), var->Name());
      if (it == need_merge_var_names_.end()) {
        VLOG(2) << "train param: " << var->Name();
        trainable_param_.push_back(var->Name());
      }
    }
  }
  place_ = place;
  return;
}

void PSGPUTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  for (size_t i = 0; i < places_.size(); ++i) {
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
        std::thread(std::bind(&TrainerBase::DumpWork, this, i)));
  }
}

void PSGPUTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  if (need_dump_field_ || need_dump_param_) {
    InitDumpEnv();
  }
  VLOG(3) << "init other env done.";
}

void PSGPUTrainer::Run() {
#ifdef PADDLE_WITH_XPU_KP
    std::string data_set_name = std::string(typeid(*dataset_ptr_).name());
    if (data_set_name.find("SlotRecordDataset") != std::string::npos) {
        auto dataset = dynamic_cast<DatasetImpl<SlotRecord> *>(dataset_ptr_);
        CHECK(dataset != nullptr) << "dataset_ptr must be derived from DatasetImpl<SlotRecord>";
        auto readers = dataset->GetReaders();
        VLOG(0) << "PSGPUTrainer Run: readers|workers:" << readers.size()
            << "|" << places_.size();
        #ifdef PADDLE_WITH_XPU_CACHE_BFID
        for (auto & reader : readers) {
           reader->Start();
        }
        auto ps_gpu_wrapper = paddle::framework::PSGPUWrapper::GetInstance();
        ps_gpu_wrapper->build_batch_fidseq(readers);
        #endif
    } else {
        auto dataset = dynamic_cast<DatasetImpl<Record> *>(dataset_ptr_);
        CHECK(dataset != nullptr) << "dataset_ptr must be derived from DatasetImpl<Record>";
        auto readers = dataset->GetReaders();
        auto & ds_chans = dataset->GetCurOutputChannel();
      
        VLOG(0) << "PSGPUTrainer Run: readers|dataset_channels|workers:" << readers.size()
                << "|" << ds_chans.size() 
                << "|" << places_.size();
        CHECK(readers.size() == ds_chans.size())
                << "readers_num should equal channel_num:"
                << readers.size() << "!=" << ds_chans.size();
        #ifdef PADDLE_WITH_XPU_CACHE_BFID
        auto & slot_is_dense = readers[0]->GetUseSlotIsDense();
        // move reader->Start from worker to Trainer
        for (auto & reader : readers) {
            reader->Start(); 
        }
        auto ps_gpu_wrapper = paddle::framework::PSGPUWrapper::GetInstance();
        std::vector<std::deque<Record> *> all_chan_data(ds_chans.size(), nullptr);
        for (size_t i = 0; i < ds_chans.size(); i++) {
          std::deque<Record>& vec_data = const_cast<std::deque<Record>&>(ds_chans[i]->GetData());
          all_chan_data[i] = &vec_data;
          VLOG(0) << "PSGPUTrainer Run: dataset multi_chan:" << i << ", size:" << vec_data.size();
        }
        ps_gpu_wrapper->build_batch_fidseq(all_chan_data, slot_is_dense);
        #endif
    }
#endif
  TIMER_SINGLETHREAD_LEAVE(platform::TIMER_TRAINER_INIT);
  TIMER_SINGLETHREAD_ENTER(platform::TIMER_TRAINER_CREATE_WORKER);
  for (size_t thidx = 0; thidx < places_.size(); ++thidx) {
    if (!debug_) {
      threads_.push_back(
          std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
    } else {
      threads_.push_back(std::thread(&DeviceWorker::TrainFilesWithProfiler,
                                     workers_[thidx].get()));
    }
  }
  TIMER_SINGLETHREAD_LEAVE(platform::TIMER_TRAINER_CREATE_WORKER);
  TIMER_SINGLETHREAD_ENTER(platform::TIMER_TRAINER_WORKERS);
}

Scope* PSGPUTrainer::GetWorkerScope(int thread_id) { return nullptr; }

template <typename T>
void PSGPUTrainer::MergeToRootScope(LoDTensor* root_tensor, LoDTensor* tensor) {
  LoDTensor tmp_root;
  TensorCopySync(*root_tensor, platform::CPUPlace(), &tmp_root);
  T* tmp_root_data = tmp_root.data<T>();
  LoDTensor tmp_tensor;
  TensorCopySync(*tensor, platform::CPUPlace(), &tmp_tensor);
  T* data = tmp_tensor.data<T>();
  for (int i = 0; i < tmp_tensor.numel(); i++) {
    tmp_root_data[i] += data[i];
  }
  TensorCopySync(tmp_root, platform::CPUPlace(), root_tensor);
}

void PSGPUTrainer::MergeDenseParam() {
  auto thread_scope = workers_[0]->GetThreadScope();
  for (auto& name : trainable_param_) {
    VLOG(2) << "merge var " << name << " to root scope";
    Variable* root_var = root_scope_->FindVar(name);
    LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();
    Variable* var = thread_scope->FindVar(name);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    TensorCopySync((*tensor), root_tensor->place(), root_tensor);
  }
}

void PSGPUTrainer::Finalize() {
  for (auto& th : threads_) {
    th.join();
  }
  TIMER_SINGLETHREAD_LEAVE(platform::TIMER_TRAINER_WORKERS);
  TIMER_SINGLETHREAD_ENTER(platform::TIMER_TRAINER_FINALIZE);
  for (size_t i = 0; i < need_merge_var_names_.size(); i++) {
    Variable* root_var = root_scope_->FindVar(need_merge_var_names_[i]);
    if (root_var == nullptr) {
      continue;
    }
    LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();
    if (root_tensor == nullptr || !root_tensor->IsInitialized()) {
      continue;
    }
    for (size_t j = 0; j < places_.size(); j++) {
      Scope* cur_thread_scope = workers_[j]->GetThreadScope();
      Variable* thread_var =
          cur_thread_scope->FindVar(need_merge_var_names_[i]);
      if (thread_var == nullptr) {
        continue;
      }
      LoDTensor* thread_tensor = thread_var->GetMutable<LoDTensor>();
      if (thread_tensor == nullptr || !thread_tensor->IsInitialized()) {
        continue;
      }
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
    }
  }
  MergeDenseParam();
  if (need_dump_field_ || need_dump_param_) {
    FinalizeDumpEnv();
  }
  root_scope_->DropKids();
  TIMER_SINGLETHREAD_LEAVE(platform::TIMER_TRAINER_FINALIZE);
  TIMER_SINGLETHREAD_LEAVE(platform::TIMER_TRAINER_ALL);
  TIMER_SINGLETHREAD_STATISTICS(platform::TIMER_TRAINER_ALL);
  TIMER_SINGLETHREAD_STATISTICS(platform::TIMER_TRAINER_INIT);
  TIMER_SINGLETHREAD_STATISTICS(platform::TIMER_TRAINER_CREATE_WORKER);
  TIMER_SINGLETHREAD_STATISTICS(platform::TIMER_TRAINER_WORKERS);
  TIMER_SINGLETHREAD_STATISTICS(platform::TIMER_TRAINER_FINALIZE);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_WORKER_ALL, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_WORKER_FEED_NEXT, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_WORKER_EACH_BATCH, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_WORKER_PULL_BOX_SPARSE, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_WORKER_PUSH_BOX_SPARSE, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_WORKER_ALL_OPS, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_WORKER_SKIP_OPS, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PUSH_SPARSE_GRAD_ALL, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PUSH_SPARSE_GRAD_DO, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PUSH_SPARSE_GRAD_COPY, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PUSH_SPARSE_GRAD_PUSH, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PUSH_SPARSE_HETER_ALL, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PUSH_SPARSE_HETER_STAGE1, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PUSH_SPARSE_HETER_STAGE2, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PUSH_SPARSE_HETER_STAGE3, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PUSH_SPARSE_HETER_STAGE4, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PUSH_SPARSE_HETER_STAGE5, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_ALL, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_DO, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_DO_FIRST, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_FID2BFID, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_PULL, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_COPY, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_COPY_FOR_PULL, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_ALL, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_STAGE1, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_STAGE2, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_STAGE3, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_STAGE4, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_STAGE5, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_STAGE6, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_STAGE7, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_STAGE8, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_STAGE9, thread_num_);
  TIMER_MULTITHREAD_STATISTICS_ALL(platform::TIMER_OPS_PULL_SPARSE_HETER_STAGE10, thread_num_);
}
}  // namespace framework
}  // namespace paddle
#endif
