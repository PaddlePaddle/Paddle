/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <atomic>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <thread>         // NOLINT
#include <unordered_map>  // NOLINT
#include <unordered_set>  // NOLINT
#include <utility>        // NOLINT
#include <vector>

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/heter_util.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/pten/backends/dynload/port.h"

namespace paddle {
namespace framework {
class ProgramDesc;
class Scope;
}  // namespace framework
}  // namespace paddle

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace framework {

std::string PrintLodTensor(Tensor* tensor, int64_t start, int64_t end);
std::pair<int64_t, int64_t> GetTensorBound(LoDTensor* tensor, int index);
bool CheckValidOutput(LoDTensor* tensor, size_t batch_size);

class FleetWrapper;

#if defined(PADDLE_WITH_PSLIB) && !defined(PADDLE_WITH_HETERPS)
class HeterWrapper;
#endif

class PullDenseWorker {
 public:
  virtual ~PullDenseWorker() {}
  virtual void Initialize(const TrainerDesc& param);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void AddStream(const gpuStream_t stream) { copy_streams_.push_back(stream); }
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU)
  void AddPlace(const paddle::platform::Place place) {
    places_.push_back(place);
  }

  void AddThreadScope(Scope* scope) { thread_scopes_.push_back(scope); }
#endif
  int Start();
  void Stop();
  void SetRootScope(Scope* scope) { root_scope_ = scope; }
  void IncreaseThreadVersion(int thread_id, uint64_t table_id);
  void ResetThreadVersion(uint64_t table_id);
  void Wait(std::vector<::std::future<int32_t>>* status_vec);
  void PullDense(bool force_update = false);
  void CreatePinVar();
  void MergeDenseParam();
  int GetThreadIdByScope(const Scope* scope);
  void SetThreadIdByScope(const Scope* scope, int tid);
  static std::shared_ptr<PullDenseWorker> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::PullDenseWorker());
    }
    return s_instance_;
  }

  static std::shared_ptr<PullDenseWorker> s_instance_;

 private:
  PullDenseWorker() : root_scope_(NULL) {}
  void Run();
  bool CheckUpdateParam(uint64_t table_id);

 private:
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;
  PullDenseWorkerParameter param_;
  DownpourWorkerParameter dwp_param_;
  Scope* root_scope_;
  bool running_;

  static std::map<uint64_t, uint64_t> last_versions_;
  static std::map<uint64_t, uint64_t> current_version_;
  static std::mutex mutex_for_version_;
  static std::map<uint64_t, std::vector<uint64_t>> training_versions_;
  static std::map<uint64_t, std::vector<std::string>> dense_value_names_;

  std::thread t_;
  int thread_num_;
  int sleep_time_ms_;
  int threshold_;

  std::vector<::std::future<int32_t>> pull_dense_status_;
  uint32_t pull_dense_fail_times_ = 0;
  std::vector<float> base_norm_param_;
  std::vector<float> mean_;
  std::vector<float> scale_;
  float squared_sum_epsilon_ = 1e-4;
  std::mutex mutex_for_mean_scale_;
  float total_batch_num_ = 0;
  std::unordered_map<const Scope*, int> scope_to_thread_id_;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::vector<gpuStream_t> copy_streams_;
#endif
  std::vector<paddle::platform::Place> places_;
  std::vector<Scope*> thread_scopes_;
};

// should incorporate different type of device
class DeviceWorker {
 public:
  DeviceWorker() {
    no_cvm_ = true;
    use_cvm_ = false;
  }
  virtual ~DeviceWorker() {}
  virtual void Initialize(const TrainerDesc& desc) = 0;
  virtual void InitRandomDumpConfig(const TrainerDesc& desc);
  virtual void SetDeviceIndex(int tid) = 0;
  virtual void TrainFiles() = 0;
  virtual void PrintFetchVars() = 0;
  virtual void TrainFilesWithProfiler() = 0;
  virtual void CreateDeviceResource(const ProgramDesc& main_prog) = 0;
  // will make this zero copy in the future
  virtual void BindingDataFeedMemory() = 0;
  virtual void SetRootScope(Scope* root_scope);
  virtual void SetDataFeed(DataFeed* data_feed);
  virtual void SetWorkerNum(int num) {}
  virtual void CacheProgram(const ProgramDesc& main_program) {}
  virtual void ProduceTasks() {}
  virtual void GetXpuOpIndex() {}
  virtual void Schedule(int taskid) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  virtual void SetStream(const gpuStream_t stream) {}
  virtual void SetEvent(const gpuEvent_t event) {}
#endif
  virtual void SetNeedDumpField(bool need_dump_field) {
    need_dump_field_ = need_dump_field;
  }
  virtual void SetNeedDumpParam(bool need_dump_param) {
    need_dump_param_ = need_dump_param;
  }
  virtual void SetDumpFieldVector(const std::vector<std::string>& dump_fields) {
    dump_fields_ = &dump_fields;
  }
  virtual void SetDumpParamVector(const std::vector<std::string>& dump_param) {
    dump_param_ = &dump_param;
  }
  virtual void SetChannelWriter(ChannelObject<std::string>* queue) {
    writer_.Reset(queue);
  }
  virtual void SetPlace(const paddle::platform::Place& place) {
    place_ = place;
  }
  virtual void SetReaderPlace(const paddle::platform::Place& place) {
    device_reader_->SetPlace(place);
  }
  virtual void SetDeviceContext(platform::DeviceContext* dev_ctx) {
    dev_ctx_ = dev_ctx;
  }
  virtual Scope* GetThreadScope() { return thread_scope_; }
  DataFeed* device_reader_ = nullptr;

 protected:
  virtual void DumpParam(const Scope& scope, const int batch_id);
  virtual void DumpField(const Scope& scope, int dump_mode,
                         int dump_interval = 10000);
  Scope* root_scope_ = nullptr;
  Scope* thread_scope_;
  paddle::platform::Place place_;
  int64_t batch_num_ = 0;
  FetchConfig fetch_config_;
  bool use_cvm_;
  bool no_cvm_;
  bool scale_sparse_gradient_with_batch_size_;
  TrainerDesc trainer_desc_;

  // dump params or grads for debug
  bool need_dump_param_;
  bool need_dump_field_;
  const std::vector<std::string>* dump_param_;
  const std::vector<std::string>* dump_fields_;
  std::vector<std::string> all_param_;

  int dump_mode_ = 0;
  int dump_interval_ = 10000;
  ChannelWriter<std::string> writer_;
  platform::DeviceContext* dev_ctx_ = nullptr;
};

class CPUWorkerBase : public DeviceWorker {
 public:
  CPUWorkerBase() {}
  virtual ~CPUWorkerBase() {}
  virtual void SetDeviceIndex(int tid) { thread_id_ = tid; }
  virtual void TrainFiles() = 0;
  virtual void TrainFilesWithProfiler() {}
  virtual void PrintFetchVars() {}
  virtual void CreateDeviceResource(const ProgramDesc& main_prog) {}

 protected:
  int thread_id_;
};

class HogwildWorker : public CPUWorkerBase {
 public:
  HogwildWorker() {}
  virtual ~HogwildWorker() {
    for (OperatorBase* op : ops_) {
      delete op;
    }
    std::vector<OperatorBase*>().swap(ops_);
  }
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();
  virtual void PrintFetchVars();
  virtual void CreateDeviceResource(const ProgramDesc& main_prog);
  virtual void BindingDataFeedMemory();
  template <typename T>
  void SetZero(LoDTensor* tensor, LoDTensor* root_tensor, int tensor_dim);

 protected:
  void CreateThreadOperators(const ProgramDesc& program);
  void CreateThreadScope(const ProgramDesc& program);

  std::vector<std::string> op_names_;
  std::vector<OperatorBase*> ops_;
  bool thread_barrier_;
  // Scope* thread_scope_;
  HogwildWorkerParameter param_;
  std::vector<std::string> skip_ops_;
  std::map<std::string, int> stat_var_name_map_;
};

class DownpourWorker : public HogwildWorker {
 public:
  DownpourWorker() {}
  virtual ~DownpourWorker() {}
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();

 protected:
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;
  std::shared_ptr<paddle::framework::PullDenseWorker> pull_dense_worker_;
  void FillSparseValue(size_t table_id);
  void PushGradients();
  void CollectLabelInfo(size_t table_id);
  void AdjustInsWeight();
  void CopySparseTable();
  void CopyDenseTable();
  void CopyDenseVars();

  DownpourWorkerParameter param_;
  // copy table
  CopyTableConfig copy_table_config_;
  std::vector<std::pair<uint64_t, uint64_t>> copy_sparse_tables_;
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> feasign_set_;
  // actually pushed feasign of each table
  std::map<uint64_t, std::vector<uint64_t>> sparse_push_keys_;
  std::map<uint64_t, std::vector<std::string>> sparse_key_names_;
  // feasign
  std::map<uint64_t, std::vector<uint64_t>> features_;
  // feasign embedding
  std::map<uint64_t, std::vector<std::vector<float>>> feature_values_;
  std::map<uint64_t, std::vector<std::string>> sparse_value_names_;
  // adjust ins weight
  AdjustInsWeightConfig adjust_ins_weight_config_;
  // check nan and inf during training
  std::vector<std::string> check_nan_var_names_;
  bool need_to_push_sparse_;
  // feasign stats
  std::map<uint64_t, std::vector<float>> feature_labels_;
  std::map<uint64_t, std::vector<std::string>> sparse_grad_names_;
  // feasign embedding gradient
  std::map<uint64_t, std::vector<std::vector<float>>> feature_grads_;
  std::vector<::std::future<int32_t>> push_sparse_status_;
  bool dump_slot_;
  bool need_to_push_dense_;
  std::map<uint64_t, std::vector<std::string>> dense_grad_names_;
  float scale_datanorm_;
  std::vector<::std::future<int32_t>> push_dense_status_;
  // skipped ops
  std::vector<std::string> skip_ops_;
  // just save the value in param_ for easy access
  std::map<uint64_t, std::string> label_var_name_;
  std::map<uint64_t, std::vector<std::string>> dense_value_names_;
  std::map<uint64_t, uint64_t> table_dependency_;
  std::vector<std::pair<uint64_t, uint64_t>> copy_dense_tables_;
  // multitask
  std::map<int32_t, uint64_t> cond2table_map_;
  std::set<uint64_t> condvalue_set_;
  bool flag_partial_push_;

 private:
  // std::vector<std::string> dump_param_;
  // just save the value in param_ for easy access
  // std::map<uint64_t, std::string> label_var_name_;
  // std::map<uint64_t, std::vector<std::string>> dense_value_names_;

  std::shared_ptr<PullDenseWorker> _pull_dense_worker;

  std::vector<float> nid_show_;
  // std::map<uint64_t, uint64_t> table_dependency_;
  // std::vector<std::pair<uint64_t, uint64_t>> copy_dense_tables_;
};

class DownpourWorkerOpt : public DownpourWorker {
 public:
  DownpourWorkerOpt() {}
  virtual ~DownpourWorkerOpt() {}
  virtual void CreateDeviceResource(const ProgramDesc& main_prog);
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();

 protected:
  void CreateThreadOperatorsWithRerank(const ProgramDesc& program);
  std::vector<std::vector<OperatorBase*>> loss_ops_;
  std::vector<std::vector<std::string>> loss_op_names_;
  std::vector<std::string> loss_names_;
  std::string async_wait_name_;
  int async_index_ = -1;
  uint64_t async_tid_ = 0;
};

#if defined(PADDLE_WITH_PSLIB) && !defined(PADDLE_WITH_HETERPS)
class HeterCpuWorker : public HogwildWorker {
 public:
  HeterCpuWorker() {}
  virtual ~HeterCpuWorker() {}
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();
  virtual void SetNeedDump(bool need_dump_field);
  virtual void SetChannelWriter(ChannelObject<std::string>* queue);
  virtual void SetWorkerNum(int num) { worker_num_ = num; }
  virtual void Schedule(int taskid);
  virtual void JumpContext(std::shared_ptr<HeterTask> task);
  virtual void CacheProgram(const ProgramDesc& main_program) {
    new (&program_) ProgramDesc(main_program);
  }
  virtual void GetXpuOpIndex();

 protected:
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;
  std::shared_ptr<paddle::framework::HeterWrapper> heter_ptr_;
  std::shared_ptr<paddle::framework::PullDenseWorker> pull_dense_worker_;
  void FillSparseValue(std::shared_ptr<HeterTask> task, size_t table_id);
  void PushGradients();
  void CollectLabelInfo(std::shared_ptr<HeterTask> task, size_t table_id);
  void AdjustInsWeight(std::shared_ptr<HeterTask> task);
  void DumpParam();
  void CopySparseTable();
  void CopyDenseTable();
  void CopyDenseVars();

 private:
  int mpi_rank_;
  int worker_num_;
  int xpu_begin_op_index_;
  int xpu_end_op_index_;
  ProgramDesc program_;
  HeterObjectPool<HeterTask> object_pool_;
  HeterList<int, std::shared_ptr<HeterTask>> run_queue_;
  HeterList<int, std::shared_ptr<HeterTask>> wait_queue_;
  bool need_dump_param_;
  std::vector<std::string> dump_param_;
  bool need_to_push_dense_;
  bool need_dump_field_;
  bool dump_slot_;
  bool need_to_push_sparse_;
  std::vector<std::string> dump_fields_;
  ChannelWriter<std::string> writer_;
  DownpourWorkerParameter param_;
  float scale_datanorm_;
  // just save the value in param_ for easy access
  std::map<uint64_t, std::string> label_var_name_;
  std::map<uint64_t, std::vector<std::string>> sparse_key_names_;
  std::map<uint64_t, std::vector<std::string>> sparse_value_names_;
  std::map<uint64_t, std::vector<std::string>> sparse_grad_names_;
  std::map<uint64_t, std::vector<std::string>> dense_value_names_;
  std::map<uint64_t, std::vector<std::string>> dense_grad_names_;
  platform::Place root_place_;
  // actually pushed feasign of each table
  std::map<uint64_t, std::vector<uint64_t>> sparse_push_keys_;

  // skipped ops
  std::vector<std::string> skip_ops_;

  std::vector<::std::future<int32_t>> push_sparse_status_;
  std::vector<::std::future<int32_t>> push_dense_status_;

  // adjust ins weight
  AdjustInsWeightConfig adjust_ins_weight_config_;
  std::vector<float> nid_show_;
  // check nan and inf during training
  std::vector<std::string> check_nan_var_names_;
  // copy table
  CopyTableConfig copy_table_config_;
  std::map<uint64_t, uint64_t> table_dependency_;
  std::vector<std::pair<uint64_t, uint64_t>> copy_sparse_tables_;
  std::vector<std::pair<uint64_t, uint64_t>> copy_dense_tables_;
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> feasign_set_;
};
#endif

#if (defined PADDLE_WITH_NCCL || defined PADDLE_WITH_RCCL) && \
    (defined PADDLE_WITH_PSLIB)
class PSGPUWorker : public HogwildWorker {
 public:
  PSGPUWorker() {}
  virtual ~PSGPUWorker() {}
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();
  virtual void SetChannelWriter(ChannelObject<std::string>* queue);
  virtual void SetWorkerNum(int num) { worker_num_ = num; }
  virtual void CacheProgram(const ProgramDesc& main_program) {
    new (&program_) ProgramDesc(main_program);
  }
  void ProduceTasks() override;
  virtual void SetStream(const gpuStream_t stream) { copy_stream_ = stream; }
  virtual void SetEvent(const gpuEvent_t event) { event_ = event; }
  void ResetStat();

 protected:
  void PushGradients();
  void CopySparseTable();
  void CopyDenseTable();
  void CopyDenseVars();

 private:
  int mpi_rank_;
  std::mutex mutex_;
  int worker_num_;
  ProgramDesc program_;
  HeterObjectPool<HeterTask> object_pool_;
  bool need_to_push_dense_;
  bool dump_slot_;
  bool need_to_push_sparse_;
  DownpourWorkerParameter param_;
  float scale_datanorm_;
  // just save the value in param_ for easy access
  std::map<uint64_t, std::string> label_var_name_;
  std::map<uint64_t, std::vector<std::string>> sparse_key_names_;
  std::map<uint64_t, std::vector<std::string>> sparse_value_names_;
  std::map<uint64_t, std::vector<std::string>> sparse_grad_names_;
  std::map<uint64_t, std::vector<std::string>> dense_value_names_;
  std::map<uint64_t, std::vector<std::string>> dense_grad_names_;
  platform::Place root_place_;
  // actually pushed feasign of each table
  std::map<uint64_t, std::vector<uint64_t>> sparse_push_keys_;

  // skipped ops
  std::vector<std::string> skip_ops_;

  std::vector<::std::future<int32_t>> push_sparse_status_;
  std::vector<::std::future<int32_t>> push_dense_status_;

  // adjust ins weight
  AdjustInsWeightConfig adjust_ins_weight_config_;
  std::vector<float> nid_show_;
  // check nan and inf during training
  std::vector<std::string> check_nan_var_names_;
  // copy table
  CopyTableConfig copy_table_config_;
  std::map<uint64_t, uint64_t> table_dependency_;
  std::vector<std::pair<uint64_t, uint64_t>> copy_sparse_tables_;
  std::vector<std::pair<uint64_t, uint64_t>> copy_dense_tables_;
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> feasign_set_;
  paddle::framework::Channel<std::shared_ptr<HeterTask>> pull_queue_;
  paddle::framework::Channel<std::shared_ptr<HeterTask>> push_queue_;
  gpuEvent_t event_;
  gpuStream_t copy_stream_;
  int batch_cnt_{0};
  std::atomic<int> done_cnt_{0};

  double total_time_;
  double read_time_;
  double pack_time_;
  double pull_sparse_local_time_;
  double op_all_time_;
  double xpu_op_time_;
  double xpu_wait_time_;
  double cpu_op_time_;
  double collect_label_time_;
  double fill_sparse_time_;
  double push_sparse_time_;
  double gpu_2_cpu_time_;
  double cpu_2_gpu_time_;
  uint64_t total_inst_;
};
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_ASCEND_CL)
class SectionWorker : public DeviceWorker {
 public:
  SectionWorker() {}
  ~SectionWorker() override {}

  void Initialize(const TrainerDesc& desc) override;
  void PrepareUnusedVar();

  void BindingDataFeedMemory() override {}
  void CreateDeviceResource(const ProgramDesc& main_prog) override{};

  void TrainFiles() override;
  void TrainFilesWithProfiler() override{};

  void PrintFetchVars() override {}

  const platform::Place& place() const { return place_; }

  void SetDeviceIndex(int tid) override {}
  void SetThreadIndex(int thread_id) { thread_id_ = thread_id; }
  void SetMicrobatchNum(int num) { num_microbatches_ = num; }
  void SetPipelineStageNum(int num) { num_pipeline_stages_ = num; }
  void SetPipelineStage(int stage) { pipeline_stage_ = stage; }
  void SetScheduleMode(int mode) { schedule_mode_ = mode; }
  void SetMicrobatchScopes(const std::vector<Scope*>& scope) {
    microbatch_scopes_ = scope;
  }
  void SetMinibatchScope(const Scope* scope) { minibatch_scope_ = scope; }
  void SetSkipVars(const std::vector<std::string>& skip_vars) {
    skip_vars_ = skip_vars;
  }
  void RunBackward(
      int micro_id, std::unique_ptr<GarbageCollector>&,
      std::unordered_map<const OperatorBase*, std::vector<std::string>>&);
  void RunForward(
      int micro_id, std::unique_ptr<GarbageCollector>&,
      std::unordered_map<const OperatorBase*, std::vector<std::string>>&);
  void RunUpdate(
      std::unique_ptr<GarbageCollector>&,
      std::unordered_map<const OperatorBase*, std::vector<std::string>>&);
  void RunFThenB(std::unique_ptr<GarbageCollector>&);
  void Run1F1B(std::unique_ptr<GarbageCollector>&);

 protected:
  int section_id_;
  int thread_id_;
  int num_microbatches_;
  int num_pipeline_stages_;
  int pipeline_stage_;
  int schedule_mode_;  // 0 for F-then-B and 1 for 1F1B
  std::vector<Scope*> microbatch_scopes_;
  const Scope* minibatch_scope_;

  // skip&backward vars are only used in 1F1B
  std::vector<std::string> skip_vars_;
  std::vector<std::string> backward_send_vars_;

  std::vector<std::unique_ptr<OperatorBase>> ops_;
  std::vector<OperatorBase*> forward_and_lr_ops_;
  std::vector<OperatorBase*> forward_ops_;
  std::vector<OperatorBase*> backward_ops_;
  std::vector<OperatorBase*> optimizer_ops_;
  std::shared_ptr<framework::ProgramDesc> program_;
  std::unordered_map<const OperatorBase*, std::vector<std::string>>
      unused_vars_;
  static uint64_t batch_id_;

  platform::DeviceContext* dev_ctx_ = nullptr;
};
#endif

#if defined(PADDLE_WITH_PSCORE)
class HeterSectionWorker : public DeviceWorker {
 public:
  HeterSectionWorker() {}
  ~HeterSectionWorker() override {}

  void Initialize(const TrainerDesc& desc) override;
  void CreateDeviceResource(const ProgramDesc& main_prog) override{};

  void TrainFiles() override;
  void TrainFilesWithProfiler() override;

  void BindingDataFeedMemory() override {}
  void BindingDataFeedMemory(int micro_id);
  void PrintFetchVars() override;
  const platform::Place& place() const { return place_; }

  void SetDeviceIndex(int tid) override { thread_id_ = tid; }
  void SetThreadNum(int thread_num) { thread_num_ = thread_num; }
  void SetMicrobatchNum(int num) { num_microbatches_ = num; }
  void SetPipelineStageNum(int num) { num_pipeline_stages_ = num; }
  void SetPipelineStage(int stage) { pipeline_stage_ = stage; }
  std::shared_ptr<std::vector<Scope*>> GetMicrobatchScopes() {
    return microbatch_scopes_;
  }
  void SetMicrobatchScopes(
      std::shared_ptr<std::vector<Scope*>> microbatch_scopes) {
    microbatch_scopes_ = microbatch_scopes;
  }
  using SHARED_THREAD_QUEUE = std::shared_ptr<
      ::paddle::framework::BlockingQueue<std::pair<std::string, int>>>;

  SHARED_THREAD_QUEUE GetThreadQueue() { return thread_queue_; }
  void SetThreadQueue(SHARED_THREAD_QUEUE thread_queue) {
    thread_queue_ = thread_queue;
  }
  void CopyParameters(int microbatch_id, const ProgramDesc& program,
                      const platform::Place& place);
  void SetMinibatchScope(Scope* scope) { minibatch_scope_ = scope; }
  void SetTrainerId(int trainer_id) { this->trainer_id_ = trainer_id; }
  void SetTrainers(int trainers) { this->trainers_ = trainers; }
  void CreateMicrobatchScopes();
  void RunForward(int micro_id);
  void RunBackward(int micro_id);
  void RunListen();
  void MiniBatchBarrier();
  void Run();
  void BatchPostProcess();
  void SetDebug(bool debug) { debug_ = debug; }
  Scope* GetThreadScope() override { return minibatch_scope_; }

  // multi-stream
  // #if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  //  void SetStream(const gpuStream_t stream) override {}
  //  void SetEvent(const gpuEvent_t event) override {}
  // #endif

 protected:
  int trainer_id_;
  int trainers_;
  int thread_num_;
  int thread_id_;
  int num_microbatches_;
  int num_pipeline_stages_;
  int pipeline_stage_;
  bool epoch_finish_;

  std::shared_ptr<std::vector<Scope*>> microbatch_scopes_;
  Scope* minibatch_scope_;
  std::vector<int> micro_ids_{};
  std::unique_ptr<OperatorBase> listen_op_{nullptr};
  std::vector<std::unique_ptr<OperatorBase>> forward_ops_;
  std::vector<std::unique_ptr<OperatorBase>> backward_ops_;
  std::shared_ptr<framework::ProgramDesc> program_;
  std::shared_ptr<
      ::paddle::framework::BlockingQueue<std::pair<std::string, int>>>
      thread_queue_;
  static uint64_t batch_id_;
  uint64_t total_ins_num_ = 0;
  platform::DeviceContext* dev_ctx_ = nullptr;

  bool debug_ = false;
  std::vector<double> op_total_time_;
  std::vector<std::string> op_name_;
  platform::Timer timeline_;
  double total_time_ = 0.0;
  double read_time_ = 0.0;
};
#endif

}  // namespace framework
}  // namespace paddle
