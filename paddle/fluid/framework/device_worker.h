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

#include "paddle/common/macros.h"
#include "paddle/fluid/framework/barrier.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/heter_util.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/port.h"
#include "paddle/phi/core/framework/reader.h"
#include "paddle/phi/core/framework/trainer_desc.pb.h"
#include "paddle/phi/core/operators/reader/blocking_queue.h"
#include "paddle/phi/core/platform/timer.h"

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

TEST_API std::string PrintDenseTensor(phi::DenseTensor* tensor,
                                      int64_t start,
                                      int64_t end,
                                      char separator = ',',
                                      bool need_leading_separator = false);
TEST_API void PrintDenseTensor(phi::DenseTensor* tensor,
                               int64_t start,
                               int64_t end,
                               std::string& output_str,  // NOLINT
                               char separator = ',',
                               bool need_leading_separator = false,
                               int num_decimals = 9);
TEST_API std::pair<int64_t, int64_t> GetTensorBound(phi::DenseTensor* tensor,
                                                    int index);
TEST_API bool CheckValidOutput(phi::DenseTensor* tensor, size_t batch_size);

class FleetWrapper;

class PullDenseWorker {
 public:
  virtual ~PullDenseWorker() {}
  virtual void Initialize(const TrainerDesc& param);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void AddStream(const gpuStream_t stream) { copy_streams_.push_back(stream); }
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU)
  void AddPlace(const phi::Place place) { places_.push_back(place); }

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
  std::vector<phi::Place> places_;
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
  virtual void SetWorkerNum(int num UNUSED) {}
  virtual void CacheProgram(const ProgramDesc& main_program UNUSED) {}
  virtual void ProduceTasks() {}
  virtual void GetXpuOpIndex() {}
  virtual void Schedule(int taskid UNUSED) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  virtual void SetStream(const gpuStream_t stream UNUSED) {}
  virtual void SetEvent(const gpuEvent_t event UNUSED) {}
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
  virtual void SetPlace(const phi::Place& place) { place_ = place; }
  virtual void SetReaderPlace(const phi::Place& place) {
    device_reader_->SetPlace(place);
  }
  virtual void SetDeviceContext(phi::DeviceContext* dev_ctx) {
    dev_ctx_ = dev_ctx;
  }

  virtual phi::DeviceContext* GetDeviceContext() { return dev_ctx_; }

  virtual void SetThreadNum(int thread_num) { thread_num_ = thread_num; }

  virtual Scope* GetThreadScope() { return thread_scope_; }
  DataFeed* device_reader_ = nullptr;
  virtual void Finalize() {}

 protected:
  virtual void DumpParam(const Scope& scope, const int batch_id);
  virtual void DumpField(const Scope& scope,
                         int dump_mode,
                         int dump_interval = 10000);
  Scope* root_scope_ = nullptr;
  Scope* thread_scope_;
  phi::Place place_;
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
  int dump_num_decimals_ = 9;
  ChannelWriter<std::string> writer_;
  const size_t tensor_iterator_thread_num = 16;
  phi::DeviceContext* dev_ctx_ = nullptr;
  int thread_num_;
};

class CPUWorkerBase : public DeviceWorker {
 public:
  CPUWorkerBase() {}
  virtual ~CPUWorkerBase() {}
  virtual void SetDeviceIndex(int tid) { thread_id_ = tid; }
  virtual void TrainFiles() = 0;
  virtual void TrainFilesWithProfiler() {}
  virtual void PrintFetchVars() {}
  virtual void CreateDeviceResource(const ProgramDesc& main_prog UNUSED) {}

 protected:
  int thread_id_;
};
class HogwildWorker : public CPUWorkerBase {
  struct OffLoadVarInfo {
    std::vector<std::string> copy_vars;
    std::vector<std::string> backup_vars;
    std::vector<std::pair<std::string, std::string>> cast_vars;
    template <typename TCopyer>
    void CopyInputs(const Scope* root,
                    const phi::Place& place,
                    Scope* scope,
                    TCopyer* copyer);
    template <typename TCopyer>
    void BackUpInputs(Scope* root, Scope* scope, TCopyer* copyer);
  };

 public:
  HogwildWorker() {}
  virtual ~HogwildWorker() {}
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();
  virtual void PrintFetchVars();
  virtual void CreateDeviceResource(const ProgramDesc& main_prog);
  virtual void BindingDataFeedMemory();
  virtual void Finalize();
  template <typename T>
  void SetZero(phi::DenseTensor* tensor, const phi::DenseTensor& root_tensor);

 protected:
  void CreateThreadOperators(const ProgramDesc& program);
  void CreateThreadScope(const ProgramDesc& program);
  // check batch num
  bool CheckBatchNum(int flag);
  bool GetPassEnd(int flag);
  // build thread sharding depends
  void BuildShardingDepends(const ProgramDesc& program);
  int IsParameter(const std::string& name, bool full_match);
  bool IsNeedOffload(const std::string& name);
  size_t AdjustOffloadOps(const ProgramDesc& program);

  std::vector<std::string> op_names_;
  std::vector<std::unique_ptr<OperatorBase>> ops_;
  bool thread_barrier_;
  // Scope* thread_scope_;
  HogwildWorkerParameter param_;
  std::vector<std::string> skip_ops_;
  std::map<std::string, int> stat_var_name_map_;
  static std::atomic<bool> quit_flag_;
  DenseTensor sync_stat_;
  // skip vars
  std::vector<std::string> skip_vars_;
  std::unordered_map<const OperatorBase*, std::vector<std::string>>
      unused_vars_;
  int ring_id_ = 0;
  int nccl_rank_id_ = 0;
  std::unordered_map<std::string, int> params2rootid_;
  std::multiset<std::string> remove_vars_;
  std::multiset<std::string> unpersist_vars_;
  std::multiset<std::string> persist_param_vars_;
  std::multiset<OpDesc*> remove_ops_;
  std::vector<std::string> need_copy_vars_;
  std::vector<std::string> shard_dump_params_;
  std::vector<std::string> shard_dump_fields_;
  std::multiset<std::string> free_param_vars_;
  bool is_multi_node_ = false;
  bool sharding_mode_ = false;
  bool enable_adjust_op_order_ = false;
  // offload vars
  bool is_offload_communication_ = false;
  bool is_offload_param_ = false;
  std::vector<std::string> offload_exts_;
  std::multiset<std::string> offload_names_;
  std::unordered_map<const OperatorBase*, OffLoadVarInfo> offload_vars_;
  // enable MixedPrecision
  std::unordered_map<std::string, std::string> cast_fp16_vars_;
  std::unordered_map<std::string, std::string> param_cast_vars_;
  std::unordered_map<std::string, std::string> need_cast_vars_;
  bool use_ps_gpu_ = false;
  bool use_gpu_graph_ = false;
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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
class SectionWorker : public DeviceWorker {
 public:
  SectionWorker() {}
  ~SectionWorker() override {}

  void Initialize(const TrainerDesc& desc) override;
  void PrepareUnusedVar();

  void BindingDataFeedMemory() override {}
  void CreateDeviceResource(const ProgramDesc& main_prog UNUSED) override{};

  void TrainFiles() override;
  void TrainFilesWithProfiler() override{};

  void PrintFetchVars() override {}

  const phi::Place& place() const { return place_; }

  void SetDeviceIndex(int tid UNUSED) override {}
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
      int micro_id,
      std::unique_ptr<GarbageCollector>&,
      std::unordered_map<const OperatorBase*, std::vector<std::string>>&);
  void RunForward(
      int micro_id,
      std::unique_ptr<GarbageCollector>&,
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

  phi::DeviceContext* dev_ctx_ = nullptr;
};
#endif

}  // namespace framework
}  // namespace paddle
