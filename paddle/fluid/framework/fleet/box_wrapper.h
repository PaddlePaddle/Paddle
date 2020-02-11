/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_BOX_PS
#include <boxps_public.h>
#endif
#include <glog/logging.h>
#include <algorithm>
#include <atomic>
#include <ctime>
#include <deque>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_BOX_PS
class BasicAucCalculator {
 public:
  BasicAucCalculator() {}
  void init(int table_size) { set_table_size(table_size); }
  void reset() {
    for (int i = 0; i < 2; i++) {
      _table[i].assign(_table_size, 0.0);
    }
    _local_abserr = 0;
    _local_sqrerr = 0;
    _local_pred = 0;
  }
  void add_data(double pred, int label) {
    PADDLE_ENFORCE_GE(pred, 0.0, platform::errors::PreconditionNotMet(
                                     "pred should be greater than 0"));
    PADDLE_ENFORCE_LE(pred, 1.0, platform::errors::PreconditionNotMet(
                                     "pred should be lower than 1"));
    PADDLE_ENFORCE_EQ(
        label * label, label,
        platform::errors::PreconditionNotMet(
            "label must be equal to 0 or 1, but its value is: %d", label));
    int pos = std::min(static_cast<int>(pred * _table_size), _table_size - 1);
    PADDLE_ENFORCE_GE(
        pos, 0,
        platform::errors::PreconditionNotMet(
            "pos must be equal or greater than 0, but its value is: %d", pos));
    PADDLE_ENFORCE_LT(
        pos, _table_size,
        platform::errors::PreconditionNotMet(
            "pos must be less than table_size, but its value is: %d", pos));
    std::lock_guard<std::mutex> lock(_table_mutex);
    _local_abserr += fabs(pred - label);
    _local_sqrerr += (pred - label) * (pred - label);
    _local_pred += pred;
    _table[label][pos]++;
  }
  void compute();
  int table_size() const { return _table_size; }
  double bucket_error() const { return _bucket_error; }
  double auc() const { return _auc; }
  double mae() const { return _mae; }
  double actual_ctr() const { return _actual_ctr; }
  double predicted_ctr() const { return _predicted_ctr; }
  double size() const { return _size; }
  double rmse() const { return _rmse; }
  std::vector<double>& get_negative() { return _table[0]; }
  std::vector<double>& get_postive() { return _table[1]; }
  double& local_abserr() { return _local_abserr; }
  double& local_sqrerr() { return _local_sqrerr; }
  double& local_pred() { return _local_pred; }
  void calculate_bucket_error();

 protected:
  double _local_abserr = 0;
  double _local_sqrerr = 0;
  double _local_pred = 0;
  double _auc = 0;
  double _mae = 0;
  double _rmse = 0;
  double _actual_ctr = 0;
  double _predicted_ctr = 0;
  double _size;
  double _bucket_error = 0;

 private:
  void set_table_size(int table_size) {
    _table_size = table_size;
    for (int i = 0; i < 2; i++) {
      _table[i] = std::vector<double>();
    }
    reset();
  }
  int _table_size;
  std::vector<double> _table[2];
  static constexpr double kRelativeErrorBound = 0.05;
  static constexpr double kMaxSpan = 0.01;
  std::mutex _table_mutex;
};

class BoxWrapper {
 public:
  virtual ~BoxWrapper() {}
  BoxWrapper() {}

  void FeedPass(int date, const std::vector<uint64_t>& feasgin_to_box) const;
  void BeginFeedPass(int date, boxps::PSAgentBase** agent) const;
  void EndFeedPass(boxps::PSAgentBase* agent) const;
  void BeginPass() const;
  void EndPass() const;
  void PullSparse(const paddle::platform::Place& place,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size);
  void PushSparseGrad(const paddle::platform::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size, const int batch_size);
  void CopyForPull(const paddle::platform::Place& place, uint64_t** gpu_keys,
                   const std::vector<float*>& values,
                   const boxps::FeatureValueGpu* total_values_gpu,
                   const int64_t* gpu_len, const int slot_num,
                   const int hidden_size, const int64_t total_length);
  void CopyForPush(const paddle::platform::Place& place,
                   const std::vector<const float*>& grad_values,
                   boxps::FeaturePushValueGpu* total_grad_values_gpu,
                   const std::vector<int64_t>& slot_lengths,
                   const int hidden_size, const int64_t total_length,
                   const int batch_size);
  void CopyKeys(const paddle::platform::Place& place, uint64_t** origin_keys,
                uint64_t* total_keys, const int64_t* gpu_len, int slot_num,
                int total_len);
  boxps::PSAgentBase* GetAgent() { return p_agent_; }
  void InitializeGPU(const char* conf_file, const std::vector<int>& slot_vector,
                     const std::vector<std::string>& slot_omit_in_feedpass) {
    if (nullptr != s_instance_) {
      VLOG(3) << "Begin InitializeGPU";
      std::vector<cudaStream_t*> stream_list;
      for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
        VLOG(3) << "before get context i[" << i << "]";
        platform::CUDADeviceContext* context =
            dynamic_cast<platform::CUDADeviceContext*>(
                platform::DeviceContextPool::Instance().Get(
                    platform::CUDAPlace(i)));
        stream_list_[i] = context->stream();
        stream_list.push_back(&stream_list_[i]);
      }
      VLOG(2) << "Begin call InitializeGPU in BoxPS";
      // the second parameter is useless
      s_instance_->boxps_ptr_->InitializeGPU(conf_file, -1, stream_list);
      p_agent_ = boxps::PSAgentBase::GetIns(feedpass_thread_num_);
      p_agent_->Init();
      for (const auto& slot_name : slot_omit_in_feedpass) {
        slot_name_omited_in_feedpass_.insert(slot_name);
      }
      slot_vector_ = slot_vector;
      keys_tensor.resize(platform::GetCUDADeviceCount());
    }
  }

  int GetFeedpassThreadNum() const { return feedpass_thread_num_; }

  void Finalize() {
    VLOG(3) << "Begin Finalize";
    if (nullptr != s_instance_) {
      s_instance_->boxps_ptr_->Finalize();
    }
  }

  void SaveBase(const char* batch_model_path, const char* xbox_model_path,
                boxps::SaveModelStat& stat) {  // NOLINT
    VLOG(3) << "Begin SaveBase";
    if (nullptr != s_instance_) {
      s_instance_->boxps_ptr_->SaveBase(batch_model_path, xbox_model_path,
                                        stat);
    }
  }

  void SaveDelta(const char* xbox_model_path,
                 boxps::SaveModelStat& stat) {  // NOLINT
    VLOG(3) << "Begin SaveDelta";
    if (nullptr != s_instance_) {
      s_instance_->boxps_ptr_->SaveDelta(xbox_model_path, stat);
    }
  }

  static std::shared_ptr<BoxWrapper> GetInstance() {
    if (nullptr == s_instance_) {
      // If main thread is guaranteed to init this, this lock can be removed
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      if (nullptr == s_instance_) {
        VLOG(3) << "s_instance_ is null";
        s_instance_.reset(new paddle::framework::BoxWrapper());
        s_instance_->boxps_ptr_.reset(boxps::BoxPSBase::GetIns());
      }
    }
    return s_instance_;
  }

  const std::unordered_set<std::string>& GetOmitedSlot() const {
    return slot_name_omited_in_feedpass_;
  }

  struct MetricMsg {
   public:
    MetricMsg() {}
    MetricMsg(const std::string& label_varname, const std::string& pred_varname,
              int is_join, int bucket_size = 1000000)
        : label_varname_(label_varname),
          pred_varname_(pred_varname),
          is_join_(is_join) {
      calculator = new BasicAucCalculator();
      calculator->init(bucket_size);
    }
    const std::string& LabelVarname() const { return label_varname_; }
    const std::string& PredVarname() const { return pred_varname_; }
    int IsJoin() const { return is_join_; }
    BasicAucCalculator* GetCalculator() { return calculator; }

   private:
    std::string label_varname_;
    std::string pred_varname_;
    int is_join_;
    BasicAucCalculator* calculator;
  };

  int PassFlag() const { return pass_flag_; }
  void FlipPassFlag() { pass_flag_ = 1 - pass_flag_; }
  bool NeedMetric() const { return need_metric_; }
  std::map<std::string, MetricMsg>& GetMetricList() { return metric_lists_; }

  void InitMetric(const std::string& name, const std::string& label_varname,
                  const std::string& pred_varname, bool is_join,
                  int bucket_size = 1000000) {
    metric_lists_.emplace(name, MetricMsg(label_varname, pred_varname,
                                          is_join ? 1 : 0, bucket_size));
    need_metric_ = true;
  }

  const std::vector<float> GetMetricMsg(const std::string& name) {
    const auto iter = metric_lists_.find(name);
    PADDLE_ENFORCE_NE(iter, metric_lists_.end(),
                      platform::errors::InvalidArgument(
                          "The metric name you provided is not registered."));
    std::vector<float> metric_return_values_(8, 0.0);
    auto* auc_cal_ = iter->second.GetCalculator();
    auc_cal_->calculate_bucket_error();
    auc_cal_->compute();
    metric_return_values_[0] = auc_cal_->auc();
    metric_return_values_[1] = auc_cal_->bucket_error();
    metric_return_values_[2] = auc_cal_->mae();
    metric_return_values_[3] = auc_cal_->rmse();
    metric_return_values_[4] = auc_cal_->actual_ctr();
    metric_return_values_[5] = auc_cal_->predicted_ctr();
    metric_return_values_[6] =
        auc_cal_->actual_ctr() / auc_cal_->predicted_ctr();
    metric_return_values_[7] = auc_cal_->size();
    auc_cal_->reset();
    return metric_return_values_;
  }

 private:
  static cudaStream_t stream_list_[8];
  static std::shared_ptr<boxps::BoxPSBase> boxps_ptr_;
  boxps::PSAgentBase* p_agent_ = nullptr;
  const int feedpass_thread_num_ = 30;  // magic number
  static std::shared_ptr<BoxWrapper> s_instance_;
  std::unordered_set<std::string> slot_name_omited_in_feedpass_;

  // Metric Related
  int pass_flag_ = 1;  // join: 1, update: 0
  bool need_metric_ = false;
  std::map<std::string, MetricMsg> metric_lists_;
  std::vector<int> slot_vector_;
  std::vector<LoDTensor> keys_tensor;  // Cache for pull_sparse
};
#endif

class BoxHelper {
 public:
  explicit BoxHelper(paddle::framework::Dataset* dataset) : dataset_(dataset) {}
  virtual ~BoxHelper() {}

  void BeginPass() {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->BeginPass();
#endif
  }

  void EndPass() {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->EndPass();
#endif
  }
  void LoadIntoMemory() {
    dataset_->LoadIntoMemory();
    FeedPass();
  }
  void PreLoadIntoMemory() {
    dataset_->PreLoadIntoMemory();
    feed_data_thread_.reset(new std::thread([&]() {
      dataset_->WaitPreLoadDone();
      FeedPass();
    }));
  }
  void WaitFeedPassDone() { feed_data_thread_->join(); }

 private:
  Dataset* dataset_;
  std::shared_ptr<std::thread> feed_data_thread_;
  // notify boxps to feed this pass feasigns from SSD to memory
  void FeedPass() {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    auto input_channel_ =
        dynamic_cast<MultiSlotDataset*>(dataset_)->GetInputChannel();
    std::vector<Record> pass_data;
    std::vector<uint64_t> feasign_to_box;
    input_channel_->ReadAll(pass_data);
    for (const auto& ins : pass_data) {
      const auto& feasign_v = ins.uint64_feasigns_;
      for (const auto feasign : feasign_v) {
        feasign_to_box.push_back(feasign.sign().uint64_feasign_);
      }
    }
    input_channel_->Open();
    input_channel_->Write(pass_data);
    input_channel_->Close();
    box_ptr->FeedPass(feasign_to_box);
#endif
  }
};

}  // end namespace framework
}  // end namespace paddle
