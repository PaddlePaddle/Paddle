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

#include <glog/logging.h>
#include <algorithm>
#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/platform/timer.h"
#ifdef PADDLE_WITH_BOX_PS
#include <boxps_public.h>
#endif
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

#define PADDLEBOX_LOG VLOG(0) << "PaddleBox: "
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
    PADDLE_ENFORCE((label == 0 || label == 1),
                   "label must be equal to 0 or 1, but its value is: %d",
                   label);
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
#ifdef PADDLE_WITH_BOX_PS
  void CopyForPull(const paddle::platform::Place& place,
                   const std::vector<const uint64_t*>& keys,
                   const std::vector<float*>& values,
                   const abacus::FeatureValueGpu* total_values_gpu,
                   const std::vector<int64_t>& slot_lengths,
                   const int hidden_size, const int64_t total_length);
  void CopyForPush(const paddle::platform::Place& place,
                   const std::vector<const float*>& grad_values,
                   abacus::FeaturePushValueGpu* total_grad_values_gpu,
                   const std::vector<int64_t>& slot_lengths,
                   const int hidden_size, const int64_t total_length,
                   const int batch_size);
#endif
  void InitializeGPU(const char* conf_file,
                     const std::vector<std::string>& slot_omit_in_feedpass) {
    if (nullptr != s_instance_) {
      PADDLEBOX_LOG << "Begin InitializeGPU";
#ifdef PADDLE_WITH_BOX_PS
      std::vector<cudaStream_t*> stream_list;
      for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
        VLOG(3) << "before get context i[" << i << "]";
        platform::CUDADeviceContext* context =
            dynamic_cast<platform::CUDADeviceContext*>(
                platform::DeviceContextPool::Instance().Get(
                    platform::CUDAPlace(i)));
        PADDLEBOX_LOG << "after get cuda context for card [" << i << "]";
        PADDLE_ENFORCE_EQ(context == nullptr, false, "context is nullptr");

        stream_list_[i] = context->stream();
        stream_list.push_back(&stream_list_[i]);
      }
      PADDLEBOX_LOG << "call InitializeGPU in boxps";
      // the second parameter is useless
      s_instance_->boxps_ptr_->InitializeGPU(conf_file, -1, stream_list);
      PADDLEBOX_LOG << "return from InitializeGPU in boxps";
#endif
      for (const auto& slot_name : slot_omit_in_feedpass) {
        slot_name_omited_in_feedpass_.insert(slot_name);
      }
    }
  }
  void Finalize() {
#ifdef PADDLE_WITH_BOX_PS
    if (nullptr != s_instance_) {
      s_instance_->boxps_ptr_->Finalize();
    }
#endif
  }
  void SaveModel() const { printf("will be implemented soon\n"); }

  static std::shared_ptr<BoxWrapper> GetInstance() {
    if (nullptr == s_instance_) {
      // If main thread is guaranteed to init this, this lock can be removed
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      if (nullptr == s_instance_) {
        VLOG(3) << "s_instance_ is null";
        s_instance_.reset(new paddle::framework::BoxWrapper());

#ifdef PADDLE_WITH_BOX_PS
        s_instance_->boxps_ptr_.reset(boxps::BoxPSBase::GetIns());
#endif
      }
    }
    return s_instance_;
  }

  // Other function
  const std::set<std::string>& GetOmitedSlot() const {
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
                      "The metric name you provided is not registered.");
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
#ifdef PADDLE_WITH_BOX_PS
  static cudaStream_t stream_list_[8];
  static std::shared_ptr<boxps::BoxPSBase> boxps_ptr_;
#endif
  static std::shared_ptr<BoxWrapper> s_instance_;
  std::set<std::string> slot_name_omited_in_feedpass_;

 private:
  // Metric Related
  int pass_flag_ = 1;  // join: 1, update: 0
  bool need_metric_ = false;
  std::map<std::string, MetricMsg> metric_lists_;
};

class BoxHelper {
 public:
  explicit BoxHelper(paddle::framework::Dataset* dataset) : dataset_(dataset) {}
  virtual ~BoxHelper() {}

  void SetDate(int year, int month, int day) {
    year_ = year;
    month_ = month;
    day_ = day;
  }
  void BeginPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->BeginPass();
  }

  void EndPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->EndPass();
  }
  void LoadIntoMemory() {
    platform::Timer timer;
    PADDLEBOX_LOG << "Begin LoadIntoMemory(), dataset[" << dataset_ << "]";
    timer.Start();
    dataset_->LoadIntoMemory();
    timer.Pause();
    PADDLEBOX_LOG << "download + parse cost: " << timer.ElapsedSec() << "s";

    timer.Start();
    FeedPass();
    timer.Pause();
    PADDLEBOX_LOG << "FeedPass cost: " << timer.ElapsedSec() << " s";
    PADDLEBOX_LOG << "End LoadIntoMemory(), dataset[" << dataset_ << "]";
  }
  void PreLoadIntoMemory() {
    dataset_->PreLoadIntoMemory();
    feed_data_thread_.reset(new std::thread([&]() {
      dataset_->WaitPreLoadDone();
      FeedPass();
    }));
    VLOG(3) << "After PreLoadIntoMemory()";
  }
  void WaitFeedPassDone() { feed_data_thread_->join(); }

 private:
  Dataset* dataset_;
  std::shared_ptr<std::thread> feed_data_thread_;
  int year_;
  int month_;
  int day_;
  // notify boxps to feed this pass feasigns from SSD to memory
  void FeedPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    auto input_channel_ =
        dynamic_cast<MultiSlotDataset*>(dataset_)->GetInputChannel();
    std::vector<Record> pass_data;
    std::vector<uint64_t> feasign_to_box;
    input_channel_->ReadAll(pass_data);

    // get feasigns that FeedPass doesn't need
    const std::set<std::string>& slot_name_omited_in_feedpass_ =
        box_ptr->GetOmitedSlot();
    std::set<int> slot_id_omited_in_feedpass_;
    const auto& all_readers = dataset_->GetReaders();
    PADDLE_ENFORCE_GT(all_readers.size(), 0,
                      platform::errors::PreconditionNotMet(
                          "Readers number must be greater than 0."));
    const auto& all_slots_name = all_readers[0]->GetAllSlotAlias();
    for (size_t i = 0; i < all_slots_name.size(); ++i) {
      if (slot_name_omited_in_feedpass_.find(all_slots_name[i]) !=
          slot_name_omited_in_feedpass_.end()) {
        slot_id_omited_in_feedpass_.insert(i);
      }
    }
    for (const auto& ins : pass_data) {
      const auto& feasign_v = ins.uint64_feasigns_;
      for (const auto feasign : feasign_v) {
        if (slot_id_omited_in_feedpass_.find(feasign.slot()) !=
            slot_id_omited_in_feedpass_.end()) {
          continue;
        }
        feasign_to_box.push_back(feasign.sign().uint64_feasign_);
      }
    }
    input_channel_->Open();
    input_channel_->Write(pass_data);
    input_channel_->Close();
    PADDLEBOX_LOG << "call boxps feedpass";

    struct std::tm b;
    b.tm_year = year_ - 1900;
    b.tm_mon = month_ - 1;
    b.tm_mday = day_;
    b.tm_min = b.tm_hour = b.tm_sec = 0;
    std::time_t x = std::mktime(&b);
    box_ptr->FeedPass(x / 86400, feasign_to_box);
    PADDLEBOX_LOG << "return from boxps feedpass";
  }
};

}  // end namespace framework
}  // end namespace paddle
