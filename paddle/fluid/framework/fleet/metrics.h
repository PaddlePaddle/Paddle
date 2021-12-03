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

#include <memory>
#include <ThreadPool.h>
#include <atomic>
#include <ctime>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"


namespace paddle {
    
namespace framework {

class BasicAucCalculator {
 public:
  BasicAucCalculator() {}
  void init(int table_size);
  void reset();
  // add single data in CPU with LOCK, deprecated
  void add_unlock_data(double pred, int label);
  // add batch data
  void add_data(const float* d_pred, const int64_t* d_label, int batch_size,
                const paddle::platform::Place& place);
  // add mask data
  void add_mask_data(const float* d_pred, const int64_t* d_label,
                     const int64_t* d_mask, int batch_size,
                     const paddle::platform::Place& place);
  void compute();
  int table_size() const { return _table_size; }
  double bucket_error() const { return _bucket_error; }
  double auc() const { return _auc; }
  double mae() const { return _mae; }
  double actual_ctr() const { return _actual_ctr; }
  double predicted_ctr() const { return _predicted_ctr; }
  double size() const { return _size; }
  double rmse() const { return _rmse; }
  // lock and unlock
  std::mutex& table_mutex(void) { return _table_mutex; }

 private:
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

  std::vector<std::shared_ptr<memory::Allocation>> _d_positive;
  std::vector<std::shared_ptr<memory::Allocation>> _d_negative;
  std::vector<std::shared_ptr<memory::Allocation>> _d_abserr;
  std::vector<std::shared_ptr<memory::Allocation>> _d_sqrerr;
  std::vector<std::shared_ptr<memory::Allocation>> _d_pred;

 private:
  void set_table_size(int table_size) { _table_size = table_size; }
  int _table_size;
  std::vector<double> _table[2];
  static constexpr double kRelativeErrorBound = 0.05;
  static constexpr double kMaxSpan = 0.01;
  std::mutex _table_mutex;
};


class Metric {
 public:
  virtual ~Metric() {}

  Metric() {
    fprintf(stdout, "init fleet Metric\n");
  }

  class MetricMsg {
   public:
    MetricMsg() {}
    MetricMsg(const std::string& label_varname, const std::string& pred_varname,
              int metric_phase, int bucket_size = 1000000)
        : label_varname_(label_varname),
          pred_varname_(pred_varname),
          metric_phase_(metric_phase) {
      calculator = new BasicAucCalculator();
      calculator->init(bucket_size);
    }
    virtual ~MetricMsg() {}

    int MetricPhase() const { return metric_phase_; }
    BasicAucCalculator* GetCalculator() { return calculator; }
    
    // add_data
    virtual void add_data(const Scope* exe_scope,
                          const paddle::platform::Place& place) {
      int label_len = 0;
      const int64_t* label_data = NULL;
      int pred_len = 0;
      const float* pred_data = NULL;
      get_data<int64_t>(exe_scope, label_varname_, &label_data, &label_len);
      get_data<float>(exe_scope, pred_varname_, &pred_data, &pred_len);
      PADDLE_ENFORCE_EQ(label_len, pred_len,
                        platform::errors::PreconditionNotMet(
                            "the predict data length should be consistent with "
                            "the label data length"));
      calculator->add_data(pred_data, label_data, label_len, place);
    }
    
    // get_data
    template <class T = float>
    static void get_data(const Scope* exe_scope, const std::string& varname,
                         const T** data, int* len) {
      auto* var = exe_scope->FindVar(varname.c_str());
      PADDLE_ENFORCE_NOT_NULL(
          var, platform::errors::NotFound(
                   "Error: var %s is not found in scope.", varname.c_str()));
      auto& cpu_tensor = var->Get<LoDTensor>();
      *data = cpu_tensor.data<T>();
      *len = cpu_tensor.numel();
    }

    // parse_cmatch_rank
    static inline std::pair<int, int> parse_cmatch_rank(uint64_t x) {
      // first 32 bit store cmatch and second 32 bit store rank
      return std::make_pair(static_cast<int>(x >> 32),
                            static_cast<int>(x & 0xff));
    }

   protected:
    std::string label_varname_;
    std::string pred_varname_;
    int metric_phase_;
    BasicAucCalculator* calculator;
  };

  static std::shared_ptr<Metric> GetInstance() {
    // PADDLE_ENFORCE_EQ(
    //     s_instance_ == nullptr, false,
    //     platform::errors::PreconditionNotMet(
    //         "GetInstance failed in Metric, you should use SetInstance firstly"));
    return s_instance_;
  }

  static std::shared_ptr<Metric> SetInstance() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (nullptr == s_instance_) {
      VLOG(3) << "s_instance_ is null";
      s_instance_.reset(new paddle::framework::Metric());
    } else {
      LOG(WARNING) << "You have already used SetInstance() before";
    }
    return s_instance_;
  }


const std::vector<std::string> GetMetricNameList(
      int metric_phase = -1) const {
    VLOG(0) << "Want to Get metric phase: " << metric_phase;
    if (metric_phase == -1) {
      return metric_name_list_;
    } else {
      std::vector<std::string> ret;
      for (const auto& name : metric_name_list_) {
        const auto iter = metric_lists_.find(name);
        PADDLE_ENFORCE_NE(
            iter, metric_lists_.end(),
            platform::errors::InvalidArgument(
                "The metric name you provided is not registered."));

        if (iter->second->MetricPhase() == metric_phase) {
          VLOG(3) << name << "'s phase is " << iter->second->MetricPhase()
                  << ", we want";
          ret.push_back(name);
        } else {
          VLOG(3) << name << "'s phase is " << iter->second->MetricPhase()
                  << ", not we want";
        }
      }
      return ret;
    }
  }
  int Phase() const { return phase_; }
  int PhaseNum() const { return phase_num_; }
  void FlipPhase() { phase_ = (phase_ + 1) % phase_num_; }
  std::map<std::string, MetricMsg*>& GetMetricList() { return metric_lists_; }

  void InitMetric(const std::string& method, const std::string& name,
                  const std::string& label_varname,
                  const std::string& pred_varname,
                  const std::string& cmatch_rank_varname,
                  const std::string& mask_varname, int metric_phase,
                  const std::string& cmatch_rank_group, bool ignore_rank,
                  int bucket_size = 1000000) {
   if (method == "AucCalculator") {
    metric_lists_.emplace(
        name, new MetricMsg(label_varname, pred_varname, metric_phase,
                            bucket_size));
  // } else if (method == "MultiTaskAucCalculator") {
  //   metric_lists_.emplace(
  //       name, new MultiTaskMetricMsg(label_varname, pred_varname,
  //                                     metric_phase, cmatch_rank_group,
  //                                     cmatch_rank_varname, bucket_size));
  // } else if (method == "CmatchRankAucCalculator") {
  //   metric_lists_.emplace(name, new CmatchRankMetricMsg(
  //                                   label_varname, pred_varname, metric_phase,
  //                                   cmatch_rank_group, cmatch_rank_varname,
  //                                   ignore_rank, bucket_size));
  // } else if (method == "MaskAucCalculator") {
  //   metric_lists_.emplace(
  //       name, new MaskMetricMsg(label_varname, pred_varname, metric_phase,
  //                               mask_varname, bucket_size,
  //                               mode_collect_in_gpu, max_batch_size));
  // } else if (method == "CmatchRankMaskAucCalculator") {
  //   metric_lists_.emplace(name, new CmatchRankMaskMetricMsg(
  //                                   label_varname, pred_varname, metric_phase,
  //                                   cmatch_rank_group, cmatch_rank_varname,
  //                                   ignore_rank, mask_varname, bucket_size));
   } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "PSLIB Metrics only support AucCalculator, MultiTaskAucCalculator, "
        "CmatchRankAucCalculator, MaskAucCalculator and "
        "CmatchRankMaskAucCalculator"));
   }
  metric_name_list_.emplace_back(name);
}

  const std::vector<float> GetMetricMsg(const std::string& name) {
    const auto iter = metric_lists_.find(name);
    PADDLE_ENFORCE_NE(iter, metric_lists_.end(),
                      platform::errors::InvalidArgument(
                          "The metric name you provided is not registered."));
    std::vector<float> metric_return_values_(8, 0.0);
    auto* auc_cal_ = iter->second->GetCalculator();
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
  static std::shared_ptr<Metric> s_instance_;

  // Metric Related
  int phase_ = 1;
  int phase_num_ = 2;
  std::map<std::string, MetricMsg*> metric_lists_;
  std::vector<std::string> metric_name_list_;

};


}

}