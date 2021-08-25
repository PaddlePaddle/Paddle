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
#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/framework/trainer_context.h"

namespace paddle {
namespace framework {

// class FeedInstanceDumper : public ThreadOpDoneCallBack {
// public:
//     virtual ~FeedInstanceDumper() {}
//     virtual void Clean() override { batch_count_ = 0; }
//     virtual void callback(TrainerContextInterface*, DeviceWorker*) override;
// private:
//     std::atomic<uint64_t> batch_count_;
// };

class FeedAucCalculator : public ContextCallBackGroup {
 public:
  virtual ~FeedAucCalculator() {}
  void Clean() override { instance_count_ = 0; }
  void thread_pulled_callback(TrainerContextInterface*, DeviceWorker*) override;
  void trainer_end_callback(TrainerContextInterface*) override;

 private:
  std::atomic<uint64_t> instance_count_;
};

class FeedWuaucCalculator : public ContextCallBackGroup {
 public:
  struct WuaucRecord {
    uint64_t uid_;
    float label_;
    uint64_t bits_;
    std::vector<float> preds_;
  };

  struct WuaucResult {
    std::vector<int> user_count_;
    std::vector<int> ins_num_;
    std::vector<double> uauc_sum_;
    std::vector<double> wuauc_sum_;
    std::vector<double> uauc_;
    std::vector<double> wuauc_;
  };

  struct WuaucRocData {
    double tp_;
    double fp_;
    double auc_;
  };

 public:
  virtual ~FeedWuaucCalculator() {}
  void Clean() override {
    tags_.clear();
    records_.clear();
    targets_.clear();
  }
  void thread_op_done_callback(TrainerContextInterface*,
                               DeviceWorker*) override;
  void trainer_begin_callback(TrainerContextInterface*) override;
  void trainer_end_callback(TrainerContextInterface*) override;

 private:
  int shuffle(TrainerContextInterface*);
  int calculate_auc_1_target(TrainerContextInterface*,
                             std::vector<WuaucRecord>&, size_t);
  WuaucRocData calculate_auc_1_tag(const std::vector<WuaucRecord>&, size_t,
                                   size_t);

  std::mutex mut_;
  std::vector<std::string> tags_;
  std::map<std::string, size_t> tag2id_;
  std::vector<WuaucResult> results_;
  std::map<std::string, std::string> targets_;
  std::vector<WuaucRecord> records_;
};

class FeedPnCalculator : public ContextCallBackGroup {
 public:
  struct PnRecord {
    uint64_t uid_;
    std::vector<float> labels_;
    std::vector<float> preds_;
    std::vector<int64_t> tags_;
    int64_t type_;
  };

  struct PnResult {
    std::vector<size_t> user_count_;
    std::vector<size_t> ins_num_;
    std::vector<size_t> positive_num_;
    std::vector<size_t> negtive_num_;
    std::vector<double> positive_wnum_;
    std::vector<double> negtive_wnum_;
    std::vector<float> final_pn_;
    std::vector<float> final_wpn_;
    // size_t user_count_;
    // size_t ins_num_;
    // size_t positive_num_;
    // size_t negtive_num_;
    // double positive_wnum_;
    // double negtive_wnum_;
    // float final_pn_;
    // float final_wpn_;
  };

  struct PnData {
    size_t ins_num_;
    size_t positive_num_;
    size_t negtive_num_;
    double positive_wnum_;
    double negtive_wnum_;
  };

 public:
  virtual ~FeedPnCalculator() {}
  void Clean() override {
    records_.clear();
    pn_targets_.clear();
    pn_labels_.clear();
    label_bounds_.clear();
  }
  void thread_op_done_callback(TrainerContextInterface*,
                               DeviceWorker*) override;
  void trainer_begin_callback(TrainerContextInterface*) override;
  void trainer_end_callback(TrainerContextInterface*) override;

 private:
  int shuffle(TrainerContextInterface*);
  int calculate_pn(TrainerContextInterface*, std::vector<PnRecord>&, size_t);
  PnData count_pairs(const std::vector<PnRecord>&, size_t, size_t, size_t,
                     size_t, size_t);

  std::mutex mut_;
  std::vector<std::vector<PnResult>> results_;
  std::map<std::string, std::string> pn_targets_;
  std::map<std::string, std::string> pn_labels_;
  std::vector<PnRecord> records_;
  std::vector<float> label_bounds_;
  std::vector<std::string> tag_names_;
  std::string resctype_name_;
  std::vector<int64_t> resc_types_;
};

template <class AR>
paddle::framework::Archive<AR>& operator<<(
    paddle::framework::Archive<AR>& ar,
    const FeedWuaucCalculator::WuaucRecord& r) {
  ar << r.uid_;
  ar << r.label_;
  ar << r.bits_;
  ar << r.preds_;
  return ar;
}

template <class AR>
paddle::framework::Archive<AR>& operator>>(
    paddle::framework::Archive<AR>& ar, FeedWuaucCalculator::WuaucRecord& r) {
  ar >> r.uid_;
  ar >> r.label_;
  ar >> r.bits_;
  ar >> r.preds_;
  return ar;
}

template <class AR>
paddle::framework::Archive<AR>& operator<<(
    paddle::framework::Archive<AR>& ar, const FeedPnCalculator::PnRecord& r) {
  ar << r.uid_;
  ar << r.labels_;
  ar << r.preds_;
  ar << r.tags_;
  ar << r.type_;
  return ar;
}

template <class AR>
paddle::framework::Archive<AR>& operator>>(paddle::framework::Archive<AR>& ar,
                                           FeedPnCalculator::PnRecord& r) {
  ar >> r.uid_;
  ar >> r.labels_;
  ar >> r.preds_;
  ar >> r.tags_;
  ar >> r.type_;
  return ar;
}

}  // namespace framework
}  // namespace paddle
