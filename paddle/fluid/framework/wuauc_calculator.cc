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

#include <sstream>
#include "paddle/fluid/framework/context_callback.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_context.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

namespace paddle {
namespace framework {

void FeedWuaucCalculator::trainer_begin_callback(
    TrainerContextInterface* context) {
  auto* trainer = (DistMultiTrainer*)(context->trainer_);  // NOLINT
  if (tags_.empty()) {
    tags_ = trainer->auc_tags_;
    for (size_t i = 0; i < tags_.size(); ++i) {
      tag2id_[tags_[i]] = i;
    }
  }
  if (targets_.empty()) {
    targets_ = trainer->targets_;
  }
  size_t tag_size_ = tags_.size();
  size_t target_size_ = targets_.size();
  results_.resize(target_size_);

  for (auto& target_res : results_) {
    target_res.user_count_.resize(tag_size_);
    target_res.ins_num_.resize(tag_size_);
    target_res.uauc_sum_.resize(tag_size_);
    target_res.wuauc_sum_.resize(tag_size_);
    target_res.uauc_.resize(tag_size_);
    target_res.wuauc_.resize(tag_size_);
    for (size_t i = 0; i < tag_size_; ++i) {
      target_res.user_count_[i] = 0;
      target_res.ins_num_[i] = 0;
      target_res.uauc_sum_[i] = 0.0;
      target_res.wuauc_sum_[i] = 0.0;
      target_res.uauc_[i] = 0.0;
      target_res.wuauc_[i] = 0.0;
    }
  }
}

LoDTensor* GetTensorByVarName(DownpourWorker* worker,
                              const std::string& var_name, size_t batch_size) {
  Variable* var = worker->thread_scope_->FindVar(var_name);
  if (var == nullptr) {
    return nullptr;
  }
  LoDTensor* tensor = var->GetMutable<LoDTensor>();
  // TODO(paddle-dev) batch size trick to be fixed
  // if (!worker->CheckValidOutput(tensor, batch_size)) {
  //     return nullptr;
  // }
  return tensor;
}

void FeedWuaucCalculator::thread_op_done_callback(
    TrainerContextInterface* context, DeviceWorker* worker_base) {
  if (targets_.empty() || tags_.empty()) {
    return;
  }
  // Get predict/label/ins_data and construct wuauc_data
  auto* worker = (DownpourWorker*)worker_base;  // NOLINT
  auto fetch_config = worker->fetch_config_;
  auto device_reader = worker->device_reader_;
  size_t batch_size = device_reader->GetCurBatchSize();
  auto& ins_data_vec = device_reader->GetInsDataVec();

  PADDLE_ENFORCE(ins_data_vec.size() == batch_size,
                 "ins_data_vec size:%d, batch_size:%d.", ins_data_vec.size(),
                 batch_size);

  // TODO(paddle-dev) PassedByConfig
  std::string label_name("click");
  LoDTensor* label_tensor = GetTensorByVarName(worker, label_name, batch_size);
  PADDLE_ENFORCE(label_tensor, "label tensor %s error.", label_name);
  int64_t* label_val = label_tensor->data<int64_t>();

  std::vector<float*> target_vals;
  for (const auto& target : targets_) {
    const auto& var_name = target.second;
    LoDTensor* target_tensor = GetTensorByVarName(worker, var_name, batch_size);
    PADDLE_ENFORCE(target_tensor, "target tensor %s error.", var_name);
    target_vals.push_back(target_tensor->data<float>());
  }

  for (size_t i = 0; i < batch_size; ++i) {
    auto& ins_data = ins_data_vec[i];
    float label = label_val[i];
    uint64_t uid = ins_data.uid_;
    auto& tags = ins_data.auc_tags_;
    if (label == 0 || label == 1) {
      uint64_t tag_bits = 0;
      for (auto& tag : tags) {
        if (tag2id_.find(tag) != tag2id_.end()) {
          tag_bits |= (1ull << tag2id_[tag]);
        }
      }
      WuaucRecord record;
      record.uid_ = uid;
      record.label_ = label;
      record.bits_ = tag_bits;
      auto& preds = record.preds_;
      for (float* targe_val : target_vals) {
        preds.emplace_back(targe_val[i]);
      }
      std::lock_guard<std::mutex> lock(mut_);
      records_.emplace_back(std::move(record));
    }
  }
}

void FeedWuaucCalculator::trainer_end_callback(
    TrainerContextInterface* context) {
  if (targets_.empty() || tags_.empty()) {
    return;
  }

  ::paddle::framework::Channel<WuaucRecord> data_channel =
      ::paddle::framework::MakeChannel<WuaucRecord>();
  data_channel->Write(std::move(records_));
  data_channel->Close();

  auto out_channel = GlobalShuffle<WuaucRecord>(
      context, data_channel, [](const WuaucRecord& data) -> size_t {
        return std::hash<uint64_t>()(data.uid_);
      });
  records_.clear();
  out_channel->ReadAll(records_);

  PADDLE_ENFORCE(targets_.size() == results_.size(), "target size error.");
  size_t idx = 0;
  for (auto& target : targets_) {
    calculate_auc_1_target(context, records_, idx);
    auto& target_result = results_[idx++];
    std::string result_str;
    for (size_t i = 0; i < tags_.size(); ++i) {
      result_str.append(string::format_string(
          "%s:Tag=%s WUAUC=%.6f UAUC=%.6f UserCount=%d, InsNum=%d\n",
          target.first.c_str(), tags_[i].c_str(), target_result.wuauc_[i],
          target_result.uauc_[i], target_result.user_count_[i],
          target_result.ins_num_[i]));
    }
    rank0_print(result_str, context);
  }
}

int FeedWuaucCalculator::calculate_auc_1_target(
    TrainerContextInterface* context, std::vector<WuaucRecord>& records,
    size_t idx) {
  std::sort(records.begin(), records.end(),
            [idx](const WuaucRecord& lhs, const WuaucRecord& rhs) {
              if (lhs.uid_ == rhs.uid_) {
                if (lhs.preds_[idx] == rhs.preds_[idx]) {
                  return lhs.label_ < rhs.label_;
                } else {
                  return lhs.preds_[idx] > rhs.preds_[idx];
                }
              } else {
                return lhs.uid_ > rhs.uid_;
              }
            });

  auto& target_results = results_[idx];
  uint64_t prev_uid = 0;
  size_t prev_pos = 0;
  WuaucRocData roc_data;
  for (size_t i = 0; i < records.size(); ++i) {
    if (records[i].uid_ != prev_uid) {
      std::vector<WuaucRecord> single_user_recs(records.begin() + prev_pos,
                                                records.begin() + i);
      for (size_t tid = 0; tid < tags_.size(); ++tid) {
        roc_data = calculate_auc_1_tag(single_user_recs, idx, tid);
        if (roc_data.auc_ != -1) {
          double ins_num = (roc_data.tp_ + roc_data.fp_);
          target_results.user_count_[tid] += 1;
          target_results.ins_num_[tid] += ins_num;
          target_results.uauc_sum_[tid] += roc_data.auc_;
          target_results.wuauc_sum_[tid] += roc_data.auc_ * ins_num;
        }
      }
      prev_uid = records[i].uid_;
      prev_pos = i;
    }
  }

  std::vector<WuaucRecord> single_user_recs(records.begin() + prev_pos,
                                            records.end());
  for (size_t tid = 0; tid < tags_.size(); ++tid) {
    roc_data = calculate_auc_1_tag(single_user_recs, idx, tid);
    if (roc_data.auc_ != -1) {
      double ins_num = (roc_data.tp_ + roc_data.fp_);
      target_results.user_count_[tid] += 1;
      target_results.ins_num_[tid] += ins_num;
      target_results.uauc_sum_[tid] += roc_data.auc_;
      target_results.wuauc_sum_[tid] += roc_data.auc_ * ins_num;
    }
  }

  auto* trainer_context = (FeedTrainerContext*)context;  // NOLINT
  auto comm = trainer_context->comm_;
  MPI_Allreduce(MPI_IN_PLACE, target_results.user_count_.data(),
                target_results.user_count_.size(), MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, target_results.ins_num_.data(),
                target_results.ins_num_.size(), MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, target_results.uauc_sum_.data(),
                target_results.uauc_sum_.size(), MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, target_results.wuauc_sum_.data(),
                target_results.wuauc_sum_.size(), MPI_DOUBLE, MPI_SUM, comm);

  for (size_t id = 0; id < tags_.size(); ++id) {
    target_results.uauc_[id] =
        target_results.uauc_sum_[id] / (target_results.user_count_[id] + 1e-10);
    target_results.wuauc_[id] =
        target_results.wuauc_sum_[id] / (target_results.ins_num_[id] + 1e-10);
  }
  return 0;
}

FeedWuaucCalculator::WuaucRocData FeedWuaucCalculator::calculate_auc_1_tag(
    const std::vector<WuaucRecord>& records, size_t target_id, size_t tag_id) {
  double tp = 0.0;
  double fp = 0.0;
  double newtp = 0.0;
  double newfp = 0.0;
  double area = 0.0;
  double auc = -1;
  size_t i = 0;
  while (i < records.size()) {
    if (records[i].bits_ & (1ULL << tag_id)) {
      newtp = tp;
      newfp = fp;
      if (records[i].label_ == 1) {
        newtp += 1;
      } else {
        newfp += 1;
      }
      // check i+1
      while (i < records.size() - 1 &&
             records[i].preds_[target_id] == records[i + 1].preds_[target_id]) {
        if (records[i + 1].bits_ & (1ULL << tag_id)) {
          if (records[i + 1].label_ == 1) {
            newtp += 1;
          } else {
            newfp += 1;
          }
        }
        i += 1;
      }
      area += (newfp - fp) * (tp + newtp) / 2.0;
      tp = newtp;
      fp = newfp;
    }
    i += 1;
  }
  if (tp > 0 && fp > 0) {
    auc = area / (fp * tp + 1e-9);
  } else {
    auc = -1;
  }
  return {tp, fp, auc};
}

}  // namespace framework
}  // namespace paddle
