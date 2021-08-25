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

void FeedPnCalculator::trainer_begin_callback(
    TrainerContextInterface* context) {
  auto* trainer = (DistMultiTrainer*)(context->trainer_);  // NOLINT

  if (pn_targets_.empty()) {
    pn_targets_ = trainer->pn_targets_;
  }

  if (pn_labels_.empty()) {
    pn_labels_ = trainer->pn_labels_;
  }

  if (label_bounds_.empty()) {
    label_bounds_ = trainer->label_bounds_;
  }

  if (tag_names_.empty()) {
    tag_names_ = trainer->tag_names_;
  }

  if (resctype_name_.empty()) {
    resctype_name_ = trainer->resctype_name_;
  }

  if (resc_types_.empty()) {
    resc_types_ = trainer->resc_types_;
  }

  size_t resctype_size_ = resc_types_.size();
  size_t tag_size_ = tag_names_.size();
  size_t target_size_ = pn_targets_.size();
  results_.resize(target_size_);

  for (auto& target_res : results_) {
    target_res.resize(tag_size_);
    target_res.resize(tag_size_);
    target_res.resize(tag_size_);
    target_res.resize(tag_size_);
    target_res.resize(tag_size_);
    target_res.resize(tag_size_);
    target_res.resize(tag_size_);
    target_res.resize(tag_size_);
    for (size_t i = 0; i < tag_size_; ++i) {
      target_res[i].user_count_.resize(resctype_size_ + 1);
      target_res[i].ins_num_.resize(resctype_size_ + 1);
      target_res[i].positive_num_.resize(resctype_size_ + 1);
      target_res[i].negtive_num_.resize(resctype_size_ + 1);
      target_res[i].positive_wnum_.resize(resctype_size_ + 1);
      target_res[i].negtive_wnum_.resize(resctype_size_ + 1);
      target_res[i].final_pn_.resize(resctype_size_ + 1);
      target_res[i].final_wpn_.resize(resctype_size_ + 1);
      for (size_t j = 0; j < resctype_size_ + 1; ++j) {
        target_res[i].user_count_[j] = 0;
        target_res[i].ins_num_[j] = 0;
        target_res[i].positive_num_[j] = 0;
        target_res[i].negtive_num_[j] = 0;
        target_res[i].positive_wnum_[j] = 0.0;
        target_res[i].negtive_wnum_[j] = 0.0;
        target_res[i].final_pn_[j] = 0.0;
        target_res[i].final_wpn_[j] = 0.0;
      }
    }
  }
}

LoDTensor* GetTensorByVarName(DownpourWorker* worker,
                              const std::string& var_name) {
  Variable* var = worker->thread_scope_->FindVar(var_name);
  if (var == nullptr) {
    return nullptr;
  }
  LoDTensor* tensor = var->GetMutable<LoDTensor>();

  return tensor;
}

void FeedPnCalculator::thread_op_done_callback(TrainerContextInterface* context,
                                               DeviceWorker* worker_base) {
  if (pn_targets_.empty() || pn_labels_.empty()) {
    return;
  }

  // Get predict/label/ins_data and construct PnRecord
  auto* worker = (DownpourWorker*)worker_base;  // NOLINT
  auto fetch_config = worker->fetch_config_;
  auto device_reader = worker->device_reader_;
  size_t batch_size = device_reader->GetCurBatchSize();
  auto& ins_data_vec = device_reader->GetInsDataVec();

  PADDLE_ENFORCE(ins_data_vec.size() == batch_size,
                 "ins_data_vec size:%d, batch_size:%d.", ins_data_vec.size(),
                 batch_size);

  std::vector<float*> label_vals;
  for (const auto& label : pn_labels_) {
    const auto& var_name = label.second;
    LoDTensor* label_tensor = GetTensorByVarName(worker, var_name);

    // size_t len = label_tensor->numel();
    // PADDLE_ENFORCE(len == batch_size,
    //           "%s size:%d, batch_size:%d.",
    //           label.first, len, batch_size);

    PADDLE_ENFORCE(label_tensor, "label tensor %s error.", var_name);
    label_vals.push_back(label_tensor->data<float>());
  }

  std::vector<float*> target_vals;
  for (const auto& target : pn_targets_) {
    const auto& var_name = target.second;
    LoDTensor* target_tensor = GetTensorByVarName(worker, var_name);
    PADDLE_ENFORCE(target_tensor, "target tensor %s error.", var_name);
    target_vals.push_back(target_tensor->data<float>());
  }

  std::vector<int64_t*> tag_vals;
  for (const auto& tag_name_ : tag_names_) {
    std::string tag_name(tag_name_);
    // VLOG(0) << "tag_name =" << tag_name;
    LoDTensor* tag_tensor = GetTensorByVarName(worker, tag_name);
    PADDLE_ENFORCE(tag_tensor, "tag tensor %s error.", tag_name);
    // int64_t* tag_val = tag_tensor->data<int64_t>();
    tag_vals.push_back(tag_tensor->data<int64_t>());
  }

  size_t resctype_size_ = resc_types_.size();
  if (resctype_size_ > 0) {
    std::string resctype_name(resctype_name_);
    LoDTensor* resctype_tensor = GetTensorByVarName(worker, resctype_name);
    PADDLE_ENFORCE(resctype_tensor, "resource type tensor %s error.",
                   resctype_name);
    int64_t* resctype_val = resctype_tensor->data<int64_t>();

    for (size_t i = 0; i < batch_size; ++i) {
      auto& ins_data = ins_data_vec[i];
      PnRecord record;
      uint64_t uid = ins_data.uid_;
      record.uid_ = uid;
      auto& labels = record.labels_;
      auto& preds = record.preds_;
      auto& tags = record.tags_;
      for (float* label_val : label_vals) {
        labels.emplace_back(label_val[i]);
      }
      for (float* targe_val : target_vals) {
        preds.emplace_back(targe_val[i]);
      }
      for (int64_t* tag_val : tag_vals) {
        tags.emplace_back(tag_val[i]);
      }
      record.type_ = resctype_val[i];
      std::lock_guard<std::mutex> lock(mut_);
      records_.emplace_back(std::move(record));
    }
  } else {
    for (size_t i = 0; i < batch_size; ++i) {
      auto& ins_data = ins_data_vec[i];
      PnRecord record;
      uint64_t uid = ins_data.uid_;
      record.uid_ = uid;
      auto& labels = record.labels_;
      auto& preds = record.preds_;
      auto& tags = record.tags_;
      for (float* label_val : label_vals) {
        labels.emplace_back(label_val[i]);
      }
      for (float* targe_val : target_vals) {
        preds.emplace_back(targe_val[i]);
      }
      for (int64_t* tag_val : tag_vals) {
        tags.emplace_back(tag_val[i]);
      }
      record.type_ = 999;
      std::lock_guard<std::mutex> lock(mut_);
      records_.emplace_back(std::move(record));
    }
  }
}

void FeedPnCalculator::trainer_end_callback(TrainerContextInterface* context) {
  if (pn_targets_.empty() || pn_labels_.empty()) {
    return;
  }

  ::paddle::framework::Channel<PnRecord> data_channel =
      ::paddle::framework::MakeChannel<PnRecord>();
  data_channel->Write(std::move(records_));
  data_channel->Close();

  auto out_channel = GlobalShuffle<PnRecord>(
      context, data_channel, [](const PnRecord& data) -> size_t {
        return std::hash<uint64_t>()(data.uid_);
      });
  records_.clear();
  out_channel->ReadAll(records_);

  PADDLE_ENFORCE(pn_targets_.size() == results_.size(), "target size error.");

  size_t idx = 0;
  for (auto& target : pn_targets_) {
    calculate_pn(context, records_, idx);
    auto& target_result = results_[idx++];
    std::string result_str;
    for (size_t i = 0; i < tag_names_.size(); ++i) {
      for (size_t j = 0; j < resc_types_.size() + 1; ++j) {
        if (j == 0) {
          result_str.append(string::format_string(
              "%s:Tag=%s ResourceType=Total AveragePN=%.6f AverageWPN=%.6f "
              "InsNum=%zu, PositivePairs=%zu NegativePairs=%zu, "
              "PostiveWNum=%.6f, NegtiveWNum=%.6f\n",
              target.first.c_str(), tag_names_[i].c_str(),
              target_result[i].final_pn_[j], target_result[i].final_wpn_[j],
              target_result[i].ins_num_[j], target_result[i].positive_num_[j],
              target_result[i].negtive_num_[j],
              target_result[i].positive_wnum_[j],
              target_result[i].negtive_wnum_[j]));
        } else {
          result_str.append(string::format_string(
              "%s:Tag=%s ResourceType=%d AveragePN=%.6f AverageWPN=%.6f "
              "InsNum=%zu, PositivePairs=%zu NegativePairs=%zu, "
              "PostiveWNum=%.6f, NegtiveWNum=%.6f\n",
              target.first.c_str(), tag_names_[i].c_str(), resc_types_[j - 1],
              target_result[i].final_pn_[j], target_result[i].final_wpn_[j],
              target_result[i].ins_num_[j], target_result[i].positive_num_[j],
              target_result[i].negtive_num_[j],
              target_result[i].positive_wnum_[j],
              target_result[i].negtive_wnum_[j]));
        }
      }
    }
    rank0_print(result_str, context);
  }
}

int FeedPnCalculator::calculate_pn(TrainerContextInterface* context,
                                   std::vector<PnRecord>& records, size_t idx) {
  std::sort(records.begin(), records.end(),
            [idx](const PnRecord& lhs, const PnRecord& rhs) {
              if (lhs.uid_ == rhs.uid_) {
                if (lhs.preds_[idx] == rhs.preds_[idx]) {
                  return lhs.labels_[idx] < rhs.labels_[idx];
                } else {
                  return lhs.preds_[idx] > rhs.preds_[idx];
                }
              } else {
                return lhs.uid_ > rhs.uid_;
              }
            });

  auto& target_result = results_[idx];
  uint64_t prev_uid = records[0].uid_;
  size_t prev_pos = 0;
  PnData pn_data;

  for (size_t i = 0; i < records.size(); ++i) {
    if (records[i].uid_ != prev_uid) {
      for (size_t tid = 0; tid < tag_names_.size(); ++tid) {
        for (size_t j = 0; j < resc_types_.size() + 1; ++j) {
          pn_data = count_pairs(records, prev_pos, i, idx, tid, j);
          target_result[tid].user_count_[j] += 1;
          target_result[tid].ins_num_[j] += pn_data.ins_num_;
          target_result[tid].positive_num_[j] += pn_data.positive_num_;
          target_result[tid].negtive_num_[j] += pn_data.negtive_num_;
          target_result[tid].positive_wnum_[j] += pn_data.positive_wnum_;
          target_result[tid].negtive_wnum_[j] += pn_data.negtive_wnum_;
        }
      }
      prev_uid = records[i].uid_;
      prev_pos = i;
    }
  }

  for (size_t tid = 0; tid < tag_names_.size(); ++tid) {
    for (size_t j = 0; j < resc_types_.size() + 1; ++j) {
      pn_data = count_pairs(records, prev_pos, records.size(), idx, tid, j);
      target_result[tid].user_count_[j] += 1;
      target_result[tid].ins_num_[j] += pn_data.ins_num_;
      target_result[tid].positive_num_[j] += pn_data.positive_num_;
      target_result[tid].negtive_num_[j] += pn_data.negtive_num_;
      target_result[tid].positive_wnum_[j] += pn_data.positive_wnum_;
      target_result[tid].negtive_wnum_[j] += pn_data.negtive_wnum_;
    }
  }

  auto* trainer_context = (FeedTrainerContext*)context;  // NOLINT
  auto comm = trainer_context->comm_;
  // MPI_Allreduce(MPI_IN_PLACE, &target_result.user_count_,
  //                     1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
  for (size_t tid = 0; tid < tag_names_.size(); ++tid) {
    MPI_Allreduce(MPI_IN_PLACE, target_result[tid].user_count_.data(),
                  target_result[tid].user_count_.size(), MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, target_result[tid].ins_num_.data(),
                  target_result[tid].ins_num_.size(), MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, target_result[tid].positive_num_.data(),
                  target_result[tid].positive_num_.size(),
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, target_result[tid].negtive_num_.data(),
                  target_result[tid].negtive_num_.size(),
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, target_result[tid].positive_wnum_.data(),
                  target_result[tid].positive_wnum_.size(), MPI_DOUBLE, MPI_SUM,
                  comm);
    MPI_Allreduce(MPI_IN_PLACE, target_result[tid].negtive_wnum_.data(),
                  target_result[tid].negtive_wnum_.size(), MPI_DOUBLE, MPI_SUM,
                  comm);
  }

  for (size_t id = 0; id < tag_names_.size(); ++id) {
    for (size_t j = 0; j < resc_types_.size() + 1; ++j) {
      if (target_result[id].negtive_num_[j] == 0) {
        target_result[id].final_pn_[j] = FLT_MAX;
      } else {
        target_result[id].final_pn_[j] =
            (float)((double)target_result[id].positive_num_[j] /  // NOLINT
                    (double)target_result[id].negtive_num_[j]);   // NOLINT
      }

      if (target_result[id].negtive_wnum_[j] <= 0) {
        target_result[id].final_wpn_[j] = FLT_MAX;
      } else {
        target_result[id].final_wpn_[j] =
            (float)(target_result[id].positive_wnum_[j] /  // NOLINT
                    target_result[id].negtive_wnum_[j]);
      }
    }
  }

  return 0;
}

FeedPnCalculator::PnData FeedPnCalculator::count_pairs(
    const std::vector<PnRecord>& recs, size_t start, size_t end, size_t idx,
    size_t tag_id, size_t type_id) {
  size_t positive_num_ = 0;
  size_t negtive_num_ = 0;
  double positive_wnum_ = 0.0;
  double negtive_wnum_ = 0.0;
  size_t ins_num_ = 0;

  if (end <= 0) {
    return {0, positive_num_, negtive_num_, positive_wnum_, negtive_wnum_};
  }

  end = std::min(end, recs.size());

  if (type_id == 0) {
    for (size_t i = start; i < end - 1; ++i) {
      if (recs[i].tags_[tag_id] == 1) {
        ins_num_ += 1;
        float label = recs[i].labels_[idx];
        for (size_t j = i + 1; j < end; ++j) {
          if (recs[j].tags_[tag_id] == 1) {
            double diff =
                fabs(std::min(label, label_bounds_[0]) -
                     std::min(recs[j].labels_[idx], label_bounds_[0]));
            if (label < recs[j].labels_[idx]) {
              negtive_num_ += 1;
              negtive_wnum_ += diff;
            } else {
              positive_num_ += 1;
              positive_wnum_ += diff;
            }
          }
        }
      }
    }
    if (recs[end - 1].tags_[tag_id] == 1) {
      ins_num_ += 1;
    }
  } else {
    for (size_t i = start; i < end - 1; ++i) {
      if (recs[i].tags_[tag_id] == 1 &&
          recs[i].type_ == resc_types_[type_id - 1]) {
        ins_num_ += 1;
        float label = recs[i].labels_[idx];
        for (size_t j = i + 1; j < end; ++j) {
          if (recs[j].tags_[tag_id] == 1 &&
              recs[j].type_ == resc_types_[type_id - 1]) {
            double diff =
                fabs(std::min(label, label_bounds_[0]) -
                     std::min(recs[j].labels_[idx], label_bounds_[0]));
            if (label < recs[j].labels_[idx]) {
              negtive_num_ += 1;
              negtive_wnum_ += diff;
            } else {
              positive_num_ += 1;
              positive_wnum_ += diff;
            }
          }
        }
      }
    }
    if (recs[end - 1].tags_[tag_id] == 1 &&
        recs[end - 1].type_ == resc_types_[type_id - 1]) {
      ins_num_ += 1;
    }
  }

  return {ins_num_, positive_num_, negtive_num_, positive_wnum_, negtive_wnum_};
}

}  // namespace framework
}  // namespace paddle
