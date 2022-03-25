// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/fluid/framework/fleet/metrics.h"

#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/lod_tensor.h"

#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_PSCORE)
namespace paddle {
namespace framework {

std::shared_ptr<Metric> Metric::s_instance_ = nullptr;

void BasicAucCalculator::init(int table_size) {
  set_table_size(table_size);

  // init CPU memory
  for (int i = 0; i < 2; i++) {
    _table[i] = std::vector<double>();
  }

  // reset
  reset();
}

void BasicAucCalculator::reset() {
  // reset CPU counter
  for (int i = 0; i < 2; i++) {
    _table[i].assign(_table_size, 0.0);
  }
  _local_abserr = 0;
  _local_sqrerr = 0;
  _local_pred = 0;
}

void BasicAucCalculator::add_data(const float* d_pred, const int64_t* d_label,
                                  int batch_size,
                                  const paddle::platform::Place& place) {
  thread_local std::vector<float> h_pred;
  thread_local std::vector<int64_t> h_label;
  h_pred.resize(batch_size);
  h_label.resize(batch_size);
  memcpy(h_pred.data(), d_pred, sizeof(float) * batch_size);
  memcpy(h_label.data(), d_label, sizeof(int64_t) * batch_size);
  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
    add_unlock_data(h_pred[i], h_label[i]);
  }
}

void BasicAucCalculator::add_unlock_data(double pred, int label) {
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
  _local_abserr += fabs(pred - label);
  _local_sqrerr += (pred - label) * (pred - label);
  _local_pred += pred;
  ++_table[label][pos];
}

// add mask data
void BasicAucCalculator::add_mask_data(const float* d_pred,
                                       const int64_t* d_label,
                                       const int64_t* d_mask, int batch_size,
                                       const paddle::platform::Place& place) {
  thread_local std::vector<float> h_pred;
  thread_local std::vector<int64_t> h_label;
  thread_local std::vector<int64_t> h_mask;
  h_pred.resize(batch_size);
  h_label.resize(batch_size);
  h_mask.resize(batch_size);

  memcpy(h_pred.data(), d_pred, sizeof(float) * batch_size);
  memcpy(h_label.data(), d_label, sizeof(int64_t) * batch_size);
  memcpy(h_mask.data(), d_mask, sizeof(int64_t) * batch_size);

  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
    if (h_mask[i]) {
      add_unlock_data(h_pred[i], h_label[i]);
    }
  }
}

void BasicAucCalculator::compute() {
#if defined(PADDLE_WITH_GLOO)
  double area = 0;
  double fp = 0;
  double tp = 0;

  auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
  if (!gloo_wrapper->IsInitialized()) {
    VLOG(0) << "GLOO is not inited";
    gloo_wrapper->Init();
  }

  if (gloo_wrapper->Size() > 1) {
    auto neg_table = gloo_wrapper->AllReduce(_table[0], "sum");
    auto pos_table = gloo_wrapper->AllReduce(_table[1], "sum");
    for (int i = _table_size - 1; i >= 0; i--) {
      double newfp = fp + neg_table[i];
      double newtp = tp + pos_table[i];
      area += (newfp - fp) * (tp + newtp) / 2;
      fp = newfp;
      tp = newtp;
    }
  } else {
    for (int i = _table_size - 1; i >= 0; i--) {
      double newfp = fp + _table[0][i];
      double newtp = tp + _table[1][i];
      area += (newfp - fp) * (tp + newtp) / 2;
      fp = newfp;
      tp = newtp;
    }
  }

  if (fp < 1e-3 || tp < 1e-3) {
    _auc = -0.5;  // which means all nonclick or click
  } else {
    _auc = area / (fp * tp);
  }

  if (gloo_wrapper->Size() > 1) {
    // allreduce sum
    std::vector<double> local_abserr_vec(1, _local_abserr);
    std::vector<double> local_sqrerr_vec(1, _local_sqrerr);
    std::vector<double> local_pred_vec(1, _local_pred);
    auto global_abserr_vec = gloo_wrapper->AllReduce(local_abserr_vec, "sum");
    auto global_sqrerr_vec = gloo_wrapper->AllReduce(local_sqrerr_vec, "sum");
    auto global_pred_vec = gloo_wrapper->AllReduce(local_pred_vec, "sum");
    _mae = global_abserr_vec[0] / (fp + tp);
    _rmse = sqrt(global_sqrerr_vec[0] / (fp + tp));
    _predicted_ctr = global_pred_vec[0] / (fp + tp);
  } else {
    _mae = _local_abserr / (fp + tp);
    _rmse = sqrt(_local_sqrerr / (fp + tp));
    _predicted_ctr = _local_pred / (fp + tp);
  }
  _actual_ctr = tp / (fp + tp);

  _size = fp + tp;

  calculate_bucket_error();
#endif
}

void BasicAucCalculator::calculate_bucket_error() {
#if defined(PADDLE_WITH_GLOO)
  double last_ctr = -1;
  double impression_sum = 0;
  double ctr_sum = 0.0;
  double click_sum = 0.0;
  double error_sum = 0.0;
  double error_count = 0;
  auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
  if (gloo_wrapper->Size() > 1) {
    auto neg_table = gloo_wrapper->AllReduce(_table[0], "sum");
    auto pos_table = gloo_wrapper->AllReduce(_table[1], "sum");
    for (int i = 0; i < _table_size; i++) {
      double click = pos_table[i];
      double show = neg_table[i] + pos_table[i];
      double ctr = static_cast<double>(i) / _table_size;
      if (fabs(ctr - last_ctr) > kMaxSpan) {
        last_ctr = ctr;
        impression_sum = 0.0;
        ctr_sum = 0.0;
        click_sum = 0.0;
      }
      impression_sum += show;
      ctr_sum += ctr * show;
      click_sum += click;
      double adjust_ctr = ctr_sum / impression_sum;
      double relative_error =
          sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum));
      if (relative_error < kRelativeErrorBound) {
        double actual_ctr = click_sum / impression_sum;
        double relative_ctr_error = fabs(actual_ctr / adjust_ctr - 1);
        error_sum += relative_ctr_error * impression_sum;
        error_count += impression_sum;
        last_ctr = -1;
      }
    }
  } else {
    double* table[2] = {&_table[0][0], &_table[1][0]};
    for (int i = 0; i < _table_size; i++) {
      double click = table[1][i];
      double show = table[0][i] + table[1][i];
      double ctr = static_cast<double>(i) / _table_size;
      if (fabs(ctr - last_ctr) > kMaxSpan) {
        last_ctr = ctr;
        impression_sum = 0.0;
        ctr_sum = 0.0;
        click_sum = 0.0;
      }
      impression_sum += show;
      ctr_sum += ctr * show;
      click_sum += click;
      double adjust_ctr = ctr_sum / impression_sum;
      double relative_error =
          sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum));
      if (relative_error < kRelativeErrorBound) {
        double actual_ctr = click_sum / impression_sum;
        double relative_ctr_error = fabs(actual_ctr / adjust_ctr - 1);
        error_sum += relative_ctr_error * impression_sum;
        error_count += impression_sum;
        last_ctr = -1;
      }
    }
  }
  _bucket_error = error_count > 0 ? error_sum / error_count : 0.0;
#endif
}

void BasicAucCalculator::reset_records() {
  // reset wuauc_records_
  wuauc_records_.clear();
  _user_cnt = 0;
  _size = 0;
  _uauc = 0;
  _wuauc = 0;
}

// add uid data
void BasicAucCalculator::add_uid_data(const float* d_pred,
                                      const int64_t* d_label,
                                      const int64_t* d_uid, int batch_size,
                                      const paddle::platform::Place& place) {
  thread_local std::vector<float> h_pred;
  thread_local std::vector<int64_t> h_label;
  thread_local std::vector<uint64_t> h_uid;
  h_pred.resize(batch_size);
  h_label.resize(batch_size);
  h_uid.resize(batch_size);

  memcpy(h_pred.data(), d_pred, sizeof(float) * batch_size);
  memcpy(h_label.data(), d_label, sizeof(int64_t) * batch_size);
  memcpy(h_uid.data(), d_uid, sizeof(uint64_t) * batch_size);

  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
    add_uid_unlock_data(h_pred[i], h_label[i], static_cast<uint64_t>(h_uid[i]));
  }
}

void BasicAucCalculator::add_uid_unlock_data(double pred, int label,
                                             uint64_t uid) {
  PADDLE_ENFORCE_GE(pred, 0.0, platform::errors::PreconditionNotMet(
                                   "pred should be greater than 0"));
  PADDLE_ENFORCE_LE(pred, 1.0, platform::errors::PreconditionNotMet(
                                   "pred should be lower than 1"));
  PADDLE_ENFORCE_EQ(
      label * label, label,
      platform::errors::PreconditionNotMet(
          "label must be equal to 0 or 1, but its value is: %d", label));

  WuaucRecord record;
  record.uid_ = uid;
  record.label_ = label;
  record.pred_ = pred;
  wuauc_records_.emplace_back(std::move(record));
}

void BasicAucCalculator::computeWuAuc() {
  std::sort(wuauc_records_.begin(), wuauc_records_.end(),
            [](const WuaucRecord& lhs, const WuaucRecord& rhs) {
              if (lhs.uid_ == rhs.uid_) {
                if (lhs.pred_ == rhs.pred_) {
                  return lhs.label_ < rhs.label_;
                } else {
                  return lhs.pred_ > rhs.pred_;
                }
              } else {
                return lhs.uid_ > rhs.uid_;
              }
            });

  WuaucRocData roc_data;
  uint64_t prev_uid = 0;
  size_t prev_pos = 0;
  for (size_t i = 0; i < wuauc_records_.size(); ++i) {
    if (wuauc_records_[i].uid_ != prev_uid) {
      std::vector<WuaucRecord> single_user_recs(
          wuauc_records_.begin() + prev_pos, wuauc_records_.begin() + i);
      roc_data = computeSingelUserAuc(single_user_recs);
      if (roc_data.auc_ != -1) {
        double ins_num = (roc_data.tp_ + roc_data.fp_);
        _user_cnt += 1;
        _size += ins_num;
        _uauc += roc_data.auc_;
        _wuauc += roc_data.auc_ * ins_num;
      }

      prev_uid = wuauc_records_[i].uid_;
      prev_pos = i;
    }
  }

  std::vector<WuaucRecord> single_user_recs(wuauc_records_.begin() + prev_pos,
                                            wuauc_records_.end());
  roc_data = computeSingelUserAuc(single_user_recs);
  if (roc_data.auc_ != -1) {
    double ins_num = (roc_data.tp_ + roc_data.fp_);
    _user_cnt += 1;
    _size += ins_num;
    _uauc += roc_data.auc_;
    _wuauc += roc_data.auc_ * ins_num;
  }
}

BasicAucCalculator::WuaucRocData BasicAucCalculator::computeSingelUserAuc(
    const std::vector<WuaucRecord>& records) {
  double tp = 0.0;
  double fp = 0.0;
  double newtp = 0.0;
  double newfp = 0.0;
  double area = 0.0;
  double auc = -1;
  size_t i = 0;

  while (i < records.size()) {
    newtp = tp;
    newfp = fp;
    if (records[i].label_ == 1) {
      newtp += 1;
    } else {
      newfp += 1;
    }
    // check i+1
    while (i < records.size() - 1 && records[i].pred_ == records[i + 1].pred_) {
      if (records[i + 1].label_ == 1) {
        newtp += 1;
      } else {
        newfp += 1;
      }
      i += 1;
    }
    area += (newfp - fp) * (tp + newtp) / 2.0;
    tp = newtp;
    fp = newfp;
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
#endif
