// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/ps/table/downpour_ctr_accessor.h"
#include <gflags/gflags.h>
#include "glog/logging.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

int DownpourCtrAccessor::Initialize() {
  auto name = _config.embed_sgd_param().name();
  _embed_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embed_sgd_rule->load_config(_config.embed_sgd_param(), 1);

  name = _config.embedx_sgd_param().name();
  _embedx_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embedx_sgd_rule->load_config(_config.embedx_sgd_param(),
                                _config.embedx_dim());

  _show_click_decay_rate = _config.ctr_accessor_param().show_click_decay_rate();
  _ssd_unseenday_threshold =
      _config.ctr_accessor_param().ssd_unseenday_threshold();
  set_time_decay_rates();
  return 0;
}

void DownpourCtrAccessor::SetTableInfo(AccessorInfo& info) {
  info.dim = Dim();
  info.size = Size();
  info.select_dim = SelectDim();
  info.select_size = SelectSize();
  info.update_dim = UpdateDim();
  info.update_size = UpdateSize();
  info.mf_size = MFSize();
}

size_t DownpourCtrAccessor::GetTableInfo(InfoKey key) {
  switch (key) {
    case DIM:
      return Dim();
    case SIZE:
      return Size();
    case SELECT_DIM:
      return SelectDim();
    case SELECT_SIZE:
      return SelectSize();
    case UPDATE_DIM:
      return UpdateDim();
    case UPDATE_SIZE:
      return UpdateSize();
    case MF_SIZE:
      return MFSize();
    default:
      return 0;
  }
  return 0;
}

size_t DownpourCtrAccessor::Dim() {
  auto embedx_dim = _config.embedx_dim();
  return DownpourCtrFeatureValue::Dim(embedx_dim);
}

size_t DownpourCtrAccessor::DimSize(size_t dim) {
  auto embedx_dim = _config.embedx_dim();
  return DownpourCtrFeatureValue::DimSize(dim, embedx_dim);
}

size_t DownpourCtrAccessor::Size() {
  auto embedx_dim = _config.embedx_dim();
  return DownpourCtrFeatureValue::Size(embedx_dim);
}

size_t DownpourCtrAccessor::MFSize() {
  return (_config.embedx_dim() + 1) * sizeof(float);  // embedx embedx_g2sum
}

// pull value
size_t DownpourCtrAccessor::SelectDim() {
  auto embedx_dim = _config.embedx_dim();
  return 3 + embedx_dim;
}

size_t DownpourCtrAccessor::SelectDimSize(size_t dim) {
  return sizeof(float);
}

size_t DownpourCtrAccessor::SelectSize() {
  return SelectDim() * sizeof(float);
}

// push value
size_t DownpourCtrAccessor::UpdateDim() {
  auto embedx_dim = _config.embedx_dim();
  return 4 + embedx_dim;
}

size_t DownpourCtrAccessor::UpdateDimSize(size_t dim) {
  return sizeof(float);
}

size_t DownpourCtrAccessor::UpdateSize() {
  return UpdateDim() * sizeof(float);
}

bool DownpourCtrAccessor::Shrink(float* value) {
  // auto base_threshold = _config.ctr_accessor_param().base_threshold();
  // auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  // auto delete_threshold = _config.ctr_accessor_param().delete_threshold();
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delete_after_unseen_days =
      _config.ctr_accessor_param().delete_after_unseen_days();
  auto delete_threshold = _config.ctr_accessor_param().delete_threshold();

  // time_decay first
  auto unseen_days = DownpourCtrFeatureValue::unseen_days(value);
  int16_t day_diff = _day_id - unseen_days;
  if (day_diff < 0 || day_diff > delete_after_unseen_days) {
    return true;
  }
  auto show_right =
      DownpourCtrFeatureValue::Show(value) * _time_decay_rates[day_diff];
  auto click_right =
      DownpourCtrFeatureValue::Click(value) * _time_decay_rates[day_diff];

  // shrink after
  auto score = show_click_score(show_right, click_right);
  if (score < delete_threshold) {
    return true;
  }
  return false;
}

void DownpourCtrAccessor::set_day_id(int day_id) { _day_id = day_id; }

int DownpourCtrAccessor::get_day_id() { return _day_id; }

bool DownpourCtrAccessor::save_ssd(float* value) {
  if (_day_id == 0) {
    return true;
  }
  auto unseen_days = DownpourCtrFeatureValue::unseen_days(value);
  if (unseen_days == 0) {
    return false;
  }
  // for the origin load (eg. unseen_days = 0-15)
  if (unseen_days < _config.ctr_accessor_param().delta_keep_days()) {
    unseen_days = _day_id - unseen_days;
  }
  int16_t day_diff = _day_id - unseen_days;
  if (day_diff > _ssd_unseenday_threshold) {
    return true;
  }
  return false;
}

// bool DownpourCtrAccessor::save_cache(
//         float* value, int param, double global_cache_threshold) {
//     auto base_threshold = _config.ctr_accessor_param().base_threshold();
//     auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
//     auto unseen_days = DownpourCtrFeatureValue::unseen_days(value);
//     int16_t day_diff = _day_id - unseen_days;
//     if (show_click_score(DownpourCtrFeatureValue::Show(value),
//     DownpourCtrFeatureValue::Click(value)) >= base_threshold
//         && day_diff <= delta_keep_days) {
//         return DownpourCtrFeatureValue::Show(value) > global_cache_threshold;
//     }
//     return false;
// }

bool DownpourCtrAccessor::Save(float* value, int param) {
  // auto base_threshold = _config.ctr_accessor_param().base_threshold();
  // auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  // auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (param == 2) {
    delta_threshold = 0;
  }
  switch (param) {
    // save all
    case 0: {
      return true;
    }
    // save xbox delta
    case 1:
    // save xbox base
    case 2: {
      auto unseen_days = DownpourCtrFeatureValue::unseen_days(value);
      int16_t day_diff = _day_id - unseen_days;

      auto show_right =
          DownpourCtrFeatureValue::Show(value) * _time_decay_rates[day_diff];
      auto click_right =
          DownpourCtrFeatureValue::Click(value) * _time_decay_rates[day_diff];

      if (show_click_score(show_right, click_right) >= base_threshold &&
          DownpourCtrFeatureValue::delta_score(value) >= delta_threshold &&
          day_diff <= delta_keep_days) {
        // do this after save, because it must not be modified when retry
        if (param == 2) {
          DownpourCtrFeatureValue::delta_score(value) = 0;
        }
        return true;
      } else {
        return false;
      }
    }
    // already decayed in shrink
    case 3: {
      // DownpourCtrFeatureValue::Show(value) *= _show_click_decay_rate;
      // DownpourCtrFeatureValue::Click(value) *= _show_click_decay_rate;
      // do this after save, because it must not be modified when retry
      // DownpourCtrFeatureValue::unseen_days(value)++;
      return true;
    }
    default:
      return true;
  };
}

void DownpourCtrAccessor::UpdateStatAfterSave(float* value, int param) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (param == 2) {
    delta_threshold = 0;
  }
  switch (param) {
    case 1: {
      auto unseen_days = DownpourCtrFeatureValue::unseen_days(value);
      int16_t day_diff = _day_id - unseen_days;
      auto show_right =
          DownpourCtrFeatureValue::Show(value) * _time_decay_rates[day_diff];
      auto click_right =
          DownpourCtrFeatureValue::Click(value) * _time_decay_rates[day_diff];

      if (show_click_score(show_right, click_right) >= base_threshold &&
          DownpourCtrFeatureValue::delta_score(value) >= delta_threshold &&
          day_diff <= delta_keep_days) {
        DownpourCtrFeatureValue::delta_score(value) = 0;
      }
    }
      return;
    //  case 3:
    //     {
    //         DownpourCtrFeatureValue::unseen_days(value)++;
    //     }
    //     return;
    default:
      return;
  };
}

int32_t DownpourCtrAccessor::Create(float** values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* value = values[value_item];
    value[DownpourCtrFeatureValue::unseen_days_index()] = 0;
    value[DownpourCtrFeatureValue::delta_score_index()] = 0;
    value[DownpourCtrFeatureValue::ShowIndex()] = 0;
    value[DownpourCtrFeatureValue::ClickIndex()] = 0;
    value[DownpourCtrFeatureValue::SlotIndex()] = -1;
    _embed_sgd_rule->init_value(
        value + DownpourCtrFeatureValue::Embed_W_Index(),
        value + DownpourCtrFeatureValue::embed_g2sum_index(), true);
    _embedx_sgd_rule->init_value(
        value + DownpourCtrFeatureValue::Embedx_W_Index(),
        value + DownpourCtrFeatureValue::embedx_g2sum_index());
  }
  return 0;
}

bool DownpourCtrAccessor::NeedExtendMF(float* value) {
  float show = value[DownpourCtrFeatureValue::ShowIndex()];
  float click = value[DownpourCtrFeatureValue::ClickIndex()];
  // float score = (show - click) * _config.ctr_accessor_param().nonclk_coeff()
  float score = (show - click) * _config.ctr_accessor_param().nonclk_coeff() +
                click * _config.ctr_accessor_param().click_coeff();
  //+ click * _config.ctr_accessor_param().click_coeff();
  return score >= _config.embedx_threshold();
}

bool DownpourCtrAccessor::HasMF(size_t size) {
  return size > DownpourCtrFeatureValue::embedx_g2sum_index();
}

// from DownpourCtrFeatureValue to DownpourCtrPullValue
int32_t DownpourCtrAccessor::Select(float** select_values, const float** values,
                                    size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* select_value = select_values[value_item];
    float* value = const_cast<float*>(values[value_item]);
    select_value[DownpourCtrPullValue::ShowIndex()] =
        value[DownpourCtrFeatureValue::ShowIndex()];
    select_value[DownpourCtrPullValue::ClickIndex()] =
        value[DownpourCtrFeatureValue::ClickIndex()];
    select_value[DownpourCtrPullValue::Embed_W_Index()] =
        value[DownpourCtrFeatureValue::Embed_W_Index()];
    memcpy(select_value + DownpourCtrPullValue::Embedx_W_Index(),
           value + DownpourCtrFeatureValue::Embedx_W_Index(),
           embedx_dim * sizeof(float));
  }
  return 0;
}

// from DownpourCtrPushValue to DownpourCtrPushValue
// first dim: item
// second dim: field num
int32_t DownpourCtrAccessor::Merge(float** update_values,
                                   const float** other_update_values,
                                   size_t num) {
  auto embedx_dim = _config.embedx_dim();
  size_t total_dim = DownpourCtrPushValue::Dim(embedx_dim);
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* other_update_value = other_update_values[value_item];
    for (auto i = 0u; i < total_dim; ++i) {
      if (i != DownpourCtrPushValue::SlotIndex()) {
        update_value[i] += other_update_value[i];
      }
    }
  }
  return 0;
}

// from DownpourCtrPushValue to DownpourCtrFeatureValue
// first dim: item
// second dim: field num
int32_t DownpourCtrAccessor::Update(float** update_values,
                                    const float** push_values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* push_value = push_values[value_item];
    float push_show = push_value[DownpourCtrPushValue::ShowIndex()];
    float push_click = push_value[DownpourCtrPushValue::ClickIndex()];
    float slot = push_value[DownpourCtrPushValue::SlotIndex()];
    update_value[DownpourCtrFeatureValue::ShowIndex()] += push_show;
    update_value[DownpourCtrFeatureValue::ClickIndex()] += push_click;
    update_value[DownpourCtrFeatureValue::SlotIndex()] = slot;
    update_value[DownpourCtrFeatureValue::delta_score_index()] +=
        (push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
        push_click * _config.ctr_accessor_param().click_coeff();
    //(push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
    // push_click * _config.ctr_accessor_param().click_coeff();
    update_value[DownpourCtrFeatureValue::unseen_days_index()] = 0;
    _embed_sgd_rule->update_value(
        update_value + DownpourCtrFeatureValue::Embed_W_Index(),
        update_value + DownpourCtrFeatureValue::embed_g2sum_index(),
        push_value + DownpourCtrPushValue::Embed_G_Index(), push_show);
    _embedx_sgd_rule->update_value(
        update_value + DownpourCtrFeatureValue::Embedx_W_Index(),
        update_value + DownpourCtrFeatureValue::embedx_g2sum_index(),
        push_value + DownpourCtrPushValue::Embedx_G_Index(), push_show);
  }
  return 0;
}

bool DownpourCtrAccessor::CreateValue(int stage, const float* value) {
  // stage == 0, pull
  // stage == 1, push
  if (stage == 0) {
    return true;
  } else if (stage == 1) {
    auto show = DownpourCtrPushValue::Show(const_cast<float*>(value));
    auto click = DownpourCtrPushValue::Click(const_cast<float*>(value));
    auto score = show_click_score(show, click);
    if (score <= 0) {
      return false;
    }
    if (score >= 1) {
      return true;
    }
    return local_uniform_real_distribution<float>()(local_random_engine()) <
           score;
  } else {
    return true;
  }
}

float DownpourCtrAccessor::show_click_score(float show, float click) {
  // auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  // auto click_coeff = _config.ctr_accessor_param().click_coeff();
  auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  auto click_coeff = _config.ctr_accessor_param().click_coeff();
  return (show - click) * nonclk_coeff + click * click_coeff;
}

std::string DownpourCtrAccessor::ParseToString(const float* v,
                                                 int param_size) {
  thread_local std::ostringstream os;
  os.clear();
  os.str("");
  os << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4] << " "
     << v[5] << " " << v[6];
  auto show = DownpourCtrFeatureValue::Show(const_cast<float*>(v));
  auto click = DownpourCtrFeatureValue::Click(const_cast<float*>(v));
  auto score = show_click_score(show, click);
  if (score >= _config.embedx_threshold() && param_size > 7) {
    os << " " << v[7];
    for (auto i = 0; i < _config.embedx_dim(); ++i) {
      os << " " << v[8 + i];
    }
  }
  return os.str();
}

int DownpourCtrAccessor::ParseFromString(const std::string& str,
                                           float* value) {
  int embedx_dim = _config.embedx_dim();
  float data_buff[Dim()];
  float* data_buff_ptr = data_buff;

  _embedx_sgd_rule->init_value(
      data_buff_ptr + DownpourCtrFeatureValue::Embedx_W_Index(),
      data_buff_ptr + DownpourCtrFeatureValue::embedx_g2sum_index());

  auto str_len = paddle::string::str_to_float(str.data(), data_buff_ptr);
  CHECK(str_len >= 6) << "expect more than 6 real:" << str_len;
  // no slot, embedx
  int value_dim = Dim();
  int embedx_g2sum_index = DownpourCtrFeatureValue::embedx_g2sum_index();
  value[DownpourCtrFeatureValue::SlotIndex()] = -1;
  // other case
  if (str_len == (value_dim - 1)) {
    memcpy(value, data_buff_ptr, (embedx_g2sum_index - 1) * sizeof(float));
    memcpy(value + embedx_g2sum_index, data_buff_ptr + embedx_g2sum_index - 1,
           (embedx_dim + 1) * sizeof(float));
  } else {
    memcpy(value, data_buff_ptr, str_len * sizeof(float));
  }
  if (str_len == (value_dim - 1) || str_len == 6) {
    str_len += 1;
  }
  return str_len;
}

void DownpourCtrAccessor::set_time_decay_rates() {
  //根据unseen_days的天数来初始化_time_decay_rates大小和对应的衰减率
  auto delete_after_unseen_days =
      _config.ctr_accessor_param().delete_after_unseen_days();
  _time_decay_rates.assign(delete_after_unseen_days + 1, 0.0);
  for (int i = 0; i <= delete_after_unseen_days; ++i) {
    _time_decay_rates[i] = pow(_show_click_decay_rate, i);
  }
}

void DownpourCtrAccessor::update_time_decay(float* value,
                                            bool is_update_seen_day) {
  // 根据day_id 来进行show click 衰减和unseen_day 更新;unseen_day
  // 为上次出现的dayid
  if (_day_id == 0) {
    return;
  }
  auto unseen_days = DownpourCtrFeatureValue::unseen_days(value);
  if (unseen_days == 0) {
    DownpourCtrFeatureValue::unseen_days(value) = _day_id;
    return;
  }
  // for the origin load (unseenday = 0 -15)
  if (unseen_days < _config.ctr_accessor_param().delete_after_unseen_days()) {
    // pull
    if (is_update_seen_day) {
      DownpourCtrFeatureValue::unseen_days(value) = _day_id;
      return;
      // save 舍弃原始的unseenday,都变为上一天出现,保证show/click不被重复decay
    } else {
      DownpourCtrFeatureValue::unseen_days(value) = _day_id - 1;
    }
  }
  int16_t day_diff = _day_id - unseen_days;
  if (day_diff < 0) {
    DownpourCtrFeatureValue::unseen_days(value) = _day_id;
    return;
  }
  if (day_diff >= _config.ctr_accessor_param().delete_after_unseen_days()) {
    return;
  }
  DownpourCtrFeatureValue::Show(value) *= _time_decay_rates[day_diff];
  DownpourCtrFeatureValue::Click(value) *= _time_decay_rates[day_diff];
  if (is_update_seen_day) {
    DownpourCtrFeatureValue::unseen_days(value) = _day_id;
  }
}

}  // namespace distributed
}  // namespace paddle
