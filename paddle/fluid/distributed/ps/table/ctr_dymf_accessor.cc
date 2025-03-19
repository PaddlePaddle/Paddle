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

#include "paddle/fluid/distributed/ps/table/ctr_dymf_accessor.h"

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle::distributed {

int CtrDymfAccessor::Initialize() {
  auto name = _config.embed_sgd_param().name();
  _embed_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embed_sgd_rule->LoadConfig(_config.embed_sgd_param(), 1);

  name = _config.embedx_sgd_param().name();
  _embedx_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embedx_sgd_rule->LoadConfig(_config.embedx_sgd_param(),
                               _config.embedx_dim());
  common_feature_value.optimizer_name = name;

  common_feature_value.embed_sgd_dim = _embed_sgd_rule->Dim();
  common_feature_value.embedx_dim = _config.embedx_dim();
  common_feature_value.embedx_sgd_dim = _embedx_sgd_rule->Dim();
  _show_click_decay_rate = _config.ctr_accessor_param().show_click_decay_rate();
  _ssd_unseenday_threshold =
      _config.ctr_accessor_param().ssd_unseenday_threshold();

  if (_config.ctr_accessor_param().show_scale()) {
    _show_scale = true;
  }
  for (int i = 0; i < _config.ctr_accessor_param().load_filter_slots_size();
       i++) {
    _filtered_slots.insert(_config.ctr_accessor_param().load_filter_slots(i));
    VLOG(0) << "CtrDymfAccessor::Initialize() load filter slot:"
            << _config.ctr_accessor_param().load_filter_slots(i);
  }
  for (int i = 0; i < _config.ctr_accessor_param().save_filter_slots_size();
       i++) {
    _save_filtered_slots.insert(
        _config.ctr_accessor_param().save_filter_slots(i));
    VLOG(0) << "CtrDymfAccessor::Initialize() save filter slot:"
            << _config.ctr_accessor_param().save_filter_slots(i);
  }
  VLOG(0) << " INTO CtrDymfAccessor::Initialize(); embed_sgd_dim:"
          << common_feature_value.embed_sgd_dim
          << " embedx_dim:" << common_feature_value.embedx_dim
          << "  embedx_sgd_dim:" << common_feature_value.embedx_sgd_dim;
  InitAccessorInfo();
  SetTimeDecayRates();
  return 0;
}

void CtrDymfAccessor::InitAccessorInfo() {
  _accessor_info.dim = common_feature_value.Dim();
  _accessor_info.size = common_feature_value.Size();

  auto embedx_dim = _config.embedx_dim();
  VLOG(0) << "InitAccessorInfo embedx_dim:" << embedx_dim;
  _accessor_info.select_dim = 4 + embedx_dim;
  _accessor_info.select_size = _accessor_info.select_dim * sizeof(float);
  _accessor_info.update_dim = 5 + embedx_dim;
  _accessor_info.update_size = _accessor_info.update_dim * sizeof(float);
  _accessor_info.mf_size =
      (embedx_dim + common_feature_value.embedx_sgd_dim) * sizeof(float);
}

void CtrDymfAccessor::SetTimeDecayRates() {
  // 根据unseen_days的天数来初始化_time_decay_rates大小和对应的衰减率
  auto delete_after_unseen_days =
      _config.ctr_accessor_param().delete_after_unseen_days();
  _time_decay_rates.assign(delete_after_unseen_days + 2, 0.0);
  for (int i = 0; i <= delete_after_unseen_days + 1; ++i) {
    _time_decay_rates[i] = pow(_show_click_decay_rate, i);
  }
}

bool CtrDymfAccessor::Shrink(float* value) {
  auto delete_after_unseen_days =
      _config.ctr_accessor_param().delete_after_unseen_days();
  auto delete_threshold = _config.ctr_accessor_param().delete_threshold();

  // time_decay first
  common_feature_value.Show(value) *= _show_click_decay_rate;
  common_feature_value.Click(value) *= _show_click_decay_rate;

  // shrink after
  auto score = ShowClickScore(common_feature_value.Show(value),
                              common_feature_value.Click(value));
  auto unseen_days = common_feature_value.UnseenDays(value);
  if (score < delete_threshold || unseen_days > delete_after_unseen_days) {
    return true;
  }
  return false;
}

bool CtrDymfAccessor::SaveCache(float* value,
                                int param,
                                double global_cache_threshold) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (ShowClickScore(common_feature_value.Show(value),
                     common_feature_value.Click(value)) >= base_threshold &&
      common_feature_value.UnseenDays(value) <= delta_keep_days) {
    return common_feature_value.Show(value) > global_cache_threshold;
  }
  return false;
}

bool CtrDymfAccessor::SaveSSD(float* value) {
  if (_day_id == 0) {
    return true;
  }
  auto unseen_days = common_feature_value.UnseenDays(value);
  if (unseen_days == 0) {
    return true;
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

bool CtrDymfAccessor::FilterSlot(float* value) {
  // 热启时过滤掉_filtered_slots中的feasign
  if (_filtered_slots.find(common_feature_value.Slot(value)) !=
      _filtered_slots.end()) {
    return true;
  }
  return false;
}

bool CtrDymfAccessor::SaveFilterSlot(float* value) {
  // 热启时过滤掉_filtered_slots中的feasign
  if (_save_filtered_slots.find(common_feature_value.Slot(value)) !=
      _save_filtered_slots.end()) {
    return false;
  }
  return true;
}

bool CtrDymfAccessor::Save(float* value, int param) {
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
      if (ShowClickScore(common_feature_value.Show(value),
                         common_feature_value.Click(value)) >= base_threshold &&
          common_feature_value.DeltaScore(value) >= delta_threshold &&
          common_feature_value.UnseenDays(value) <= delta_keep_days) {
        // do this after save, because it must not be modified when retry
        if (param == 2) {
          common_feature_value.DeltaScore(value) = 0;
        }
        return true;
      } else {
        return false;
      }
    }
    // already decayed in shrink
    case 3: {
      // do this after save, because it must not be modified when retry
      // common_feature_value.UnseenDays(value)++;
      return true;
    }
    // save revert batch_model
    case 5: {
      return true;
    }
    default:
      return true;
  }
}

void CtrDymfAccessor::UpdateStatAfterSave(float* value, int param) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (param == 2) {
    delta_threshold = 0;
  }
  switch (param) {
    case 1: {
      if (ShowClickScore(common_feature_value.Show(value),
                         common_feature_value.Click(value)) >= base_threshold &&
          common_feature_value.DeltaScore(value) >= delta_threshold &&
          common_feature_value.UnseenDays(value) <= delta_keep_days) {
        common_feature_value.DeltaScore(value) = 0;
      }
    }
      return;
#ifndef PADDLE_WITH_HETERPS
    case 3: {
      common_feature_value.UnseenDays(value)++;
    }
      return;
#endif
    default:
      return;
  }
}

int32_t CtrDymfAccessor::Create(float** values, size_t num) {
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* value = values[value_item];
#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_HETERPS)
    common_feature_value.UnseenDays(value) = 0;
    common_feature_value.PassId(value) = 0;
#else
    value[common_feature_value.UnseenDaysIndex()] = 0;
#endif
    value[common_feature_value.DeltaScoreIndex()] = 0;
    value[common_feature_value.ShowIndex()] = 0;
    value[common_feature_value.ClickIndex()] = 0;
    value[common_feature_value.SlotIndex()] = -1;
    value[common_feature_value.MfDimIndex()] = -1;
    bool zero_init = _config.ctr_accessor_param().zero_init();
    _embed_sgd_rule->InitValue(
        value + common_feature_value.EmbedWIndex(),
        value + common_feature_value.EmbedG2SumIndex(),
        zero_init);  // adam embed init not zero, adagrad embed init zero;
                     // pglbox set false for adam, gpups set true for adagrad
                     // users can set this in python config, default is true
    _embedx_sgd_rule->InitValue(value + common_feature_value.EmbedxWIndex(),
                                value + common_feature_value.EmbedxG2SumIndex(),
                                false);
  }
  return 0;
}

bool CtrDymfAccessor::NeedExtendMF(float* value) {
  float show = value[common_feature_value.ShowIndex()];
  float click = value[common_feature_value.ClickIndex()];
  float score = (show - click) * _config.ctr_accessor_param().nonclk_coeff() +
                click * _config.ctr_accessor_param().click_coeff();
  return score >= _config.embedx_threshold();
}

bool CtrDymfAccessor::HasMF(int size) {
  return size > common_feature_value.EmbedxG2SumIndex();
}

// from CommonFeatureValue to CtrDymfPullValue
int32_t CtrDymfAccessor::Select(float** select_values,
                                const float** values,
                                size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* select_value = select_values[value_item];
    const float* value = values[value_item];
    select_value[CtrDymfPullValue::ShowIndex()] =
        value[common_feature_value.ShowIndex()];
    select_value[CtrDymfPullValue::ClickIndex()] =
        value[common_feature_value.ClickIndex()];
    select_value[CtrDymfPullValue::EmbedWIndex()] =
        value[common_feature_value.EmbedWIndex()];
    memcpy(select_value + CtrDymfPullValue::EmbedxWIndex(),
           value + common_feature_value.EmbedxWIndex(),
           embedx_dim * sizeof(float));
  }
  return 0;
}

// from CtrDymfPushValue to CtrDymfPushValue
// first dim: item
// second dim: field num
int32_t CtrDymfAccessor::Merge(float** update_values,
                               const float** other_update_values,
                               size_t num) {
  // currently merge in cpu is not supported
  return 0;
}

// from CtrDymfPushValue to CommonFeatureValue
// first dim: item
// second dim: field num
int32_t CtrDymfAccessor::Update(float** update_values,
                                const float** push_values,
                                size_t num) {
  // currently update in cpu is not supported
  return 0;
}

bool CtrDymfAccessor::CreateValue(int stage, const float* value) {
  // stage == 0, pull
  // stage == 1, push
  if (stage == 0) {
    return true;
  } else if (stage == 1) {
    // operation
    auto show = CtrDymfPushValue::Show(const_cast<float*>(value));
    auto click = CtrDymfPushValue::Click(const_cast<float*>(value));
    auto score = ShowClickScore(show, click);
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

float CtrDymfAccessor::ShowClickScore(float show, float click) {
  auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  auto click_coeff = _config.ctr_accessor_param().click_coeff();
  return (show - click) * nonclk_coeff + click * click_coeff;
}

std::string CtrDymfAccessor::ParseToString(const float* v, int param) {
  /*
      float unseen_days;
      float delta_score;
      float show;
      float click;
      float embed_w;
      std::vector<float> embed_g2sum; // float embed_g2sum
      float slot;
      float mf_dim;
      std::<vector>float embedx_g2sum; // float embedx_g2sum
      std::vector<float> embedx_w;
  */
  thread_local std::ostringstream os;
  os.clear();
  os.str("");
#ifdef PADDLE_WITH_PSLIB
  os << common_feature_value.UnseenDays(const_cast<float*>(v)) << " " << v[1]
     << " " << v[2] << " " << v[3] << " " << v[4];
#else
  os << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4];
#endif
  //    << v[5] << " " << v[6];
  for (int i = common_feature_value.EmbedG2SumIndex();
       i < common_feature_value.EmbedxG2SumIndex();
       i++) {
    os << " " << v[i];
  }
  auto show = common_feature_value.Show(const_cast<float*>(v));
  auto click = common_feature_value.Click(const_cast<float*>(v));
  auto score = ShowClickScore(show, click);
  auto mf_dim =
      static_cast<int>(common_feature_value.MfDim(const_cast<float*>(v)));
  if (score >= _config.embedx_threshold() &&
      param > common_feature_value.EmbedxG2SumIndex()) {
    for (auto i = common_feature_value.EmbedxG2SumIndex();
         i < common_feature_value.Dim(mf_dim);
         ++i) {
      os << " " << v[i];
    }
  }
  return os.str();
}

int CtrDymfAccessor::ParseFromString(const std::string& str, float* value) {
  auto ret = paddle::string::str_to_float(str.data(), value);
#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_HETERPS)
  float unseen_day = value[common_feature_value.UnseenDaysIndex()];
  common_feature_value.UnseenDays(value) = (uint16_t)(unseen_day);
  common_feature_value.PassId(value) = 0;
#endif
  PADDLE_ENFORCE_GE(
      ret,
      7UL,
      common::errors::InvalidArgument(
          "Invalid return value. Expect more than 7. But received %d.", ret));
  return ret;
}

void CtrDymfAccessor::SetDayId(int day_id) { _day_id = day_id; }

void CtrDymfAccessor::UpdateTimeDecay(float* value, bool is_update_seen_day) {
  // 根据day_id 来进行show click 衰减和unseen_day 更新;unseen_day
  // 为上次出现的dayid
  if (_day_id == 0) {
    return;
  }
  auto unseen_days = common_feature_value.UnseenDays(value);
  if (unseen_days == 0) {
    common_feature_value.UnseenDays(value) = _day_id;
    return;
  }
  // for the origin load (unseenday = 0 -15)
  if (unseen_days < _config.ctr_accessor_param().delete_after_unseen_days()) {
    // pull
    if (is_update_seen_day) {
      common_feature_value.UnseenDays(value) = _day_id;
      return;
      // save 舍弃原始的unseenday,都变为上一天出现,保证show/click不被重复decay
    } else {
      common_feature_value.UnseenDays(value) = _day_id - 1;
    }
  }
  int16_t day_diff = _day_id - unseen_days;
  if (day_diff < 0) {
    common_feature_value.UnseenDays(value) = _day_id;
    return;
  }
  if (day_diff >= _config.ctr_accessor_param().delete_after_unseen_days()) {
    return;
  }
  common_feature_value.Show(value) *= _time_decay_rates[day_diff];
  common_feature_value.Click(value) *= _time_decay_rates[day_diff];
  if (is_update_seen_day) {
    common_feature_value.UnseenDays(value) = _day_id;
  }
}

#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_HETERPS)
bool CtrDymfAccessor::SaveMemCache(float* value,
                                   int param,
                                   double global_cache_threshold,
                                   uint16_t pass_id) {
  return common_feature_value.Show(value) > global_cache_threshold ||
         common_feature_value.PassId(value) >= pass_id;
}

void CtrDymfAccessor::UpdatePassId(float* value, uint16_t pass_id) {
  common_feature_value.PassId(value) = pass_id;
}
#endif

}  // namespace paddle::distributed
