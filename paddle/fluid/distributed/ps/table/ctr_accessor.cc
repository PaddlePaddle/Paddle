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

#include "paddle/fluid/distributed/ps/table/ctr_accessor.h"

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle::distributed {

int CtrCommonAccessor::Initialize() {
  auto name = _config.embed_sgd_param().name();
  _embed_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embed_sgd_rule->LoadConfig(_config.embed_sgd_param(), 1);

  name = _config.embedx_sgd_param().name();
  _embedx_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embedx_sgd_rule->LoadConfig(_config.embedx_sgd_param(),
                               _config.embedx_dim());

  common_feature_value.embed_sgd_dim = _embed_sgd_rule->Dim();
  common_feature_value.embedx_dim = _config.embedx_dim();
  common_feature_value.embedx_sgd_dim = _embedx_sgd_rule->Dim();
  _show_click_decay_rate = _config.ctr_accessor_param().show_click_decay_rate();
  _ssd_unseenday_threshold =
      _config.ctr_accessor_param().ssd_unseenday_threshold();

  if (_config.ctr_accessor_param().show_scale()) {
    _show_scale = true;
  }

  InitAccessorInfo();
  return 0;
}

void CtrCommonAccessor::InitAccessorInfo() {
  _accessor_info.dim = common_feature_value.Dim();
  _accessor_info.size = common_feature_value.Size();

  auto embedx_dim = _config.embedx_dim();
  _accessor_info.select_dim = 3 + embedx_dim;
  _accessor_info.select_size = _accessor_info.select_dim * sizeof(float);
  _accessor_info.update_dim = 4 + embedx_dim;
  _accessor_info.update_size = _accessor_info.update_dim * sizeof(float);
  _accessor_info.mf_size =
      (embedx_dim + common_feature_value.embedx_sgd_dim) * sizeof(float);
}

bool CtrCommonAccessor::Shrink(float* value) {
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

bool CtrCommonAccessor::SaveCache(float* value,
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

bool CtrCommonAccessor::SaveSSD(float* value) {
  if (common_feature_value.UnseenDays(value) > _ssd_unseenday_threshold) {
    return true;
  }
  return false;
}

bool CtrCommonAccessor::Save(float* value, int param) {
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

void CtrCommonAccessor::UpdateStatAfterSave(float* value, int param) {
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
    case 3: {
      common_feature_value.UnseenDays(value)++;
    }
      return;
    default:
      return;
  }
}

int32_t CtrCommonAccessor::Create(float** values, size_t num) {
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* value = values[value_item];
    value[common_feature_value.UnseenDaysIndex()] = 0;
    value[common_feature_value.DeltaScoreIndex()] = 0;
    value[common_feature_value.ShowIndex()] = 0;
    value[common_feature_value.ClickIndex()] = 0;
    value[common_feature_value.SlotIndex()] = -1;
    bool zero_init = _config.ctr_accessor_param().zero_init();
    _embed_sgd_rule->InitValue(value + common_feature_value.EmbedWIndex(),
                               value + common_feature_value.EmbedG2SumIndex(),
                               zero_init);
    _embedx_sgd_rule->InitValue(value + common_feature_value.EmbedxWIndex(),
                                value + common_feature_value.EmbedxG2SumIndex(),
                                false);
  }
  return 0;
}

bool CtrCommonAccessor::NeedExtendMF(float* value) {
  float show = value[common_feature_value.ShowIndex()];
  float click = value[common_feature_value.ClickIndex()];
  float score = (show - click) * _config.ctr_accessor_param().nonclk_coeff() +
                click * _config.ctr_accessor_param().click_coeff();
  return score >= _config.embedx_threshold();
}

bool CtrCommonAccessor::HasMF(int size) {
  return size > common_feature_value.EmbedxG2SumIndex();
}

// from CommonFeatureValue to CtrCommonPullValue
int32_t CtrCommonAccessor::Select(float** select_values,
                                  const float** values,
                                  size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* select_value = select_values[value_item];
    const float* value = values[value_item];
    select_value[CtrCommonPullValue::ShowIndex()] =
        value[common_feature_value.ShowIndex()];
    select_value[CtrCommonPullValue::ClickIndex()] =
        value[common_feature_value.ClickIndex()];
    select_value[CtrCommonPullValue::EmbedWIndex()] =
        value[common_feature_value.EmbedWIndex()];
    memcpy(select_value + CtrCommonPullValue::EmbedxWIndex(),
           value + common_feature_value.EmbedxWIndex(),
           embedx_dim * sizeof(float));
  }
  return 0;
}

// from CtrCommonPushValue to CtrCommonPushValue
// first dim: item
// second dim: field num
int32_t CtrCommonAccessor::Merge(float** update_values,
                                 const float** other_update_values,
                                 size_t num) {
  auto embedx_dim = _config.embedx_dim();
  int total_dim = CtrCommonPushValue::Dim(embedx_dim);
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* other_update_value = other_update_values[value_item];
    for (int i = 0; i < total_dim; ++i) {
      if (i != CtrCommonPushValue::SlotIndex()) {
        update_value[i] += other_update_value[i];
      }
    }
  }
  return 0;
}

// from CtrCommonPushValue to CommonFeatureValue
// first dim: item
// second dim: field num
int32_t CtrCommonAccessor::Update(float** update_values,
                                  const float** push_values,
                                  size_t num) {
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* push_value = push_values[value_item];
    float push_show = push_value[CtrCommonPushValue::ShowIndex()];
    float push_click = push_value[CtrCommonPushValue::ClickIndex()];
    float slot = push_value[CtrCommonPushValue::SlotIndex()];
    update_value[common_feature_value.ShowIndex()] += push_show;
    update_value[common_feature_value.ClickIndex()] += push_click;
    update_value[common_feature_value.SlotIndex()] = slot;
    update_value[common_feature_value.DeltaScoreIndex()] +=
        (push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
        push_click * _config.ctr_accessor_param().click_coeff();
    update_value[common_feature_value.UnseenDaysIndex()] = 0;
    // TODO(zhaocaibei123): add configure show_scale
    if (!_show_scale) {
      push_show = 1;
    }
    VLOG(3) << "accessor show scale:" << _show_scale
            << ", push_show:" << push_show;
    _embed_sgd_rule->UpdateValue(
        update_value + common_feature_value.EmbedWIndex(),
        update_value + common_feature_value.EmbedG2SumIndex(),
        push_value + CtrCommonPushValue::EmbedGIndex(),
        push_show);
    _embedx_sgd_rule->UpdateValue(
        update_value + common_feature_value.EmbedxWIndex(),
        update_value + common_feature_value.EmbedxG2SumIndex(),
        push_value + CtrCommonPushValue::EmbedxGIndex(),
        push_show);
  }
  return 0;
}

bool CtrCommonAccessor::CreateValue(int stage, const float* value) {
  // stage == 0, pull
  // stage == 1, push
  if (stage == 0) {
    return true;
  } else if (stage == 1) {
    // operation
    auto show = CtrCommonPushValue::Show(const_cast<float*>(value));
    auto click = CtrCommonPushValue::Click(const_cast<float*>(value));
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

float CtrCommonAccessor::ShowClickScore(float show, float click) {
  auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  auto click_coeff = _config.ctr_accessor_param().click_coeff();
  return (show - click) * nonclk_coeff + click * click_coeff;
}

std::string CtrCommonAccessor::ParseToString(const float* v, int param) {
  thread_local std::ostringstream os;
  os.clear();
  os.str("");
  os << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4] << " "
     << v[5];
  for (int i = common_feature_value.EmbedG2SumIndex();
       i < common_feature_value.EmbedxWIndex();
       i++) {
    os << " " << v[i];
  }
  auto show = common_feature_value.Show(const_cast<float*>(v));
  auto click = common_feature_value.Click(const_cast<float*>(v));
  auto score = ShowClickScore(show, click);
  if (score >= _config.embedx_threshold() &&
      param > common_feature_value.EmbedxWIndex()) {
    for (auto i = common_feature_value.EmbedxWIndex();
         i < common_feature_value.Dim();
         ++i) {
      os << " " << v[i];
    }
  }
  return os.str();
}

int CtrCommonAccessor::ParseFromString(const std::string& str, float* value) {
  _embedx_sgd_rule->InitValue(value + common_feature_value.EmbedxWIndex(),
                              value + common_feature_value.EmbedxG2SumIndex());
  auto ret = paddle::string::str_to_float(str.data(), value);
  PADDLE_ENFORCE_GE(
      ret,
      6UL,
      common::errors::InvalidArgument(
          "Invalid return value. Expect more than 6. But received %d.", ret));
  return ret;
}

}  // namespace paddle::distributed
