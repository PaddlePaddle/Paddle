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

#include <gflags/gflags.h>

#include "glog/logging.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

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
  VLOG(0) << " INTO CtrDymfAccessor::Initialize(); embed_sgd_dim:"
          << common_feature_value.embed_sgd_dim
          << " embedx_dim:" << common_feature_value.embedx_dim
          << "  embedx_sgd_dim:" << common_feature_value.embedx_sgd_dim;
  InitAccessorInfo();
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
  if (common_feature_value.UnseenDays(value) > _ssd_unseenday_threshold) {
    return true;
  }
  return false;
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
    case 3: {
      common_feature_value.UnseenDays(value)++;
    }
      return;
    default:
      return;
  }
}

int32_t CtrDymfAccessor::Create(float** values, size_t num) {
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* value = values[value_item];
    value[common_feature_value.UnseenDaysIndex()] = 0;
    value[common_feature_value.DeltaScoreIndex()] = 0;
    value[common_feature_value.ShowIndex()] = 0;
    value[common_feature_value.ClickIndex()] = 0;
    value[common_feature_value.SlotIndex()] = -1;
    value[common_feature_value.MfDimIndex()] = -1;
    _embed_sgd_rule->InitValue(
        value + common_feature_value.EmbedWIndex(),
        value + common_feature_value.EmbedG2SumIndex(),
        false);  // adam embed init not zero, adagrad embed init zero
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
  os << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4];
  //    << v[5] << " " << v[6];
  for (int i = common_feature_value.EmbedG2SumIndex();
       i < common_feature_value.EmbedxG2SumIndex();
       i++) {
    os << " " << v[i];
  }
  auto show = common_feature_value.Show(const_cast<float*>(v));
  auto click = common_feature_value.Click(const_cast<float*>(v));
  auto score = ShowClickScore(show, click);
  auto mf_dim = int(common_feature_value.MfDim(const_cast<float*>(v)));
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
  CHECK(ret >= 7) << "expect more than 7 real:" << ret;
  return ret;
}

}  // namespace distributed
}  // namespace paddle
