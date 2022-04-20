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

#include "paddle/fluid/distributed/ps/table/ctr_double_accessor.h"
#include <gflags/gflags.h>
#include "glog/logging.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

int CtrDoubleAccessor::Initialize() {
  auto name = _config.embed_sgd_param().name();
  _embed_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embed_sgd_rule->LoadConfig(_config.embed_sgd_param(), 1);

  name = _config.embedx_sgd_param().name();
  _embedx_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embedx_sgd_rule->LoadConfig(_config.embedx_sgd_param(),
                               _config.embedx_dim());

  _show_click_decay_rate = _config.ctr_accessor_param().show_click_decay_rate();
  _ssd_unseenday_threshold =
      _config.ctr_accessor_param().ssd_unseenday_threshold();

  if (_config.ctr_accessor_param().show_scale()) {
    _show_scale = true;
  }

  InitAccessorInfo();
  return 0;
}

void CtrDoubleAccessor::InitAccessorInfo() {
  auto embedx_dim = _config.embedx_dim();
  _accessor_info.dim = CtrDoubleFeatureValue::Dim(embedx_dim);
  _accessor_info.size = CtrDoubleFeatureValue::Size(embedx_dim);
  _accessor_info.select_dim = 3 + embedx_dim;
  _accessor_info.select_size = _accessor_info.select_dim * sizeof(float);
  _accessor_info.update_dim = 4 + embedx_dim;
  _accessor_info.update_size = _accessor_info.update_dim * sizeof(float);
  _accessor_info.mf_size = (embedx_dim + 1) * sizeof(float);
}

bool CtrDoubleAccessor::Shrink(float* value) {
  // auto base_threshold = _config.ctr_accessor_param().base_threshold();
  // auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  // auto delete_threshold = _config.ctr_accessor_param().delete_threshold();
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delete_after_unseen_days =
      _config.ctr_accessor_param().delete_after_unseen_days();
  auto delete_threshold = _config.ctr_accessor_param().delete_threshold();
  // time_decay first
  CtrDoubleFeatureValue::Show(value) *= _show_click_decay_rate;
  CtrDoubleFeatureValue::Click(value) *= _show_click_decay_rate;
  // shrink after
  auto score = ShowClickScore(CtrDoubleFeatureValue::Show(value),
                              CtrDoubleFeatureValue::Click(value));
  auto unseen_days = CtrDoubleFeatureValue::UnseenDays(value);
  if (score < delete_threshold || unseen_days > delete_after_unseen_days) {
    return true;
  }
  return false;
}

bool CtrDoubleAccessor::SaveSSD(float* value) {
  if (CtrDoubleFeatureValue::UnseenDays(value) > _ssd_unseenday_threshold) {
    return true;
  }
  return false;
}

bool CtrDoubleAccessor::SaveCache(float* value, int param,
                                  double global_cache_threshold) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (ShowClickScore(CtrDoubleFeatureValue::Show(value),
                     CtrDoubleFeatureValue::Click(value)) >= base_threshold &&
      CtrDoubleFeatureValue::UnseenDays(value) <= delta_keep_days) {
    return CtrDoubleFeatureValue::Show(value) > global_cache_threshold;
  }
  return false;
}

bool CtrDoubleAccessor::Save(float* value, int param) {
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
      if (ShowClickScore(CtrDoubleFeatureValue::Show(value),
                         CtrDoubleFeatureValue::Click(value)) >=
              base_threshold &&
          CtrDoubleFeatureValue::DeltaScore(value) >= delta_threshold &&
          CtrDoubleFeatureValue::UnseenDays(value) <= delta_keep_days) {
        // do this after save, because it must not be modified when retry
        if (param == 2) {
          CtrDoubleFeatureValue::DeltaScore(value) = 0;
        }
        return true;
      } else {
        return false;
      }
    }
    // already decayed in shrink
    case 3: {
      // CtrDoubleFeatureValue::Show(value) *= _show_click_decay_rate;
      // CtrDoubleFeatureValue::Click(value) *= _show_click_decay_rate;
      // do this after save, because it must not be modified when retry
      // CtrDoubleFeatureValue::UnseenDays(value)++;
      return true;
    }
    default:
      return true;
  };
}

void CtrDoubleAccessor::UpdateStatAfterSave(float* value, int param) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (param == 2) {
    delta_threshold = 0;
  }
  switch (param) {
    case 1: {
      if (ShowClickScore(CtrDoubleFeatureValue::Show(value),
                         CtrDoubleFeatureValue::Click(value)) >=
              base_threshold &&
          CtrDoubleFeatureValue::DeltaScore(value) >= delta_threshold &&
          CtrDoubleFeatureValue::UnseenDays(value) <= delta_keep_days) {
        CtrDoubleFeatureValue::DeltaScore(value) = 0;
      }
    }
      return;
    case 3: {
      CtrDoubleFeatureValue::UnseenDays(value)++;
    }
      return;
    default:
      return;
  };
}

int32_t CtrDoubleAccessor::Create(float** values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* value = values[value_item];
    value[CtrDoubleFeatureValue::UnseenDaysIndex()] = 0;
    value[CtrDoubleFeatureValue::DeltaScoreIndex()] = 0;
    *(double*)(value + CtrDoubleFeatureValue::ShowIndex()) = 0;
    *(double*)(value + CtrDoubleFeatureValue::ClickIndex()) = 0;
    value[CtrDoubleFeatureValue::SlotIndex()] = -1;
    _embed_sgd_rule->InitValue(
        value + CtrDoubleFeatureValue::EmbedWIndex(),
        value + CtrDoubleFeatureValue::EmbedG2SumIndex());
    _embedx_sgd_rule->InitValue(
        value + CtrDoubleFeatureValue::EmbedxWIndex(),
        value + CtrDoubleFeatureValue::EmbedxG2SumIndex(), false);
  }
  return 0;
}
bool CtrDoubleAccessor::NeedExtendMF(float* value) {
  auto show = ((double*)(value + CtrDoubleFeatureValue::ShowIndex()))[0];
  auto click = ((double*)(value + CtrDoubleFeatureValue::ClickIndex()))[0];
  // float score = (show - click) * _config.ctr_accessor_param().nonclk_coeff()
  auto score = (show - click) * _config.ctr_accessor_param().nonclk_coeff() +
               click * _config.ctr_accessor_param().click_coeff();
  //+ click * _config.ctr_accessor_param().click_coeff();
  return score >= _config.embedx_threshold();
}
// from CtrDoubleFeatureValue to CtrDoublePullValue
int32_t CtrDoubleAccessor::Select(float** select_values, const float** values,
                                  size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* select_value = select_values[value_item];
    float* value = const_cast<float*>(values[value_item]);
    select_value[CtrDoublePullValue::ShowIndex()] =
        (float)*(double*)(value + CtrDoubleFeatureValue::ShowIndex());
    select_value[CtrDoublePullValue::ClickIndex()] =
        (float)*(double*)(value + CtrDoubleFeatureValue::ClickIndex());
    select_value[CtrDoublePullValue::EmbedWIndex()] =
        value[CtrDoubleFeatureValue::EmbedWIndex()];
    memcpy(select_value + CtrDoublePullValue::EmbedxWIndex(),
           value + CtrDoubleFeatureValue::EmbedxWIndex(),
           embedx_dim * sizeof(float));
  }
  return 0;
}
// from CtrDoublePushValue to CtrDoublePushValue
// first dim: item
// second dim: field num
int32_t CtrDoubleAccessor::Merge(float** update_values,
                                 const float** other_update_values,
                                 size_t num) {
  auto embedx_dim = _config.embedx_dim();
  size_t total_dim = CtrDoublePushValue::Dim(embedx_dim);
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* other_update_value = other_update_values[value_item];
    /**(double*)(update_value + CtrDoublePushValue::ShowIndex()) +=
    *(double*)(other_update_value + CtrDoublePushValue::ShowIndex());
    *(double*)(update_value + CtrDoublePushValue::ClickIndex()) +=
    *(double*)(other_update_value + CtrDoublePushValue::ClickIndex());
    for (auto i = 3u; i < total_dim; ++i) {
        update_value[i] += other_update_value[i];
    }*/
    for (auto i = 0u; i < total_dim; ++i) {
      if (i != CtrDoublePushValue::SlotIndex()) {
        update_value[i] += other_update_value[i];
      }
    }
  }
  return 0;
}
// from CtrDoublePushValue to CtrDoubleFeatureValue
// first dim: item
// second dim: field num
int32_t CtrDoubleAccessor::Update(float** update_values,
                                  const float** push_values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* push_value = push_values[value_item];
    float push_show = push_value[CtrDoublePushValue::ShowIndex()];
    float push_click = push_value[CtrDoublePushValue::ClickIndex()];
    float slot = push_value[CtrDoublePushValue::SlotIndex()];
    *(double*)(update_value + CtrDoubleFeatureValue::ShowIndex()) +=
        (double)push_show;
    *(double*)(update_value + CtrDoubleFeatureValue::ClickIndex()) +=
        (double)push_click;
    update_value[CtrDoubleFeatureValue::SlotIndex()] = slot;
    update_value[CtrDoubleFeatureValue::DeltaScoreIndex()] +=
        (push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
        push_click * _config.ctr_accessor_param().click_coeff();
    //(push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
    // push_click * _config.ctr_accessor_param().click_coeff();
    update_value[CtrDoubleFeatureValue::UnseenDaysIndex()] = 0;
    if (!_show_scale) {
      push_show = 1;
    }
    VLOG(3) << "accessor show scale:" << _show_scale
            << ", push_show:" << push_show;
    _embed_sgd_rule->UpdateValue(
        update_value + CtrDoubleFeatureValue::EmbedWIndex(),
        update_value + CtrDoubleFeatureValue::EmbedG2SumIndex(),
        push_value + CtrDoublePushValue::EmbedGIndex(), push_show);
    _embedx_sgd_rule->UpdateValue(
        update_value + CtrDoubleFeatureValue::EmbedxWIndex(),
        update_value + CtrDoubleFeatureValue::EmbedxG2SumIndex(),
        push_value + CtrDoublePushValue::EmbedxGIndex(), push_show);
  }
  return 0;
}
bool CtrDoubleAccessor::CreateValue(int stage, const float* value) {
  // stage == 0, pull
  // stage == 1, push
  if (stage == 0) {
    return true;
  } else if (stage == 1) {
    auto show = CtrDoublePushValue::Show(const_cast<float*>(value));
    auto click = CtrDoublePushValue::Click(const_cast<float*>(value));
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
double CtrDoubleAccessor::ShowClickScore(double show, double click) {
  // auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  // auto click_coeff = _config.ctr_accessor_param().click_coeff();
  auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  auto click_coeff = _config.ctr_accessor_param().click_coeff();
  return (show - click) * nonclk_coeff + click * click_coeff;
}
std::string CtrDoubleAccessor::ParseToString(const float* v, int param_size) {
  thread_local std::ostringstream os;
  os.clear();
  os.str("");
  os << v[0] << " " << v[1] << " " << (float)((double*)(v + 2))[0] << " "
     << (float)((double*)(v + 4))[0] << " " << v[6] << " " << v[7] << " "
     << v[8];
  auto show = CtrDoubleFeatureValue::Show(const_cast<float*>(v));
  auto click = CtrDoubleFeatureValue::Click(const_cast<float*>(v));
  auto score = ShowClickScore(show, click);
  if (score >= _config.embedx_threshold() && param_size > 9) {
    os << " " << v[9];
    for (auto i = 0; i < _config.embedx_dim(); ++i) {
      os << " " << v[10 + i];
    }
  }
  return os.str();
}
int CtrDoubleAccessor::ParseFromString(const std::string& str, float* value) {
  int embedx_dim = _config.embedx_dim();
  float data_buff[_accessor_info.dim + 2];
  float* data_buff_ptr = data_buff;
  _embedx_sgd_rule->InitValue(
      data_buff_ptr + CtrDoubleFeatureValue::EmbedxWIndex(),
      data_buff_ptr + CtrDoubleFeatureValue::EmbedxG2SumIndex());
  auto str_len = paddle::string::str_to_float(str.data(), data_buff_ptr);
  CHECK(str_len >= 6) << "expect more than 6 real:" << str_len;
  int show_index = CtrDoubleFeatureValue::ShowIndex();
  int click_index = CtrDoubleFeatureValue::ClickIndex();
  int embed_w_index = CtrDoubleFeatureValue::EmbedWIndex();
  // no slot, embedx
  int value_dim = _accessor_info.dim;
  int embedx_g2sum_index = CtrDoubleFeatureValue::EmbedxG2SumIndex();
  value[CtrDoubleFeatureValue::SlotIndex()] = -1;
  // other case
  if (str_len == (value_dim - 1)) {
    // copy unseen_days..delta_score
    memcpy(value, data_buff_ptr, show_index * sizeof(float));
    // copy show & click
    *(double*)(value + show_index) = (double)data_buff_ptr[2];
    *(double*)(value + click_index) = (double)data_buff_ptr[3];
    // copy others
    value[CtrDoubleFeatureValue::EmbedWIndex()] = data_buff_ptr[4];
    value[CtrDoubleFeatureValue::EmbedG2SumIndex()] = data_buff_ptr[5];
    memcpy(value + embedx_g2sum_index, data_buff_ptr + 6,
           (embedx_dim + 1) * sizeof(float));
  } else {
    // copy unseen_days..delta_score
    memcpy(value, data_buff_ptr, show_index * sizeof(float));
    // copy show & click
    *(double*)(value + show_index) = (double)data_buff_ptr[2];
    *(double*)(value + click_index) = (double)data_buff_ptr[3];
    // copy embed_w..embedx_w
    memcpy(value + embed_w_index, data_buff_ptr + 4,
           (str_len - 4) * sizeof(float));
  }
  if (str_len == (value_dim - 1) || str_len == 6) {
    str_len += 1;
  }
  return str_len + 2;
}

}  // namespace distributed
}  // namespace paddle
