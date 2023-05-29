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

#include "paddle/fluid/distributed/ps/table/sparse_accessor.h"

#include <gflags/gflags.h>

#include "glog/logging.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

int SparseAccessor::Initialize() {
  auto name = _config.embed_sgd_param().name();
  _embed_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embed_sgd_rule->LoadConfig(_config.embed_sgd_param(), 1);

  name = _config.embedx_sgd_param().name();
  _embedx_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embedx_sgd_rule->LoadConfig(_config.embedx_sgd_param(),
                               _config.embedx_dim());

  sparse_feature_value.embed_sgd_dim = _embed_sgd_rule->Dim();
  sparse_feature_value.embedx_dim = _config.embedx_dim();
  sparse_feature_value.embedx_sgd_dim = _embedx_sgd_rule->Dim();
  _show_click_decay_rate = _config.ctr_accessor_param().show_click_decay_rate();

  InitAccessorInfo();
  return 0;
}

void SparseAccessor::InitAccessorInfo() {
  _accessor_info.dim = sparse_feature_value.Dim();
  _accessor_info.size = sparse_feature_value.Size();
  auto embedx_dim = _config.embedx_dim();
  _accessor_info.select_dim = 1 + embedx_dim;
  _accessor_info.select_size = _accessor_info.select_dim * sizeof(float);
  _accessor_info.update_dim = 4 + embedx_dim;
  _accessor_info.update_size = _accessor_info.update_dim * sizeof(float);
  _accessor_info.mf_size =
      (embedx_dim + sparse_feature_value.embedx_sgd_dim) * sizeof(float);
}

bool SparseAccessor::Shrink(float* value) {
  auto delete_after_unseen_days =
      _config.ctr_accessor_param().delete_after_unseen_days();
  auto delete_threshold = _config.ctr_accessor_param().delete_threshold();

  // time_decay first
  sparse_feature_value.Show(value) *= _show_click_decay_rate;
  sparse_feature_value.Click(value) *= _show_click_decay_rate;

  // shrink after
  auto score = ShowClickScore(sparse_feature_value.Show(value),
                              sparse_feature_value.Click(value));
  auto unseen_days = sparse_feature_value.UnseenDays(value);
  if (score < delete_threshold || unseen_days > delete_after_unseen_days) {
    return true;
  }
  return false;
}

bool SparseAccessor::Save(float* value, int param) {
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
      if (ShowClickScore(sparse_feature_value.Show(value),
                         sparse_feature_value.Click(value)) >= base_threshold &&
          sparse_feature_value.DeltaScore(value) >= delta_threshold &&
          sparse_feature_value.UnseenDays(value) <= delta_keep_days) {
        // do this after save, because it must not be modified when retry
        if (param == 2) {
          sparse_feature_value.DeltaScore(value) = 0;
        }
        return true;
      } else {
        return false;
      }
    }
    // already decayed in shrink
    case 3: {
      // do this after save, because it must not be modified when retry
      // sparse_feature_value.UnseenDays(value)++;
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

void SparseAccessor::UpdateStatAfterSave(float* value, int param) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (param == 2) {
    delta_threshold = 0;
  }
  switch (param) {
    case 1: {
      if (ShowClickScore(sparse_feature_value.Show(value),
                         sparse_feature_value.Click(value)) >= base_threshold &&
          sparse_feature_value.DeltaScore(value) >= delta_threshold &&
          sparse_feature_value.UnseenDays(value) <= delta_keep_days) {
        sparse_feature_value.DeltaScore(value) = 0;
      }
    }
      return;
    case 3: {
      sparse_feature_value.UnseenDays(value)++;
    }
      return;
    default:
      return;
  }
}

int32_t SparseAccessor::Create(float** values, size_t num) {
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* value = values[value_item];
    value[sparse_feature_value.UnseenDaysIndex()] = 0;
    value[sparse_feature_value.DeltaScoreIndex()] = 0;
    value[sparse_feature_value.ShowIndex()] = 0;
    value[sparse_feature_value.ClickIndex()] = 0;
    value[sparse_feature_value.SlotIndex()] = -1;
    bool zero_init = _config.ctr_accessor_param().zero_init();
    _embed_sgd_rule->InitValue(value + sparse_feature_value.EmbedWIndex(),
                               value + sparse_feature_value.EmbedG2SumIndex(),
                               zero_init);
    _embedx_sgd_rule->InitValue(value + sparse_feature_value.EmbedxWIndex(),
                                value + sparse_feature_value.EmbedxG2SumIndex(),
                                false);
  }
  return 0;
}

bool SparseAccessor::NeedExtendMF(float* value) {
  float show = value[sparse_feature_value.ShowIndex()];
  float click = value[sparse_feature_value.ClickIndex()];
  float score = (show - click) * _config.ctr_accessor_param().nonclk_coeff() +
                click * _config.ctr_accessor_param().click_coeff();
  return score >= _config.embedx_threshold();
}

bool SparseAccessor::HasMF(int size) {
  return size > sparse_feature_value.EmbedxG2SumIndex();
}

// from SparseFeatureValue to SparsePullValue
int32_t SparseAccessor::Select(float** select_values,
                               const float** values,
                               size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* select_value = select_values[value_item];
    const float* value = values[value_item];
    select_value[SparsePullValue::EmbedWIndex()] =
        value[sparse_feature_value.EmbedWIndex()];
    memcpy(select_value + SparsePullValue::EmbedxWIndex(),
           value + sparse_feature_value.EmbedxWIndex(),
           embedx_dim * sizeof(float));
  }
  return 0;
}

// from SparsePushValue to SparsePushValue
// first dim: item
// second dim: field num
int32_t SparseAccessor::Merge(float** update_values,
                              const float** other_update_values,
                              size_t num) {
  auto embedx_dim = _config.embedx_dim();
  size_t total_dim = SparsePushValue::Dim(embedx_dim);
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* other_update_value = other_update_values[value_item];
    for (size_t i = 0; i < total_dim; ++i) {
      if (static_cast<int>(i) != SparsePushValue::SlotIndex()) {
        update_value[i] += other_update_value[i];
      }
    }
  }
  return 0;
}

// from SparsePushValue to SparseFeatureValue
// first dim: item
// second dim: field num
int32_t SparseAccessor::Update(float** update_values,
                               const float** push_values,
                               size_t num) {
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* push_value = push_values[value_item];
    float push_show = push_value[SparsePushValue::ShowIndex()];
    float push_click = push_value[SparsePushValue::ClickIndex()];
    float slot = push_value[SparsePushValue::SlotIndex()];
    update_value[sparse_feature_value.ShowIndex()] += push_show;
    update_value[sparse_feature_value.ClickIndex()] += push_click;
    update_value[sparse_feature_value.SlotIndex()] = slot;
    update_value[sparse_feature_value.DeltaScoreIndex()] +=
        (push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
        push_click * _config.ctr_accessor_param().click_coeff();
    update_value[sparse_feature_value.UnseenDaysIndex()] = 0;
    _embed_sgd_rule->UpdateValue(
        update_value + sparse_feature_value.EmbedWIndex(),
        update_value + sparse_feature_value.EmbedG2SumIndex(),
        push_value + SparsePushValue::EmbedGIndex(),
        push_show);
    _embedx_sgd_rule->UpdateValue(
        update_value + sparse_feature_value.EmbedxWIndex(),
        update_value + sparse_feature_value.EmbedxG2SumIndex(),
        push_value + SparsePushValue::EmbedxGIndex(),
        push_show);
  }
  return 0;
}

bool SparseAccessor::CreateValue(int stage, const float* value) {
  // stage == 0, pull
  // stage == 1, push
  if (stage == 0) {
    return true;
  } else if (stage == 1) {
    // operation
    auto show = SparsePushValue::Show(const_cast<float*>(value));
    auto click = SparsePushValue::Click(const_cast<float*>(value));
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

float SparseAccessor::ShowClickScore(float show, float click) {
  auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  auto click_coeff = _config.ctr_accessor_param().click_coeff();
  return (show - click) * nonclk_coeff + click * click_coeff;
}

std::string SparseAccessor::ParseToString(const float* v, int param) {
  thread_local std::ostringstream os;
  os.clear();
  os.str("");
  os << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4] << " "
     << v[5];
  for (int i = sparse_feature_value.EmbedG2SumIndex();
       i < sparse_feature_value.EmbedxWIndex();
       i++) {
    os << " " << v[i];
  }
  auto show = sparse_feature_value.Show(const_cast<float*>(v));
  auto click = sparse_feature_value.Click(const_cast<float*>(v));
  auto score = ShowClickScore(show, click);
  if (score >= _config.embedx_threshold() &&
      param > sparse_feature_value.EmbedxWIndex()) {
    for (auto i = sparse_feature_value.EmbedxWIndex();
         i < sparse_feature_value.Dim();
         ++i) {
      os << " " << v[i];
    }
  }
  return os.str();
}

int SparseAccessor::ParseFromString(const std::string& str, float* value) {
  _embedx_sgd_rule->InitValue(value + sparse_feature_value.EmbedxWIndex(),
                              value + sparse_feature_value.EmbedxG2SumIndex());
  auto ret = paddle::string::str_to_float(str.data(), value);
  CHECK(ret >= 6) << "expect more than 6 real:" << ret;
  return ret;
}

}  // namespace distributed
}  // namespace paddle
