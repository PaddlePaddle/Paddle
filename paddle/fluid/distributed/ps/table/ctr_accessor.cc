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
#include <gflags/gflags.h>
#include "glog/logging.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

int CtrCommonAccessor::initialize() {
  auto name = _config.embed_sgd_param().name();
  _embed_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embed_sgd_rule->load_config(_config.embed_sgd_param(), 1);

  name = _config.embedx_sgd_param().name();
  _embedx_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embedx_sgd_rule->load_config(_config.embedx_sgd_param(),
                                _config.embedx_dim());

  common_feature_value.embed_sgd_dim = _embed_sgd_rule->dim();
  common_feature_value.embedx_dim = _config.embedx_dim();
  common_feature_value.embedx_sgd_dim = _embedx_sgd_rule->dim();
  _show_click_decay_rate = _config.ctr_accessor_param().show_click_decay_rate();

  return 0;
}

void CtrCommonAccessor::GetTableInfo(AccessorInfo& info) {
  info.dim = dim();
  info.size = size();
  info.select_dim = select_dim();
  info.select_size = select_size();
  info.update_dim = update_dim();
  info.update_size = update_size();
  info.fea_dim = fea_dim();
}

size_t CtrCommonAccessor::dim() { return common_feature_value.dim(); }

size_t CtrCommonAccessor::dim_size(size_t dim) {
  auto embedx_dim = _config.embedx_dim();
  return common_feature_value.dim_size(dim, embedx_dim);
}

size_t CtrCommonAccessor::size() { return common_feature_value.size(); }

size_t CtrCommonAccessor::mf_size() {
  return (_config.embedx_dim() + common_feature_value.embedx_sgd_dim) *
         sizeof(float);  // embedx embedx_g2sum
}

// pull value
size_t CtrCommonAccessor::select_dim() {
  auto embedx_dim = _config.embedx_dim();
  return 3 + embedx_dim;
}

size_t CtrCommonAccessor::select_dim_size(size_t dim) { return sizeof(float); }

size_t CtrCommonAccessor::select_size() { return select_dim() * sizeof(float); }

// push value
size_t CtrCommonAccessor::update_dim() {
  auto embedx_dim = _config.embedx_dim();
  return 4 + embedx_dim;
}

size_t CtrCommonAccessor::update_dim_size(size_t dim) { return sizeof(float); }

size_t CtrCommonAccessor::update_size() { return update_dim() * sizeof(float); }

bool CtrCommonAccessor::shrink(float* value) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delete_after_unseen_days =
      _config.ctr_accessor_param().delete_after_unseen_days();
  auto delete_threshold = _config.ctr_accessor_param().delete_threshold();

  // time_decay first
  common_feature_value.show(value) *= _show_click_decay_rate;
  common_feature_value.click(value) *= _show_click_decay_rate;

  // shrink after
  auto score = show_click_score(common_feature_value.show(value),
                                common_feature_value.click(value));
  auto unseen_days = common_feature_value.unseen_days(value);
  if (score < delete_threshold || unseen_days > delete_after_unseen_days) {
    return true;
  }
  return false;
}

bool CtrCommonAccessor::save(float* value, int param) {
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
      if (show_click_score(common_feature_value.show(value),
                           common_feature_value.click(value)) >=
              base_threshold &&
          common_feature_value.delta_score(value) >= delta_threshold &&
          common_feature_value.unseen_days(value) <= delta_keep_days) {
        // do this after save, because it must not be modified when retry
        if (param == 2) {
          common_feature_value.delta_score(value) = 0;
        }
        return true;
      } else {
        return false;
      }
    }
    // already decayed in shrink
    case 3: {
      // do this after save, because it must not be modified when retry
      // common_feature_value.unseen_days(value)++;
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

void CtrCommonAccessor::update_stat_after_save(float* value, int param) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (param == 2) {
    delta_threshold = 0;
  }
  switch (param) {
    case 1: {
      if (show_click_score(common_feature_value.show(value),
                           common_feature_value.click(value)) >=
              base_threshold &&
          common_feature_value.delta_score(value) >= delta_threshold &&
          common_feature_value.unseen_days(value) <= delta_keep_days) {
        common_feature_value.delta_score(value) = 0;
      }
    }
      return;
    case 3: {
      common_feature_value.unseen_days(value)++;
    }
      return;
    default:
      return;
  }
}

int32_t CtrCommonAccessor::create(float** values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* value = values[value_item];
    value[common_feature_value.unseen_days_index()] = 0;
    value[common_feature_value.delta_score_index()] = 0;
    value[common_feature_value.show_index()] = 0;
    value[common_feature_value.click_index()] = 0;
    value[common_feature_value.slot_index()] = -1;
    _embed_sgd_rule->init_value(
        value + common_feature_value.embed_w_index(),
        value + common_feature_value.embed_g2sum_index());
    _embedx_sgd_rule->init_value(
        value + common_feature_value.embedx_w_index(),
        value + common_feature_value.embedx_g2sum_index(), false);
  }
  return 0;
}

bool CtrCommonAccessor::need_extend_mf(float* value) {
  float show = value[common_feature_value.show_index()];
  float click = value[common_feature_value.click_index()];
  float score = (show - click) * _config.ctr_accessor_param().nonclk_coeff() +
                click * _config.ctr_accessor_param().click_coeff();
  return score >= _config.embedx_threshold();
}

bool CtrCommonAccessor::has_mf(size_t size) {
  return size > common_feature_value.embedx_g2sum_index();
}

// from CommonFeatureValue to CtrCommonPullValue
int32_t CtrCommonAccessor::select(float** select_values, const float** values,
                                  size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* select_value = select_values[value_item];
    const float* value = values[value_item];
    select_value[CtrCommonPullValue::show_index()] =
        value[common_feature_value.show_index()];
    select_value[CtrCommonPullValue::click_index()] =
        value[common_feature_value.click_index()];
    select_value[CtrCommonPullValue::embed_w_index()] =
        value[common_feature_value.embed_w_index()];
    memcpy(select_value + CtrCommonPullValue::embedx_w_index(),
           value + common_feature_value.embedx_w_index(),
           embedx_dim * sizeof(float));
  }
  return 0;
}

// from CtrCommonPushValue to CtrCommonPushValue
// first dim: item
// second dim: field num
int32_t CtrCommonAccessor::merge(float** update_values,
                                 const float** other_update_values,
                                 size_t num) {
  auto embedx_dim = _config.embedx_dim();
  size_t total_dim = CtrCommonPushValue::dim(embedx_dim);
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* other_update_value = other_update_values[value_item];
    for (auto i = 0u; i < total_dim; ++i) {
      if (i != CtrCommonPushValue::slot_index()) {
        update_value[i] += other_update_value[i];
      }
    }
  }
  return 0;
}

// from CtrCommonPushValue to CommonFeatureValue
// first dim: item
// second dim: field num
int32_t CtrCommonAccessor::update(float** update_values,
                                  const float** push_values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* push_value = push_values[value_item];
    float push_show = push_value[CtrCommonPushValue::show_index()];
    float push_click = push_value[CtrCommonPushValue::click_index()];
    float slot = push_value[CtrCommonPushValue::slot_index()];
    update_value[common_feature_value.show_index()] += push_show;
    update_value[common_feature_value.click_index()] += push_click;
    update_value[common_feature_value.slot_index()] = slot;
    update_value[common_feature_value.delta_score_index()] +=
        (push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
        push_click * _config.ctr_accessor_param().click_coeff();
    update_value[common_feature_value.unseen_days_index()] = 0;
    _embed_sgd_rule->update_value(
        update_value + common_feature_value.embed_w_index(),
        update_value + common_feature_value.embed_g2sum_index(),
        push_value + CtrCommonPushValue::embed_g_index());
    _embedx_sgd_rule->update_value(
        update_value + common_feature_value.embedx_w_index(),
        update_value + common_feature_value.embedx_g2sum_index(),
        push_value + CtrCommonPushValue::embedx_g_index());
  }
  return 0;
}

bool CtrCommonAccessor::create_value(int stage, const float* value) {
  // stage == 0, pull
  // stage == 1, push
  if (stage == 0) {
    return true;
  } else if (stage == 1) {
    // operation
    auto show = CtrCommonPushValue::show(const_cast<float*>(value));
    auto click = CtrCommonPushValue::click(const_cast<float*>(value));
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

float CtrCommonAccessor::show_click_score(float show, float click) {
  auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  auto click_coeff = _config.ctr_accessor_param().click_coeff();
  return (show - click) * nonclk_coeff + click * click_coeff;
}

std::string CtrCommonAccessor::parse_to_string(const float* v, int param) {
  thread_local std::ostringstream os;
  os.clear();
  os.str("");
  os << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4] << " "
     << v[5];
  for (int i = common_feature_value.embed_g2sum_index();
       i < common_feature_value.embedx_w_index(); i++) {
    os << " " << v[i];
  }
  auto show = common_feature_value.show(const_cast<float*>(v));
  auto click = common_feature_value.click(const_cast<float*>(v));
  auto score = show_click_score(show, click);
  if (score >= _config.embedx_threshold() &&
      param > common_feature_value.embedx_w_index()) {
    for (auto i = common_feature_value.embedx_w_index();
         i < common_feature_value.dim(); ++i) {
      os << " " << v[i];
    }
  }
  return os.str();
}

int CtrCommonAccessor::parse_from_string(const std::string& str, float* value) {
  int embedx_dim = _config.embedx_dim();

  _embedx_sgd_rule->init_value(
      value + common_feature_value.embedx_w_index(),
      value + common_feature_value.embedx_g2sum_index());
  auto ret = paddle::string::str_to_float(str.data(), value);
  CHECK(ret >= 6) << "expect more than 6 real:" << ret;
  return ret;
}

}  // namespace distributed
}  // namespace paddle
