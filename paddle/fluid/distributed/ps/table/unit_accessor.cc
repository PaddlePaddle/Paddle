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

#include "paddle/fluid/distributed/ps/table/unit_accessor.h"
#include <gflags/gflags.h>
#include "glog/logging.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

int DownpourUnitAccessor::initialize() {
  auto name = _config.embed_sgd_param().name();
  _embed_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embed_sgd_rule->load_config(_config.embed_sgd_param(), 1);

  name = _config.embedx_sgd_param().name();
  _embedx_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embedx_sgd_rule->load_config(_config.embedx_sgd_param(),
                                _config.embedx_dim());

  unit_feature_value.embed_sgd_dim = _embed_sgd_rule->dim();
  unit_feature_value.embedx_dim = _config.embedx_dim();
  unit_feature_value.embedx_sgd_dim = _embedx_sgd_rule->dim();
  _show_click_decay_rate = _config.ctr_accessor_param().show_click_decay_rate();

  return 0;
}

size_t DownpourUnitAccessor::dim() { return unit_feature_value.dim(); }

size_t DownpourUnitAccessor::dim_size(size_t dim) {
  auto embedx_dim = _config.embedx_dim();
  return unit_feature_value.dim_size(dim, embedx_dim);
}

size_t DownpourUnitAccessor::size() { return unit_feature_value.size(); }

size_t DownpourUnitAccessor::mf_size() {
  return (_config.embedx_dim() + unit_feature_value.embedx_sgd_dim) *
         sizeof(float);  // embedx embedx_g2sum
}

// pull value
size_t DownpourUnitAccessor::select_dim() {
  auto embedx_dim = _config.embedx_dim();
  return 3 + embedx_dim;
}

size_t DownpourUnitAccessor::select_dim_size(size_t dim) {
  return sizeof(float);
}

size_t DownpourUnitAccessor::select_size() {
  return select_dim() * sizeof(float);
}

// push value
size_t DownpourUnitAccessor::update_dim() {
  auto embedx_dim = _config.embedx_dim();
  return 4 + embedx_dim;
}

size_t DownpourUnitAccessor::update_dim_size(size_t dim) {
  return sizeof(float);
}

size_t DownpourUnitAccessor::update_size() {
  return update_dim() * sizeof(float);
}

bool DownpourUnitAccessor::shrink(float* value) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delete_after_unseen_days =
      _config.ctr_accessor_param().delete_after_unseen_days();
  auto delete_threshold = _config.ctr_accessor_param().delete_threshold();

  // time_decay first
  unit_feature_value.show(value) *= _show_click_decay_rate;
  unit_feature_value.click(value) *= _show_click_decay_rate;

  // shrink after
  auto score = show_click_score(unit_feature_value.show(value),
                                unit_feature_value.click(value));
  auto unseen_days = unit_feature_value.unseen_days(value);
  if (score < delete_threshold || unseen_days > delete_after_unseen_days) {
    return true;
  }
  return false;
}

bool DownpourUnitAccessor::save_ssd(float* value) {
  if (unit_feature_value.unseen_days(value) > _ssd_unseenday_threshold) {
    return true;
  }
  return false;
}

// bool DownpourUnitAccessor::save_cache(
//         float* value, int param, double global_cache_threshold) {
//     auto base_threshold = _config.ctr_accessor_param().base_threshold();
//     auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
//     if (show_click_score(unit_feature_value.show(value),
//     unit_feature_value.click(value)) >= base_threshold
//         && unit_feature_value.unseen_days(value) <= delta_keep_days) {
//         return unit_feature_value.show(value) > global_cache_threshold;
//     }
//     return false;
// }

bool DownpourUnitAccessor::save(float* value, int param) {
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
      if (show_click_score(unit_feature_value.show(value),
                           unit_feature_value.click(value)) >= base_threshold &&
          unit_feature_value.delta_score(value) >= delta_threshold &&
          unit_feature_value.unseen_days(value) <= delta_keep_days) {
        // do this after save, because it must not be modified when retry
        if (param == 2) {
          unit_feature_value.delta_score(value) = 0;
        }
        return true;
      } else {
        return false;
      }
    }
    // already decayed in shrink
    case 3: {
      // do this after save, because it must not be modified when retry
      // unit_feature_value.unseen_days(value)++;
      return true;
    }
    default:
      return true;
  }
}

void DownpourUnitAccessor::update_stat_after_save(float* value, int param) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (param == 2) {
    delta_threshold = 0;
  }
  switch (param) {
    case 1: {
      if (show_click_score(unit_feature_value.show(value),
                           unit_feature_value.click(value)) >= base_threshold &&
          unit_feature_value.delta_score(value) >= delta_threshold &&
          unit_feature_value.unseen_days(value) <= delta_keep_days) {
        unit_feature_value.delta_score(value) = 0;
      }
    }
      return;
    case 3: {
      unit_feature_value.unseen_days(value)++;
    }
      return;
    default:
      return;
  };
}

int32_t DownpourUnitAccessor::create(float** values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* value = values[value_item];
    value[unit_feature_value.unseen_days_index()] = 0;
    value[unit_feature_value.delta_score_index()] = 0;
    value[unit_feature_value.show_index()] = 0;
    value[unit_feature_value.click_index()] = 0;
    value[unit_feature_value.slot_index()] = -1;
    _embed_sgd_rule->init_value(value + unit_feature_value.embed_w_index(),
                                value + unit_feature_value.embed_g2sum_index());
    _embedx_sgd_rule->init_value(
        value + unit_feature_value.embedx_w_index(),
        value + unit_feature_value.embedx_g2sum_index());
  }
  return 0;
}

bool DownpourUnitAccessor::need_extend_mf(float* value) {
  float show = value[unit_feature_value.show_index()];
  float click = value[unit_feature_value.click_index()];
  // float score = (show - click) * _config.ctr_accessor_param().nonclk_coeff()
  float score = (show - click) * _config.ctr_accessor_param().nonclk_coeff() +
                click * _config.ctr_accessor_param().click_coeff();
  //+ click * _config.ctr_accessor_param().click_coeff();
  return score >= _config.embedx_threshold();
}

bool DownpourUnitAccessor::has_mf(size_t size) {
  return size > unit_feature_value.embedx_g2sum_index();
}

// from UnitFeatureValue to DownpourUnitPullValue
int32_t DownpourUnitAccessor::select(float** select_values,
                                     const float** values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* select_value = select_values[value_item];
    float* value = const_cast<float*>(values[value_item]);
    select_value[DownpourUnitPullValue::show_index()] =
        value[unit_feature_value.show_index()];
    select_value[DownpourUnitPullValue::click_index()] =
        value[unit_feature_value.click_index()];
    select_value[DownpourUnitPullValue::embed_w_index()] =
        value[unit_feature_value.embed_w_index()];
    memcpy(select_value + DownpourUnitPullValue::embedx_w_index(),
           value + unit_feature_value.embedx_w_index(),
           embedx_dim * sizeof(float));
  }
  return 0;
}

// from DownpourUnitPushValue to DownpourUnitPushValue
// first dim: item
// second dim: field num
int32_t DownpourUnitAccessor::merge(float** update_values,
                                    const float** other_update_values,
                                    size_t num) {
  auto embedx_dim = _config.embedx_dim();
  size_t total_dim = DownpourUnitPushValue::dim(embedx_dim);
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* other_update_value = other_update_values[value_item];
    for (auto i = 0u; i < total_dim; ++i) {
      if (i != DownpourUnitPushValue::slot_index()) {
        update_value[i] += other_update_value[i];
      }
    }
  }
  return 0;
}

// from DownpourUnitPushValue to UnitFeatureValue
// first dim: item
// second dim: field num
int32_t DownpourUnitAccessor::update(float** update_values,
                                     const float** push_values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* push_value = push_values[value_item];
    float push_show = push_value[DownpourUnitPushValue::show_index()];
    float push_click = push_value[DownpourUnitPushValue::click_index()];
    float slot = push_value[DownpourUnitPushValue::slot_index()];
    update_value[unit_feature_value.show_index()] += push_show;
    update_value[unit_feature_value.click_index()] += push_click;
    update_value[unit_feature_value.slot_index()] = slot;
    update_value[unit_feature_value.delta_score_index()] +=
        (push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
        push_click * _config.ctr_accessor_param().click_coeff();
    //(push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
    // push_click * _config.ctr_accessor_param().click_coeff();
    update_value[unit_feature_value.unseen_days_index()] = 0;
    _embed_sgd_rule->update_value(
        update_value + unit_feature_value.embed_w_index(),
        update_value + unit_feature_value.embed_g2sum_index(),
        push_value + DownpourUnitPushValue::embed_g_index());
    _embedx_sgd_rule->update_value(
        update_value + unit_feature_value.embedx_w_index(),
        update_value + unit_feature_value.embedx_g2sum_index(),
        push_value + DownpourUnitPushValue::embedx_g_index());
  }
  return 0;
}

bool DownpourUnitAccessor::create_value(int stage, const float* value) {
  // stage == 0, pull
  // stage == 1, push
  if (stage == 0) {
    return true;
  } else if (stage == 1) {
    auto show = DownpourUnitPushValue::show(const_cast<float*>(value));
    auto click = DownpourUnitPushValue::click(const_cast<float*>(value));
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

float DownpourUnitAccessor::show_click_score(float show, float click) {
  // auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  // auto click_coeff = _config.ctr_accessor_param().click_coeff();
  auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  auto click_coeff = _config.ctr_accessor_param().click_coeff();
  return (show - click) * nonclk_coeff + click * click_coeff;
}

std::string DownpourUnitAccessor::parse_to_string(const float* v, int param) {
  thread_local std::ostringstream os;
  os.clear();
  os.str("");
  os << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4] << " "
     << v[5];
  for (int i = unit_feature_value.embed_g2sum_index();
       i < unit_feature_value.embedx_w_index(); i++) {
    os << " " << v[i];
  }
  auto show = unit_feature_value.show(const_cast<float*>(v));
  auto click = unit_feature_value.click(const_cast<float*>(v));
  auto score = show_click_score(show, click);
  if (score >= _config.embedx_threshold()) {
    for (auto i = unit_feature_value.embedx_w_index();
         i < unit_feature_value.dim(); ++i) {
      os << " " << v[i];
    }
  }
  return os.str();
}

int DownpourUnitAccessor::parse_from_string(const std::string& str,
                                            float* value) {
  int embedx_dim = _config.embedx_dim();

  _embedx_sgd_rule->init_value(value + unit_feature_value.embedx_w_index(),
                               value + unit_feature_value.embedx_g2sum_index());
  auto ret = paddle::string::str_to_float(str.data(), value);
  CHECK(ret >= 6) << "expect more than 6 real:" << ret;
  return ret;
}

}  // namespace distributed
}  // namespace paddle
