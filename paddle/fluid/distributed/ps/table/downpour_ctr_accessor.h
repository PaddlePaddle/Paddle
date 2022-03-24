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

#pragma once
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/sparse_sgd_rule.h"

namespace paddle {
namespace distributed {

/**
 * @brief Accessor for unit
 **/
class DownpourCtrAccessor : public ValueAccessor {
 public:
  struct DownpourCtrFeatureValue {
    /*
    float unseen_days;
    float delta_score;
    float show;
    float click;
    float embed_w;
    float embed_g2sum;
    float slot;
    float embedx_g2sum;
    std::vector<float> embedx_w;
    */

    static int dim(int embedx_dim) { return 8 + embedx_dim; }
    static int dim_size(size_t dim, int embedx_dim) { return sizeof(float); }
    static int size(int embedx_dim) { return dim(embedx_dim) * sizeof(float); }
    static int unseen_days_index() { return 0; }
    static int delta_score_index() {
      return DownpourCtrFeatureValue::unseen_days_index() + 1;
    }
    static int show_index() {
      return DownpourCtrFeatureValue::delta_score_index() + 1;
    }
    static int click_index() {
      return DownpourCtrFeatureValue::show_index() + 1;
    }
    static int embed_w_index() {
      return DownpourCtrFeatureValue::click_index() + 1;
    }
    static int embed_g2sum_index() {
      return DownpourCtrFeatureValue::embed_w_index() + 1;
    }
    static int slot_index() {
      return DownpourCtrFeatureValue::embed_g2sum_index() + 1;
    }
    static int embedx_g2sum_index() {
      return DownpourCtrFeatureValue::slot_index() + 1;
    }
    static int embedx_w_index() {
      return DownpourCtrFeatureValue::embedx_g2sum_index() + 1;
    }
    static float& unseen_days(float* val) {
      return val[DownpourCtrFeatureValue::unseen_days_index()];
    }
    static float& delta_score(float* val) {
      return val[DownpourCtrFeatureValue::delta_score_index()];
    }
    static float& show(float* val) {
      return val[DownpourCtrFeatureValue::show_index()];
    }
    static float& click(float* val) {
      return val[DownpourCtrFeatureValue::click_index()];
    }
    static float& slot(float* val) {
      return val[DownpourCtrFeatureValue::slot_index()];
    }
    static float& embed_w(float* val) {
      return val[DownpourCtrFeatureValue::embed_w_index()];
    }
    static float& embed_g2sum(float* val) {
      return val[DownpourCtrFeatureValue::embed_g2sum_index()];
    }
    static float& embedx_g2sum(float* val) {
      return val[DownpourCtrFeatureValue::embedx_g2sum_index()];
    }
    static float* embedx_w(float* val) {
      return (val + DownpourCtrFeatureValue::embedx_w_index());
    }
  };

  struct DownpourCtrPushValue {
    /*
    float slot;
    float show;
    float click;
    float embed_g;
    std::vector<float> embedx_g;
    */

    static int dim(int embedx_dim) { return 4 + embedx_dim; }

    static int dim_size(int dim, int embedx_dim) { return sizeof(float); }
    static int size(int embedx_dim) { return dim(embedx_dim) * sizeof(float); }
    static int slot_index() { return 0; }
    static int show_index() { return DownpourCtrPushValue::slot_index() + 1; }
    static int click_index() { return DownpourCtrPushValue::show_index() + 1; }
    static int embed_g_index() {
      return DownpourCtrPushValue::click_index() + 1;
    }
    static int embedx_g_index() {
      return DownpourCtrPushValue::embed_g_index() + 1;
    }
    static float& slot(float* val) { return val[0]; }
    static float& show(float* val) { return val[1]; }
    static float& click(float* val) { return val[2]; }
    static float& embed_g(float* val) { return val[3]; }
    static float* embedx_g(float* val) { return val + 4; }
  };

  struct DownpourCtrPullValue {
    /*
    float show;
    float click;
    float embed_w;
    std::vector<float> embedx_w;
    */

    static int dim(int embedx_dim) { return 3 + embedx_dim; }
    static int dim_size(size_t dim) { return sizeof(float); }
    static int size(int embedx_dim) { return dim(embedx_dim) * sizeof(float); }
    static int show_index() { return 0; }
    static int click_index() { return 1; }
    static int embed_w_index() { return 2; }
    static int embedx_w_index() { return 3; }
    static float& show(float* val) {
      return val[DownpourCtrPullValue::show_index()];
    }
    static float& click(float* val) {
      return val[DownpourCtrPullValue::click_index()];
    }
    static float& embed_w(float* val) {
      return val[DownpourCtrPullValue::embed_w_index()];
    }
    static float* embedx_w(float* val) {
      return val + DownpourCtrPullValue::embedx_w_index();
    }
  };
  DownpourCtrAccessor() {}
  virtual ~DownpourCtrAccessor() {}

  virtual int initialize();
  virtual void GetTableInfo(AccessorInfo& info);
  // value维度
  virtual size_t dim();
  // value各个维度的size
  virtual size_t dim_size(size_t dim);
  // value各维度相加总size
  virtual size_t size();
  // value中mf动态长度部分总size大小, sparse下生效
  virtual size_t mf_size();
  // pull value维度
  virtual size_t select_dim();
  // pull value各个维度的size
  virtual size_t select_dim_size(size_t dim);
  // pull value各维度相加总size
  virtual size_t select_size();
  // push value维度
  virtual size_t update_dim();
  // push value各个维度的size
  virtual size_t update_dim_size(size_t dim);
  // push value各维度相加总size
  virtual size_t update_size();
  // 判断该value是否进行shrink
  virtual bool shrink(float* value);
  // 判断该value是否保存到ssd
  virtual bool save_ssd(float* value);
  virtual bool need_extend_mf(float* value);
  virtual bool has_mf(size_t size);
  // 判断该value是否在save阶段dump,
  // param作为参数用于标识save阶段，如downpour的xbox与batch_model
  // param = 0, save all feature
  // param = 1, save delta feature
  // param = 3, save all feature with time decay
  virtual bool save(float* value, int param) override;
  // update delta_score and unseen_days after save
  virtual void update_stat_after_save(float* value, int param) override;
  // virtual bool save_cache(float* value, int param, double
  // global_cache_threshold) override;
  // keys不存在时，为values生成随机值
  // 要求value的内存由外部调用者分配完毕
  virtual int32_t create(float** value, size_t num);
  // 从values中选取到select_values中
  virtual int32_t select(float** select_values, const float** values,
                         size_t num);
  // 将update_values聚合到一起
  virtual int32_t merge(float** update_values,
                        const float** other_update_values, size_t num);
  // 将update_values聚合到一起，通过it.next判定是否进入下一个key
  // virtual int32_t merge(float** update_values, iterator it);
  // 将update_values更新应用到values中
  virtual int32_t update(float** values, const float** update_values,
                         size_t num);

  virtual std::string parse_to_string(const float* value, int param) override;
  virtual int32_t parse_from_string(const std::string& str, float* v) override;
  virtual bool create_value(int type, const float* value);

  //这个接口目前只用来取show
  virtual float get_field(float* value, const std::string& name) override {
    CHECK(name == "show");
    if (name == "show") {
      auto unseen_days = DownpourCtrFeatureValue::unseen_days(value);
      int16_t day_diff = _day_id - unseen_days;
      auto show_right =
          DownpourCtrFeatureValue::show(value) * _time_decay_rates[day_diff];
      return (float)show_right;
    }
    return 0.0;
  }
  // DEFINE_GET_INDEX(DownpourCtrFeatureValue, show)
  // DEFINE_GET_INDEX(DownpourCtrFeatureValue, click)
  // DEFINE_GET_INDEX(DownpourCtrFeatureValue, embed_w)
  // DEFINE_GET_INDEX(DownpourCtrFeatureValue, embedx_w)

  virtual void update_time_decay(float* value, bool is_update_seen_day);
  virtual void set_day_id(int day_id);
  virtual int get_day_id();
  bool test_func() { return false; }

 private:
  float show_click_score(float show, float click);
  void set_time_decay_rates();

 private:
  SparseValueSGDRule* _embed_sgd_rule;
  SparseValueSGDRule* _embedx_sgd_rule;
  float _show_click_decay_rate;
  int32_t _ssd_unseenday_threshold;
  std::vector<double> _time_decay_rates;
  int _day_id;
};
}  // namespace distributed
}  // namespace paddle
