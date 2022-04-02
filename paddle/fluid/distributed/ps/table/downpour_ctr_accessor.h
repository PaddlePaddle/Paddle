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

    static int Dim(int embedx_dim) { return 8 + embedx_dim; }
    static int DimSize(size_t dim, int embedx_dim) { return sizeof(float); }
    static int Size(int embedx_dim) { return Dim(embedx_dim) * sizeof(float); }
    static int UnseenDaysIndex() { return 0; }
    static int DeltaScoreIndex() {
      return DownpourCtrFeatureValue::UnseenDaysIndex() + 1;
    }
    static int ShowIndex() {
      return DownpourCtrFeatureValue::DeltaScoreIndex() + 1;
    }
    static int ClickIndex() { return DownpourCtrFeatureValue::ShowIndex() + 1; }
    static int EmbedWIndex() {
      return DownpourCtrFeatureValue::ClickIndex() + 1;
    }
    static int EmbedG2SumIndex() {
      return DownpourCtrFeatureValue::EmbedWIndex() + 1;
    }
    static int SlotIndex() {
      return DownpourCtrFeatureValue::EmbedG2SumIndex() + 1;
    }
    static int EmbedxG2SumIndex() {
      return DownpourCtrFeatureValue::SlotIndex() + 1;
    }
    static int EmbedxWIndex() {
      return DownpourCtrFeatureValue::EmbedxG2SumIndex() + 1;
    }
    static float& UnseenDays(float* val) {
      return val[DownpourCtrFeatureValue::UnseenDaysIndex()];
    }
    static float& DeltaScore(float* val) {
      return val[DownpourCtrFeatureValue::DeltaScoreIndex()];
    }
    static float& Show(float* val) {
      return val[DownpourCtrFeatureValue::ShowIndex()];
    }
    static float& Click(float* val) {
      return val[DownpourCtrFeatureValue::ClickIndex()];
    }
    static float& Slot(float* val) {
      return val[DownpourCtrFeatureValue::SlotIndex()];
    }
    static float& EmbedW(float* val) {
      return val[DownpourCtrFeatureValue::EmbedWIndex()];
    }
    static float& EmbedG2Sum(float* val) {
      return val[DownpourCtrFeatureValue::EmbedG2SumIndex()];
    }
    static float& EmbedxG2Sum(float* val) {
      return val[DownpourCtrFeatureValue::EmbedxG2SumIndex()];
    }
    static float* EmbedxW(float* val) {
      return (val + DownpourCtrFeatureValue::EmbedxWIndex());
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

    static int Dim(int embedx_dim) { return 4 + embedx_dim; }

    static int DimSize(int dim, int embedx_dim) { return sizeof(float); }
    static int Size(int embedx_dim) { return Dim(embedx_dim) * sizeof(float); }
    static int SlotIndex() { return 0; }
    static int ShowIndex() { return DownpourCtrPushValue::SlotIndex() + 1; }
    static int ClickIndex() { return DownpourCtrPushValue::ShowIndex() + 1; }
    static int EmbedGIndex() { return DownpourCtrPushValue::ClickIndex() + 1; }
    static int EmbedxGIndex() {
      return DownpourCtrPushValue::EmbedGIndex() + 1;
    }
    static float& Slot(float* val) { return val[0]; }
    static float& Show(float* val) { return val[1]; }
    static float& Click(float* val) { return val[2]; }
    static float& EmbedG(float* val) { return val[3]; }
    static float* EmbedxG(float* val) { return val + 4; }
  };

  struct DownpourCtrPullValue {
    /*
    float show;
    float click;
    float embed_w;
    std::vector<float> embedx_w;
    */

    static int Dim(int embedx_dim) { return 3 + embedx_dim; }
    static int DimSize(size_t dim) { return sizeof(float); }
    static int Size(int embedx_dim) { return Dim(embedx_dim) * sizeof(float); }
    static int ShowIndex() { return 0; }
    static int ClickIndex() { return 1; }
    static int EmbedWIndex() { return 2; }
    static int EmbedxWIndex() { return 3; }
    static float& Show(float* val) {
      return val[DownpourCtrPullValue::ShowIndex()];
    }
    static float& Click(float* val) {
      return val[DownpourCtrPullValue::ClickIndex()];
    }
    static float& EmbedW(float* val) {
      return val[DownpourCtrPullValue::EmbedWIndex()];
    }
    static float* EmbedxW(float* val) {
      return val + DownpourCtrPullValue::EmbedxWIndex();
    }
  };
  DownpourCtrAccessor() {}
  virtual ~DownpourCtrAccessor() {}

  virtual int Initialize();
  // 初始化AccessorInfo
  virtual void InitAccessorInfo();
  // 判断该value是否进行shrink
  virtual bool Shrink(float* value);
  // 判断该value是否保存到ssd
  virtual bool save_ssd(float* value);
  virtual bool NeedExtendMF(float* value);
  virtual bool HasMF(size_t size);
  // 判断该value是否在save阶段dump,
  // param作为参数用于标识save阶段，如downpour的xbox与batch_model
  // param = 0, save all feature
  // param = 1, save delta feature
  // param = 3, save all feature with time decay
  virtual bool Save(float* value, int param) override;
  // update delta_score and unseen_days after save
  virtual void UpdateStatAfterSave(float* value, int param) override;
  // virtual bool save_cache(float* value, int param, double
  // global_cache_threshold) override;
  // keys不存在时，为values生成随机值
  // 要求value的内存由外部调用者分配完毕
  virtual int32_t Create(float** value, size_t num);
  // 从values中选取到select_values中
  virtual int32_t Select(float** select_values, const float** values,
                         size_t num);
  // 将update_values聚合到一起
  virtual int32_t Merge(float** update_values,
                        const float** other_update_values, size_t num);
  // 将update_values聚合到一起，通过it.next判定是否进入下一个key
  // virtual int32_t Merge(float** update_values, iterator it);
  // 将update_values更新应用到values中
  virtual int32_t Update(float** values, const float** update_values,
                         size_t num);

  virtual std::string ParseToString(const float* value, int param) override;
  virtual int32_t ParseFromString(const std::string& str, float* v) override;
  virtual bool CreateValue(int type, const float* value);

  //这个接口目前只用来取show
  virtual float GetField(float* value, const std::string& name) override {
    CHECK(name == "show");
    if (name == "show") {
      auto unseen_days = DownpourCtrFeatureValue::UnseenDays(value);
      int16_t day_diff = _day_id - unseen_days;
      auto show_right =
          DownpourCtrFeatureValue::Show(value) * _time_decay_rates[day_diff];
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
  float ShowClickScore(float show, float click);
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
