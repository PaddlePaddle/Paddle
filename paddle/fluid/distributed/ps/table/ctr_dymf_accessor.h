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
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/sparse_sgd_rule.h"
#include "paddle/fluid/distributed/ps/thirdparty/round_robin.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"

namespace paddle {
namespace distributed {

// DownpourUnitAccessor
class CtrDymfAccessor : public ValueAccessor {
 public:
  struct CtrDymfFeatureValue {
    /*v1: old version
      float unseen_days;
      float delta_score;
      float show;
      float click;
      float embed_w;
      // float embed_g2sum;
      std::vector<float> embed_g2sum;
      float slot;
      float mf_dim
      std::<vector>float embedx_g2sum;
      // float embedx_g2sum;
      std::vector<float> embedx_w;
       */
    /* V2: support pass_id (pslib gpups)
      uint16_t pass_id; (diff)
      uint16_t unseen_days; (diff)
      float show;
      float click;
      float embed_w;
      // float embed_g2sum;
      std::vector<float> embed_g2sum;
      float slot;
      float mf_dim
      std::<vector>float embedx_g2sum;
      // float embedx_g2sum;
      std::vector<float> embedx_w;
    */

    int Dim() { return 7 + embed_sgd_dim + embedx_sgd_dim + embedx_dim; }
    int DimSize(size_t dim, int embedx_dim) { return sizeof(float); }
    int Size() { return Dim() * sizeof(float); }
    int UnseenDaysIndex() { return 0; }
    int DeltaScoreIndex() { return UnseenDaysIndex() + 1; }
    int ShowIndex() { return DeltaScoreIndex() + 1; }
    int ClickIndex() { return ShowIndex() + 1; }
    int EmbedWIndex() { return ClickIndex() + 1; }
    int EmbedG2SumIndex() { return EmbedWIndex() + 1; }
    int SlotIndex() { return EmbedG2SumIndex() + embed_sgd_dim; }
    int MfDimIndex() { return SlotIndex() + 1; }
    int EmbedxG2SumIndex() { return MfDimIndex() + 1; }
    int EmbedxWIndex() { return EmbedxG2SumIndex() + embedx_sgd_dim; }

    // 根据mf_dim计算的总长度
    int Dim(int mf_dim) {
      int tmp_embedx_sgd_dim = 1;
      if (optimizer_name == "SparseAdamSGDRule") {  // adam
        tmp_embedx_sgd_dim = mf_dim * 2 + 2;
      } else if (optimizer_name == "SparseSharedAdamSGDRule") {  // shared_adam
        tmp_embedx_sgd_dim = 4;
      }
      return 7 + embed_sgd_dim + tmp_embedx_sgd_dim + mf_dim;
    }

    // 根据mf_dim计算的总byte数
    int Size(int mf_dim) { return (Dim(mf_dim)) * sizeof(float); }

#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_HETERPS)
    uint16_t& PassId(float* val) {
      uint16_t* int16_val =
          reinterpret_cast<uint16_t*>(val + UnseenDaysIndex());
      return int16_val[0];
    }

    uint16_t& UnseenDays(float* val) {
      uint16_t* int16_val =
          reinterpret_cast<uint16_t*>(val + UnseenDaysIndex());
      return int16_val[1];
    }
#else
    float& UnseenDays(float* val) { return val[UnseenDaysIndex()]; }
#endif
    float& DeltaScore(float* val) { return val[DeltaScoreIndex()]; }
    float& Show(float* val) { return val[ShowIndex()]; }
    float& Click(float* val) { return val[ClickIndex()]; }
    float& Slot(float* val) { return val[SlotIndex()]; }
    float& MfDim(float* val) { return val[MfDimIndex()]; }
    float& EmbedW(float* val) { return val[EmbedWIndex()]; }
    float& EmbedG2Sum(float* val) { return val[EmbedG2SumIndex()]; }
    float& EmbedxG2Sum(float* val) { return val[EmbedxG2SumIndex()]; }
    float& EmbedxW(float* val) { return val[EmbedxWIndex()]; }

    int embed_sgd_dim;
    int embedx_dim;
    int embedx_sgd_dim;
    std::string optimizer_name;
  };

  struct CtrDymfPushValue {
    /*
       float slot;
       float show;
       float click;
       float mf_dim;
       float embed_g;
       std::vector<float> embedx_g;
       */

    static int Dim(int embedx_dim) { return 5 + embedx_dim; }

    static int DimSize(int dim, int embedx_dim) { return sizeof(float); }
    static int Size(int embedx_dim) { return Dim(embedx_dim) * sizeof(float); }
    static int SlotIndex() { return 0; }
    static int ShowIndex() { return CtrDymfPushValue::SlotIndex() + 1; }
    static int ClickIndex() { return CtrDymfPushValue::ShowIndex() + 1; }
    static int MfDimIndex() { return CtrDymfPushValue::ClickIndex() + 1; }
    static int EmbedGIndex() { return CtrDymfPushValue::MfDimIndex() + 1; }
    static int EmbedxGIndex() { return CtrDymfPushValue::EmbedGIndex() + 1; }
    static float& Slot(float* val) {
      return val[CtrDymfPushValue::SlotIndex()];
    }
    static float& Show(float* val) {
      return val[CtrDymfPushValue::ShowIndex()];
    }
    static float& Click(float* val) {
      return val[CtrDymfPushValue::ClickIndex()];
    }
    static float& MfDim(float* val) {
      return val[CtrDymfPushValue::MfDimIndex()];
    }
    static float& EmbedG(float* val) {
      return val[CtrDymfPushValue::EmbedGIndex()];
    }
    static float* EmbedxG(float* val) {
      return val + CtrDymfPushValue::EmbedxGIndex();
    }
  };

  struct CtrDymfPullValue {
    /*
       float show;
       float click;
       float mf_dim;
       float embed_w;
       std::vector<float> embedx_w;
       */

    static int Dim(int embedx_dim) { return 4 + embedx_dim; }
    static int DimSize(size_t dim) { return sizeof(float); }
    static int Size(int embedx_dim) { return Dim(embedx_dim) * sizeof(float); }
    static int ShowIndex() { return 0; }
    static int ClickIndex() { return 1; }
    static int MfDimIndex() { return 2; }
    static int EmbedWIndex() { return 3; }
    static int EmbedxWIndex() { return 4; }
    static float& Show(float* val) {
      return val[CtrDymfPullValue::ShowIndex()];
    }
    static float& Click(float* val) {
      return val[CtrDymfPullValue::ClickIndex()];
    }
    static float& MfDim(float* val) {
      return val[CtrDymfPullValue::MfDimIndex()];
    }
    static float& EmbedW(float* val) {
      return val[CtrDymfPullValue::EmbedWIndex()];
    }
    static float* EmbedxW(float* val) {
      return val + CtrDymfPullValue::EmbedxWIndex();
    }
  };
  CtrDymfAccessor() {}
  virtual ~CtrDymfAccessor() {}
  virtual int Initialize();
  // 初始化AccessorInfo
  virtual void InitAccessorInfo();
  // 判断该value是否进行shrink
  virtual bool Shrink(float* value);
  // 判断该value是否保存到ssd
  // virtual bool save_ssd(float* value);
  virtual bool NeedExtendMF(float* value);
  virtual bool HasMF(int size);
  // 判断该value是否在save阶段dump,
  // param作为参数用于标识save阶段，如downpour的xbox与batch_model
  // param = 0, save all feature
  // param = 1, save delta feature
  // param = 2, save xbox base feature
  bool Save(float* value, int param) override;
  bool SaveCache(float* value,
                 int param,
                 double global_cache_threshold) override;
  bool SaveSSD(float* value) override;
  bool FilterSlot(float* value);
  bool SaveFilterSlot(float* value) override;
  // update delta_score and unseen_days after save
  void UpdateStatAfterSave(float* value, int param) override;
  // keys不存在时，为values生成随机值
  // 要求value的内存由外部调用者分配完毕
  virtual int32_t Create(float** value, size_t num);
  // 从values中选取到select_values中
  virtual int32_t Select(float** select_values,
                         const float** values,
                         size_t num);
  // 将update_values聚合到一起
  virtual int32_t Merge(float** update_values,
                        const float** other_update_values,
                        size_t num);
  // 将update_values聚合到一起，通过it.next判定是否进入下一个key
  // virtual int32_t Merge(float** update_values, iterator it);
  // 将update_values更新应用到values中
  virtual int32_t Update(float** values,
                         const float** update_values,
                         size_t num);

  std::string ParseToString(const float* value, int param) override;
  int32_t ParseFromString(const std::string& str, float* v) override;
  virtual bool CreateValue(int type, const float* value);

  // 这个接口目前只用来取show
  float GetField(float* value, const std::string& name) override {
    // CHECK(name == "show");
    if (name == "show") {
      return common_feature_value.Show(value);
    }
    return 0.0;
  }

  robin_hood::unordered_set<float>* GetFilteredSlots() override {
    return &_filtered_slots;
  }

  robin_hood::unordered_set<float>* GetSaveFilteredSlots() override {
    return &_save_filtered_slots;
  }

  void SetDayId(int day_id) override;

#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_HETERPS)
  // 根据pass_id和show_threshold阈值来判断cache到ssd
  bool SaveMemCache(float* value,
                    int param,
                    double global_cache_threshold,
                    uint16_t pass_id);
  // 更新pass_id
  void UpdatePassId(float* value, uint16_t pass_id);
#endif
  void UpdateTimeDecay(float* value, bool is_update_seen_day);

 private:
  void SetTimeDecayRates();
  // float ShowClickScore(float show, float click);

  // SparseValueSGDRule* _embed_sgd_rule;
  // SparseValueSGDRule* _embedx_sgd_rule;
  // CtrDymfFeatureValue common_feature_value;
  float _show_click_decay_rate;
  int32_t _ssd_unseenday_threshold;
  bool _show_scale = false;
  int _day_id = 0;
  std::vector<double> _time_decay_rates;

 public:  // TODO(zhaocaibei123): it should be private, but we make it public
          // for unit test
  CtrDymfFeatureValue common_feature_value;
  float ShowClickScore(float show, float click);
  SparseValueSGDRule* _embed_sgd_rule;
  SparseValueSGDRule* _embedx_sgd_rule;
  robin_hood::unordered_set<float> _filtered_slots;
  robin_hood::unordered_set<float> _save_filtered_slots;
};
}  // namespace distributed
}  // namespace paddle
