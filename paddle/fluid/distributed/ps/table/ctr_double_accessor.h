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

#include "paddle/common/enforce.h"
#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/sparse_sgd_rule.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"

namespace paddle {
namespace distributed {

class CtrDoubleAccessor : public ValueAccessor {
 public:
  struct CtrDoubleFeatureValue {
    /*
    float unseen_days;
    float delta_score;
    double show;
    double click;
    float embed_w;
    float embed_g2sum;
    float slot;
    float embedx_g2sum;
    std::vector<float> embedx_w;
    */
    static int Dim(int embedx_dim) { return 8 + embedx_dim; }
    static int DimSize(size_t dim, int embedx_dim) { return sizeof(float); }
    static int Size(int embedx_dim) {
      return (Dim(embedx_dim) + 2) * sizeof(float);
    }
    static int UnseenDaysIndex() { return 0; }
    static int DeltaScoreIndex() {
      return CtrDoubleFeatureValue::UnseenDaysIndex() + 1;
    }
    static int ShowIndex() {
      return CtrDoubleFeatureValue::DeltaScoreIndex() + 1;
    }
    // show is double
    static int ClickIndex() { return CtrDoubleFeatureValue::ShowIndex() + 2; }
    // click is double
    static int EmbedWIndex() { return CtrDoubleFeatureValue::ClickIndex() + 2; }
    static int EmbedG2SumIndex() {
      return CtrDoubleFeatureValue::EmbedWIndex() + 1;
    }
    static int SlotIndex() {
      return CtrDoubleFeatureValue::EmbedG2SumIndex() + 1;
    }
    static int EmbedxG2SumIndex() {
      return CtrDoubleFeatureValue::SlotIndex() + 1;
    }
    static int EmbedxWIndex() {
      return CtrDoubleFeatureValue::EmbedxG2SumIndex() + 1;
    }
    static float& UnseenDays(float* val) {
      return val[CtrDoubleFeatureValue::UnseenDaysIndex()];
    }
    static float& DeltaScore(float* val) {
      return val[CtrDoubleFeatureValue::DeltaScoreIndex()];
    }
    static double& Show(float* val) {
      return (reinterpret_cast<double*>(val +
                                        CtrDoubleFeatureValue::ShowIndex()))[0];
    }
    static double& Click(float* val) {
      return (reinterpret_cast<double*>(
          val + CtrDoubleFeatureValue::ClickIndex()))[0];
    }
    static float& Slot(float* val) {
      return val[CtrDoubleFeatureValue::SlotIndex()];
    }
    static float& EmbedW(float* val) {
      return val[CtrDoubleFeatureValue::EmbedWIndex()];
    }
    static float& EmbedG2Sum(float* val) {
      return val[CtrDoubleFeatureValue::EmbedG2SumIndex()];
    }
    static float& EmbedxG2Sum(float* val) {
      return val[CtrDoubleFeatureValue::EmbedxG2SumIndex()];
    }
    static float* EmbedxW(float* val) {
      return (val + CtrDoubleFeatureValue::EmbedxWIndex());
    }
  };
  struct CtrDoublePushValue {
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
    static int ShowIndex() { return CtrDoublePushValue::SlotIndex() + 1; }
    static int ClickIndex() { return CtrDoublePushValue::ShowIndex() + 1; }
    static int EmbedGIndex() { return CtrDoublePushValue::ClickIndex() + 1; }
    static int EmbedxGIndex() { return CtrDoublePushValue::EmbedGIndex() + 1; }
    static float& Slot(float* val) {
      return val[CtrDoublePushValue::SlotIndex()];
    }
    static float& Show(float* val) {
      return val[CtrDoublePushValue::ShowIndex()];
    }
    static float& Click(float* val) {
      return val[CtrDoublePushValue::ClickIndex()];
    }
    static float& EmbedG(float* val) {
      return val[CtrDoublePushValue::EmbedGIndex()];
    }
    static float* EmbedxG(float* val) {
      return val + CtrDoublePushValue::EmbedxGIndex();
    }
  };
  struct CtrDoublePullValue {
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
      return val[CtrDoublePullValue::ShowIndex()];
    }
    static float& Click(float* val) {
      return val[CtrDoublePullValue::ClickIndex()];
    }
    static float& EmbedW(float* val) {
      return val[CtrDoublePullValue::EmbedWIndex()];
    }
    static float* EmbedxW(float* val) {
      return val + CtrDoublePullValue::EmbedxWIndex();
    }
  };
  CtrDoubleAccessor() {}
  virtual ~CtrDoubleAccessor() {}
  virtual int Initialize();
  // 初始化AccessorInfo
  virtual void InitAccessorInfo();
  // 判断该value是否进行shrink
  virtual bool Shrink(float* value);
  virtual bool NeedExtendMF(float* value);
  // 判断该value是否在save阶段dump,
  // param作为参数用于标识save阶段，如downpour的xbox与batch_model
  // param = 0, save all feature
  // param = 1, save delta feature
  // param = 3, save all feature with time decay
  bool Save(float* value, int param) override;
  bool SaveCache(float* value,
                 int param,
                 double global_cache_threshold) override;
  // update delta_score and unseen_days after save
  void UpdateStatAfterSave(float* value, int param) override;
  // 判断该value是否保存到ssd
  virtual bool SaveSSD(float* value);
  // virtual bool save_cache(float* value, int param, double
  // global_cache_threshold) override;
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
    PADDLE_ENFORCE_EQ(name,
                      "show",
                      common::errors::InvalidArgument("name must be 'show', "
                                                      "but received '%s'",
                                                      name.c_str()));
    if (name == "show") {
      return static_cast<float>(CtrDoubleFeatureValue::Show(value));
    }
    return 0.0;
  }
  // DEFINE_GET_INDEX(CtrDoubleFeatureValue, show)
  // DEFINE_GET_INDEX(CtrDoubleFeatureValue, click)
  // DEFINE_GET_INDEX(CtrDoubleFeatureValue, embed_w)
  // DEFINE_GET_INDEX(CtrDoubleFeatureValue, embedx_w)
 private:
  double ShowClickScore(double show, double click);

 private:
  SparseValueSGDRule* _embed_sgd_rule;
  SparseValueSGDRule* _embedx_sgd_rule;
  float _show_click_decay_rate;
  int32_t _ssd_unseenday_threshold;
  bool _show_scale = false;
};
}  // namespace distributed
}  // namespace paddle
