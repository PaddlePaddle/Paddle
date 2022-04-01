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

// DownpourUnitAccessor
class CtrCommonAccessor : public ValueAccessor {
 public:
  struct CtrCommonFeatureValue {
    /*
       float slot;
       float unseen_days;
       float delta_score;
       float show;
       float click;
       float embed_w;
       std::vector<float> embed_g2sum;
       std::vector<float> embedx_w;
       std::<vector>float embedx_g2sum;
       */

    int Dim() { return 6 + embed_sgd_dim + embedx_sgd_dim + embedx_dim; }
    int DimSize(size_t dim, int embedx_dim) { return sizeof(float); }
    int Size() { return Dim() * sizeof(float); }
    int SlotIndex() { return 0; }
    int unseen_days_index() { return SlotIndex() + 1; }
    int delta_score_index() { return unseen_days_index() + 1; }
    int ShowIndex() { return delta_score_index() + 1; }
    int ClickIndex() { return ShowIndex() + 1; }
    int Embed_W_Index() { return ClickIndex() + 1; }
    int embed_g2sum_index() { return Embed_W_Index() + 1; }
    int Embedx_W_Index() { return embed_g2sum_index() + embed_sgd_dim; }
    int embedx_g2sum_index() { return Embedx_W_Index() + embedx_dim; }

    float& unseen_days(float* val) { return val[unseen_days_index()]; }
    float& delta_score(float* val) { return val[delta_score_index()]; }
    float& Show(float* val) { return val[ShowIndex()]; }
    float& Click(float* val) { return val[ClickIndex()]; }
    float& Slot(float* val) { return val[SlotIndex()]; }
    float& EmbedW(float* val) { return val[Embed_W_Index()]; }
    float& embed_g2sum(float* val) { return val[embed_g2sum_index()]; }
    float& EmbedxW(float* val) { return val[Embedx_W_Index()]; }
    float& embedx_g2sum(float* val) { return val[embedx_g2sum_index()]; }

    int embed_sgd_dim;
    int embedx_dim;
    int embedx_sgd_dim;
  };

  struct CtrCommonPushValue {
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
    static int ShowIndex() { return CtrCommonPushValue::SlotIndex() + 1; }
    static int ClickIndex() { return CtrCommonPushValue::ShowIndex() + 1; }
    static int Embed_G_Index() { return CtrCommonPushValue::ClickIndex() + 1; }
    static int Embedx_G_Index() {
      return CtrCommonPushValue::Embed_G_Index() + 1;
    }
    static float& Slot(float* val) {
      return val[CtrCommonPushValue::SlotIndex()];
    }
    static float& Show(float* val) {
      return val[CtrCommonPushValue::ShowIndex()];
    }
    static float& Click(float* val) {
      return val[CtrCommonPushValue::ClickIndex()];
    }
    static float& EmbedG(float* val) {
      return val[CtrCommonPushValue::Embed_G_Index()];
    }
    static float* EmbedxG(float* val) {
      return val + CtrCommonPushValue::Embedx_G_Index();
    }
  };

  struct CtrCommonPullValue {
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
    static int Embed_W_Index() { return 2; }
    static int Embedx_W_Index() { return 3; }
    static float& Show(float* val) {
      return val[CtrCommonPullValue::ShowIndex()];
    }
    static float& Click(float* val) {
      return val[CtrCommonPullValue::ClickIndex()];
    }
    static float& EmbedW(float* val) {
      return val[CtrCommonPullValue::Embed_W_Index()];
    }
    static float* EmbedxW(float* val) {
      return val + CtrCommonPullValue::Embedx_W_Index();
    }
  };
  CtrCommonAccessor() {}
  virtual ~CtrCommonAccessor() {}
  virtual int Initialize();
  // 初始化AccessorInfo
  virtual void InitAccessorInfo();
  // 判断该value是否进行shrink
  virtual bool Shrink(float* value);
  // 判断该value是否保存到ssd
  // virtual bool save_ssd(float* value);
  virtual bool NeedExtendMF(float* value);
  virtual bool HasMF(size_t size);
  // 判断该value是否在save阶段dump,
  // param作为参数用于标识save阶段，如downpour的xbox与batch_model
  // param = 0, save all feature
  // param = 1, save delta feature
  // param = 2, save xbox base feature
  bool Save(float* value, int param) override;
  // update delta_score and unseen_days after save
  void UpdateStatAfterSave(float* value, int param) override;
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

 private:
  // float show_click_score(float show, float click);

  // SparseValueSGDRule* _embed_sgd_rule;
  // SparseValueSGDRule* _embedx_sgd_rule;
  // CtrCommonFeatureValue common_feature_value;
  float _show_click_decay_rate;
  int32_t _ssd_unseenday_threshold;

 public:  // TODO(zhaocaibei123): it should be private, but we make it public
          // for unit test
  CtrCommonFeatureValue common_feature_value;
  float show_click_score(float show, float click);
  SparseValueSGDRule* _embed_sgd_rule;
  SparseValueSGDRule* _embedx_sgd_rule;
};
}  // namespace distributed
}  // namespace paddle
