#pragma once
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include "paddle/fluid/distributed/ps.pb.h"
#include "accessor.h"
#include "sgd/sparse_sgd.h"  //TODO
#include "paddle/fluid/distributed/common/registerer.h"

namespace paddle {
namespace ps {

// double unit accessor
class DownpourDoubleUnitAccessor : public ValueAccessor {
  public:
    struct DownpourDoubleUnitFeatureValue {
      /*
	 float unseen_days;
	 float delta_score;
	 double show;
	 double click;
	 float slot;
	 float embed_w;
	 std::vector<float> embed_g2sum;
	 std::vector<float> embedx_w;
	 std::vector<float> embedx_g2sum;
	 */
      int dim() {
	return 6 + embed_sgd_dim + embedx_sgd_dim + embedx_dim;
      }
      int dim_size(size_t dim, int embedx_dim) {
	return sizeof(float);
      }
      int size() {
	return (dim() + 2) * sizeof(float);
      }
      int unseen_days_index() {
	return 0;
      }
      int delta_score_index() {
	return unseen_days_index() + 1;
      }
      int show_index() {
	return delta_score_index() + 1;
      }
      //show is double
      int click_index() {
	return show_index() + 2;
      }
      int slot_index() {
	return click_index() + 2;
      }
      //click is double
      int embed_w_index() {
	return slot_index() + 1;
      }
      int embed_g2sum_index() {
	return embed_w_index() + 1;
      }
      int embedx_w_index() {
	return embed_g2sum_index() + embed_sgd_dim;
      }
      int embedx_g2sum_index() {
	return embedx_w_index() + embedx_dim;
      }
      float& unseen_days(float* val) {
	return val[unseen_days_index()];
      }
      float& delta_score(float* val) {
	return val[delta_score_index()];
      }
      double& show(float* val) {
	return ((double*)(val + show_index()))[0];
      }
      double& click(float* val) {
	return ((double*)(val + click_index()))[0];
      }
      float& slot(float* val) {
	return val[slot_index()];
      }
      float& embed_w(float* val) {
	return val[embed_w_index()];
      }
      float& embed_g2sum(float* val) {
	return val[embed_g2sum_index()];
      }
      float& embedx_g2sum(float* val) {
	return val[embedx_g2sum_index()];
      }
      float* embedx_w(float* val) {
	return (val + embedx_w_index());
      }
      int embed_sgd_dim;
      int embedx_dim;
      int embedx_sgd_dim;
    };
    struct DownpourDoubleUnitPushValue {
      /*
	 float slot;
	 float show;
	 float click;
	 float embed_g;
	 std::vector<float> embedx_g;
	 */
      static int dim(int embedx_dim) {
	return 4 + embedx_dim;
      }
      static int dim_size(int dim, int embedx_dim) {
	return sizeof(float);
      }
      static int size(int embedx_dim) {
	return dim(embedx_dim) * sizeof(float);
      }
      static int slot_index() {
	return 0;
      }
      static int show_index() {
	return DownpourDoubleUnitPushValue::slot_index() + 1;
      }
      static int click_index() {
	return DownpourDoubleUnitPushValue::show_index() + 1;
      }
      static int embed_g_index() {
	return DownpourDoubleUnitPushValue::click_index() + 1;
      }
      static int embedx_g_index() {
	return DownpourDoubleUnitPushValue::embed_g_index() + 1;
      }
      static float& slot(float* val) {
	return val[DownpourDoubleUnitPushValue::slot_index()];
      }
      static float& show(float* val) {
	return val[DownpourDoubleUnitPushValue::show_index()];
      }
      static float& click(float* val) {
	return val[DownpourDoubleUnitPushValue::click_index()];
      }
      static float& embed_g(float* val) {
	return val[DownpourDoubleUnitPushValue::embed_g_index()];
      }
      static float* embedx_g(float* val) {
	return val + DownpourDoubleUnitPushValue::embedx_g_index();
      }
    };
    struct DownpourDoubleUnitPullValue {
      /*
	 float show;
	 float click;
	 float embed_w;
	 std::vector<float> embedx_w;
	 */
      static int dim(int embedx_dim) {
	return 3 + embedx_dim;
      }
      static int dim_size(size_t dim) {
	return sizeof(float);
      }
      static int size(int embedx_dim) {
	return dim(embedx_dim) * sizeof(float);
      }
      static int show_index() {
	return 0;
      }
      static int click_index() {
	return 1;
      }
      static int embed_w_index() {
	return 2;
      }
      static int embedx_w_index() {
	return 3;
      }
      static float& show(float* val) {
	return val[DownpourDoubleUnitPullValue::show_index()];
      }
      static float& click(float* val) {
	return val[DownpourDoubleUnitPullValue::click_index()];
      }
      static float& embed_w(float* val) {
	return val[DownpourDoubleUnitPullValue::embed_w_index()];
      }
      static float* embedx_w(float* val) {
	return val + DownpourDoubleUnitPullValue::embedx_w_index();
      }
    };
    DownpourDoubleUnitAccessor() {}
    virtual ~DownpourDoubleUnitAccessor() {}
    virtual int initialize();
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
    virtual bool need_extend_mf(float* value);
    // 判断该value是否在save阶段dump, param作为参数用于标识save阶段，如downpour的xbox与batch_model
    // param = 0, save all feature
    // param = 1, save delta feature
    // param = 3, save all feature with time decay
    virtual bool save(float* value, int param) override;
    // update delta_score and unseen_days after save
    virtual void update_stat_after_save(float* value, int param) override;
    // 判断该value是否保存到ssd
    virtual bool save_ssd(float* value);
    virtual bool save_cache(float* value, int param, double global_cache_threshold) override;
    // keys不存在时，为values生成随机值
    // 要求value的内存由外部调用者分配完毕
    virtual int32_t create(float** value, size_t num);
    // 从values中选取到select_values中
    virtual int32_t select(float** select_values, const float** values, size_t num);
    // 将update_values聚合到一起
    virtual int32_t merge(float** update_values, const float** other_update_values, size_t num);
    // 将update_values聚合到一起，通过it.next判定是否进入下一个key
    //virtual int32_t merge(float** update_values, iterator it);
    // 将update_values更新应用到values中
    virtual int32_t update(float** values, const float** update_values, size_t num);
    virtual std::string parse_to_string(const float* value, int param) override;
    virtual int32_t parse_from_string(const std::string& str, float* v) override;
    virtual bool create_value(int type, const float* value);
    //这个接口目前只用来取show
    virtual float get_field(float* value, const std::string& name) override {
      CHECK(name == "show");
      if (name == "show") {
	return (float)unit_feature_value.show(value);
      }
      return 0.0;
    }
  private:
    double show_click_score(double show, double click);
  private:
    std::shared_ptr<SparseValueSGDRule> _embed_sgd_rule;
    std::shared_ptr<SparseValueSGDRule> _embedx_sgd_rule;
    DownpourDoubleUnitFeatureValue unit_feature_value;
    float         _show_click_decay_rate;
    int32_t      _ssd_unseenday_threshold;
};

/** 
 * @brief Accessor for unit
 **/
class DownpourUnitAccessor : public ValueAccessor {
  public:
    struct DownpourUnitFeatureValue {
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

      int dim() {
	return 6 + embed_sgd_dim + embedx_sgd_dim + embedx_dim;
      }
      int dim_size(size_t dim, int embedx_dim) {
	return sizeof(float);
      }
      int size() {
	return dim() * sizeof(float);
      }
      int slot_index() {
	return 0;
      }
      int unseen_days_index() {
	return slot_index() + 1;
      }
      int delta_score_index() {
	return unseen_days_index() + 1;
      } 
      int show_index() {
	return delta_score_index() + 1;
      }
      int click_index() {
	return show_index() + 1;
      }
      int embed_w_index() {
	return click_index() + 1;
      }
      int embed_g2sum_index() {
	return embed_w_index() + 1;
      }
      int embedx_w_index() {
	return embed_g2sum_index() + embed_sgd_dim;
      }
      int embedx_g2sum_index() {
	return embedx_w_index() + embedx_dim;
      }

      float& unseen_days(float* val) {
	return val[unseen_days_index()];
      }
      float& delta_score(float* val) {
	return val[delta_score_index()];
      }
      float& show(float* val) {
	return val[show_index()];
      }
      float& click(float* val) {
	return val[click_index()];
      }
      float& slot(float* val) {
	return val[slot_index()];
      }
      float& embed_w(float* val) {
	return val[embed_w_index()];
      }
      float& embed_g2sum(float* val) {
	return val[embed_g2sum_index()];
      }
      float& embedx_w(float* val) {
	return val[embedx_w_index()];
      }
      float& embedx_g2sum(float* val) {
	return val[embedx_g2sum_index()];
      }
      int embed_sgd_dim;
      int embedx_dim;
      int embedx_sgd_dim;
    };

    struct DownpourUnitPushValue {
      /*
	 float slot;
	 float show;
	 float click;
	 float embed_g;
	 std::vector<float> embedx_g;
	 */

      static int dim(int embedx_dim) {
	return 4 + embedx_dim;
      }

      static int dim_size(int dim, int embedx_dim) {
	return sizeof(float);
      }
      static int size(int embedx_dim) {
	return dim(embedx_dim) * sizeof(float);
      }
      static int slot_index() {
	return 0;
      }
      static int show_index() {
	return DownpourUnitPushValue::slot_index() + 1;
      }
      static int click_index() {
	return DownpourUnitPushValue::show_index() + 1;
      }
      static int embed_g_index() {
	return DownpourUnitPushValue::click_index() + 1;
      }
      static int embedx_g_index() {
	return DownpourUnitPushValue::embed_g_index() + 1;
      }
      static float& slot(float* val) {
	return val[0];
      }
      static float& show(float* val) {
	return val[1];
      }
      static float& click(float* val) {
	return val[2];
      }
      static float& embed_g(float* val) {
	return val[3];
      }
      static float* embedx_g(float* val) {
	return val + 4;
      }
    };

    struct DownpourUnitPullValue {
      /*
	 float show;
	 float click;
	 float embed_w;
	 std::vector<float> embedx_w;
	 */

      static int dim(int embedx_dim) {
	return 3 + embedx_dim;
      }
      static int dim_size(size_t dim) {
	return sizeof(float);
      }
      static int size(int embedx_dim) {
	return dim(embedx_dim) * sizeof(float); 
      }
      static int show_index() {
	return 0;
      }
      static int click_index() {
	return 1;
      }
      static int embed_w_index() {
	return 2;
      }
      static int embedx_w_index() {
	return 3;
      }
      static float& show(float* val) {
	return val[DownpourUnitPullValue::show_index()];
      }
      static float& click(float* val) {
	return val[DownpourUnitPullValue::click_index()];
      }
      static float& embed_w(float* val) {
	return val[DownpourUnitPullValue::embed_w_index()];
      }
      static float* embedx_w(float* val) {
	return val + DownpourUnitPullValue::embedx_w_index();
      }
    };
    DownpourUnitAccessor() {}
    virtual int initialize();
    virtual ~DownpourUnitAccessor() {}

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
    // 判断该value是否在save阶段dump, param作为参数用于标识save阶段，如downpour的xbox与batch_model
    // param = 0, save all feature
    // param = 1, save delta feature
    // param = 3, save all feature with time decay
    virtual bool save(float* value, int param) override;
    // update delta_score and unseen_days after save
    virtual void update_stat_after_save(float* value, int param) override;
    virtual bool save_cache(float* value, int param, double global_cache_threshold) override;
    // keys不存在时，为values生成随机值
    // 要求value的内存由外部调用者分配完毕
    virtual int32_t create(float** value, size_t num);
    // 从values中选取到select_values中
    virtual int32_t select(float** select_values, const float** values, size_t num);
    // 将update_values聚合到一起
    virtual int32_t merge(float** update_values, const float** other_update_values, size_t num);
    // 将update_values聚合到一起，通过it.next判定是否进入下一个key
    //virtual int32_t merge(float** update_values, iterator it);
    // 将update_values更新应用到values中
    virtual int32_t update(float** values, const float** update_values, size_t num);

    virtual std::string parse_to_string(const float* value, int param) override;
    virtual int32_t parse_from_string(const std::string& str, float* v) override;
    virtual bool create_value(int type, const float* value);

    //这个接口目前只用来取show
    virtual float get_field(float* value, const std::string& name) override {
      CHECK(name == "show");
      if (name == "show") {
	return unit_feature_value.show(value);
      }
      return 0.0;
    }
  private:
    float show_click_score(float show, float click);

  private:
    std::shared_ptr<SparseValueSGDRule> _embed_sgd_rule;
    std::shared_ptr<SparseValueSGDRule> _embedx_sgd_rule;
    DownpourUnitFeatureValue unit_feature_value;
    float         _show_click_decay_rate;
    int32_t      _ssd_unseenday_threshold;
};

/** 
 * @brief Accessor for unit
 **/
class DownpourCommonAccessor : public ValueAccessor {
  public:
    struct DownpourCommonFeatureValue {
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

      int dim() {
	return 6 + embed_sgd_dim + embedx_sgd_dim + embedx_dim;
      }
      int dim_size(size_t dim, int embedx_dim) {
	return sizeof(float);
      }
      int size() {
	return dim() * sizeof(float);
      }
      int slot_index() {
	return 0;
      }
      int unseen_days_index() {
	return slot_index() + 1;
      }
      int delta_score_index() {
	return unseen_days_index() + 1;
      } 
      int show_index() {
	return delta_score_index() + 1;
      }
      int click_index() {
	return show_index() + 1;
      }
      int embed_w_index() {
	return click_index() + 1;
      }
      int embed_g2sum_index() {
	return embed_w_index() + 1;
      }
      int embedx_w_index() {
	return embed_g2sum_index() + embed_sgd_dim;
      }
      int embedx_g2sum_index() {
	return embedx_w_index() + embedx_dim;
      }

      float& unseen_days(float* val) {
	return val[unseen_days_index()];
      }
      float& delta_score(float* val) {
	return val[delta_score_index()];
      }
      float& show(float* val) {
	return val[show_index()];
      }
      float& click(float* val) {
	return val[click_index()];
      }
      float& slot(float* val) {
	return val[slot_index()];
      }
      float& embed_w(float* val) {
	return val[embed_w_index()];
      }
      float& embed_g2sum(float* val) {
	return val[embed_g2sum_index()];
      }
      float& embedx_w(float* val) {
	return val[embedx_w_index()];
      }
      float& embedx_g2sum(float* val) {
	return val[embedx_g2sum_index()];
      }
      int embed_sgd_dim;
      int embedx_dim;
      int embedx_sgd_dim;
    };

    struct DownpourCommonPushValue {
      /*
	 float slot;
	 float show;
	 float click;
	 float embed_g;
	 std::vector<float> embedx_g;
	 */

      static int dim(int embedx_dim) {
	return 4 + embedx_dim;
      }

      static int dim_size(int dim, int embedx_dim) {
	return sizeof(float);
      }
      static int size(int embedx_dim) {
	return dim(embedx_dim) * sizeof(float);
      }
      static int slot_index() {
	return 0;
      }
      static int show_index() {
	return DownpourCommonPushValue::slot_index() + 1;
      }
      static int click_index() {
	return DownpourCommonPushValue::show_index() + 1;
      }
      static int embed_g_index() {
	return DownpourCommonPushValue::click_index() + 1;
      }
      static int embedx_g_index() {
	return DownpourCommonPushValue::embed_g_index() + 1;
      }
      static float& slot(float* val) {
	return val[DownpourCommonPushValue::slot_index()];
      }
      static float& show(float* val) {
	return val[DownpourCommonPushValue::show_index()];
      }
      static float& click(float* val) {
	return val[DownpourCommonPushValue::click_index()];
      }
      static float& embed_g(float* val) {
	return val[DownpourCommonPushValue::embed_g_index()];
      }
      static float* embedx_g(float* val) {
	return val + DownpourCommonPushValue::embedx_g_index();
      }
    };

    struct DownpourCommonPullValue {
      /*
	 float embed_w;
	 std::vector<float> embedx_w;
	 */

      static int dim(int embedx_dim) {
	return 1 + embedx_dim;
      }
      static int dim_size(size_t dim) {
	return sizeof(float);
      }
      static int size(int embedx_dim) {
	return dim(embedx_dim) * sizeof(float); 
      }
      static int embed_w_index() {
	return 0;
      }
      static int embedx_w_index() {
	return 1;
      }
      static float& embed_w(float* val) {
	return val[DownpourCommonPullValue::embed_w_index()];
      }
      static float* embedx_w(float* val) {
	return val + DownpourCommonPullValue::embedx_w_index();
      }
    };
    DownpourCommonAccessor() {}
    virtual int initialize();
    virtual ~DownpourCommonAccessor() {}

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
    // 判断该value是否在save阶段dump, param作为参数用于标识save阶段，如downpour的xbox与batch_model
    // param = 0, save all feature
    // param = 1, save delta feature
    // param = 2, save xbox base feature
    virtual bool save(float* value, int param) override;
    // update delta_score and unseen_days after save
    virtual void update_stat_after_save(float* value, int param) override;
    // keys不存在时，为values生成随机值
    // 要求value的内存由外部调用者分配完毕
    virtual int32_t create(float** value, size_t num);
    // 从values中选取到select_values中
    virtual int32_t select(float** select_values, const float** values, size_t num);
    // 将update_values聚合到一起
    virtual int32_t merge(float** update_values, const float** other_update_values, size_t num);
    // 将update_values聚合到一起，通过it.next判定是否进入下一个key
    //virtual int32_t merge(float** update_values, iterator it);
    // 将update_values更新应用到values中
    virtual int32_t update(float** values, const float** update_values, size_t num);

    virtual std::string parse_to_string(const float* value, int param) override;
    virtual int32_t parse_from_string(const std::string& str, float* v) override;
    virtual bool create_value(int type, const float* value);

    //这个接口目前只用来取show
    virtual float get_field(float* value, const std::string& name) override {
      CHECK(name == "show");
      if (name == "show") {
	return unit_feature_value.show(value);
      }
      return 0.0;
    }
  private:
    float show_click_score(float show, float click);

  private:
    std::shared_ptr<SparseValueSGDRule> _embed_sgd_rule;
    std::shared_ptr<SparseValueSGDRule> _embedx_sgd_rule;
    DownpourCommonFeatureValue common_feature_value;
    float         _show_click_decay_rate;
    int32_t      _ssd_unseenday_threshold;
};

