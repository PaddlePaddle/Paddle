/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/distributed/ps/table/ctr_accessor.h"
#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/ps/table/sparse_sgd_rule.h"

namespace paddle {
namespace distributed {
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseAdaGradSGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, StdAdaGradSGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseAdamSGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseNaiveSGDRule);

TableAccessorParameter gen_param() {
  TableAccessorParameter param;
  param.set_accessor_class("CtrCommonAccessor");
  param.set_fea_dim(11);
  param.set_embedx_dim(8);
  param.mutable_ctr_accessor_param()->set_nonclk_coeff(0.2);
  param.mutable_ctr_accessor_param()->set_click_coeff(1);
  param.mutable_ctr_accessor_param()->set_base_threshold(0.5);
  param.mutable_ctr_accessor_param()->set_delta_threshold(0.2);
  param.mutable_ctr_accessor_param()->set_delta_keep_days(16);
  param.mutable_ctr_accessor_param()->set_show_click_decay_rate(0.99);
  /*
  param.mutable_embed_sgd_param()->set_name("naive");
  auto* naive_param = param.mutable_embed_sgd_param()->mutable_naive();
  naive_param->set_learning_rate(0.1);
  naive_param->set_initial_range(0.3);
  naive_param->add_weight_bounds(-10.0);
  naive_param->add_weight_bounds(10.0);
  */
  param.mutable_embed_sgd_param()->set_name("StdAdaGradSGDRule");
  auto* adagrad_param = param.mutable_embed_sgd_param()->mutable_adagrad();
  adagrad_param->set_learning_rate(0.1);
  adagrad_param->set_initial_range(0.3);
  adagrad_param->set_initial_g2sum(0.0);
  adagrad_param->add_weight_bounds(-10.0);
  adagrad_param->add_weight_bounds(10.0);

  param.mutable_embedx_sgd_param()->set_name("SparseNaiveSGDRule");
  auto* naive_param = param.mutable_embedx_sgd_param()->mutable_naive();
  naive_param->set_learning_rate(0.1);
  naive_param->set_initial_range(0.3);
  naive_param->add_weight_bounds(-10.0);
  naive_param->add_weight_bounds(10.0);

  return std::move(param);
}

TEST(downpour_feature_value_accessor_test, test_shrink) {
  TableAccessorParameter parameter = gen_param();
  CtrCommonAccessor* acc = new CtrCommonAccessor();
  ASSERT_EQ(acc->configure(parameter), 0);
  ASSERT_EQ(acc->initialize(), 0);

  VLOG(3) << "size of struct: " << acc->common_feature_value.embed_sgd_dim
          << " " << acc->common_feature_value.embedx_dim << " "
          << acc->common_feature_value.embedx_sgd_dim << " "
          << acc->common_feature_value.dim() << "\n";

  float* value = new float[acc->dim()];
  for (auto i = 0u; i < acc->dim(); ++i) {
    value[i] = i * 1.0;
  }
  ASSERT_TRUE(!acc->shrink(value));

  // set unseen_days too long
  value[1] = 1000;
  // set delta score too small
  value[2] = 0.001;
  ASSERT_TRUE(acc->shrink(value));
}

TEST(downpour_feature_value_accessor_test, test_save) {
  TableAccessorParameter parameter = gen_param();
  CtrCommonAccessor* acc = new CtrCommonAccessor();
  ASSERT_EQ(acc->configure(parameter), 0);
  ASSERT_EQ(acc->initialize(), 0);

  float* value = new float[acc->dim()];
  for (auto i = 0u; i < acc->dim(); ++i) {
    value[i] = i * 1.0;
  }

  // save all feature
  ASSERT_TRUE(acc->save(value, 0));

  // save delta feature
  ASSERT_TRUE(acc->save(value, 1));

  // save base feature with time decay
  ASSERT_TRUE(acc->save(value, 2));

  VLOG(3) << "test_save:";
  for (auto i = 0u; i < acc->dim(); ++i) {
    VLOG(3) << value[i];
  }
}

TEST(downpour_feature_value_accessor_test, test_create) {
  TableAccessorParameter parameter = gen_param();
  CtrCommonAccessor* acc = new CtrCommonAccessor();
  ASSERT_EQ(acc->configure(parameter), 0);
  ASSERT_EQ(acc->initialize(), 0);

  const int field_size = 7 + 8;
  const int item_size = 10;

  float** value = new float*[item_size];
  for (auto i = 0u; i < item_size; ++i) {
    value[i] = new float[field_size];
  }
  ASSERT_EQ(acc->create(value, item_size), 0);

  for (auto i = 0u; i < item_size; ++i) {
    for (auto j = 0u; j < field_size; ++j) {
      VLOG(3) << value[i][j] << " ";
      // ASSERT_FLOAT_EQ(value[i][j], 0);
    }
    VLOG(3) << "\n";
  }
}

TEST(downpour_feature_value_accessor_test, test_update) {
  TableAccessorParameter parameter = gen_param();
  CtrCommonAccessor* acc = new CtrCommonAccessor();
  ASSERT_EQ(acc->configure(parameter), 0);
  ASSERT_EQ(acc->initialize(), 0);

  VLOG(3) << "dim: " << acc->common_feature_value.dim() << "\n";
  VLOG(3) << "update_dim: " << acc->update_dim() << "\n";

  const int field_size = 7 + 8;
  const int item_size = 10;

  float** value = new float*[item_size];
  for (auto i = 0u; i < item_size; ++i) {
    value[i] = new float[field_size];

    for (auto j = 0u; j < field_size; ++j) {
      value[i][j] = 0;
    }
  }

  typedef const float* const_float_ptr;
  const_float_ptr* grad = new const_float_ptr[item_size];
  for (auto i = 0u; i < item_size; ++i) {
    float* p = new float[acc->update_dim()];
    for (auto j = 0u; j < acc->update_dim(); ++j) {
      p[j] = i;
    }
    grad[i] = p;
  }

  struct DownpourSparseValueTest {
    float slot;
    float unseen_days;
    float delta_score;
    float show;
    float click;
    float embed_w;
    std::vector<float> embed_g2sum;
    std::vector<float> embedx_w;
    std::vector<float> embedx_g2sum;

    void to_array(float* ptr, size_t dim) {
      ptr[0] = slot;
      ptr[1] = unseen_days;
      ptr[2] = delta_score;
      ptr[3] = show;
      ptr[4] = click;
      ptr[5] = embed_w;
      int idx = 6;
      for (auto j = 0u; j < 1; ++j) {
        ptr[idx + j] = embed_g2sum[j];
      }
      idx += 1;
      for (auto j = 0u; j < 8; ++j) {
        ptr[idx + j] = embedx_w[j];
      }
      idx += 8;
      for (auto j = 0u; j < 0; ++j) {
        ptr[idx + j] = embedx_g2sum[j];
      }
    }
  };
  struct DownpourSparsePushValueTest {
    float slot;
    float show;
    float click;
    float embed_g;
    std::vector<float> embedx_g;
  };
  std::vector<float*> exp_value;
  for (auto i = 0u; i < item_size; ++i) {
    DownpourSparseValueTest v;
    v.slot = value[i][0];
    v.unseen_days = value[i][1];
    v.delta_score = value[i][2];
    v.show = value[i][3];
    v.click = value[i][4];
    v.embed_w = value[i][5];

    int idx = 6;
    for (auto j = 0u; j < acc->common_feature_value.embed_sgd_dim; ++j) {
      v.embed_g2sum.push_back(value[i][idx + j]);
    }
    idx += acc->common_feature_value.embed_sgd_dim;
    for (auto j = 0u; j < acc->common_feature_value.embedx_dim; ++j) {
      v.embedx_w.push_back(value[i][idx + j]);
    }
    idx += acc->common_feature_value.embedx_dim;
    for (auto j = 0u; j < acc->common_feature_value.embedx_sgd_dim; ++j) {
      v.embedx_g2sum.push_back(value[i][idx + j]);
    }

    DownpourSparsePushValueTest push_v;
    push_v.slot = grad[i][0];
    push_v.show = grad[i][1];
    push_v.click = grad[i][2];
    push_v.embed_g = grad[i][3];
    for (auto j = 0; j < parameter.embedx_dim(); ++j) {
      push_v.embedx_g.push_back(grad[i][4 + j]);
    }

    v.slot = push_v.slot;
    v.unseen_days = 0;
    v.show += push_v.show;
    v.click += push_v.click;
    v.delta_score += acc->show_click_score(push_v.show, push_v.click);

    acc->_embed_sgd_rule->update_value(&v.embed_w, &v.embed_g2sum[0],
                                       &push_v.embed_g);
    acc->_embedx_sgd_rule->update_value(&v.embedx_w[0], &v.embedx_g2sum[0],
                                        &push_v.embedx_g[0]);

    float* ptr = new float[acc->dim()];
    v.to_array(ptr, parameter.embedx_dim());
    exp_value.push_back(ptr);
  }
  acc->update(value, grad, item_size);

  for (auto i = 0u; i < item_size; ++i) {
    for (auto j = 0u; j < acc->dim(); ++j) {
      VLOG(3) << value[i][j] << ":" << exp_value[i][j] << " ";
      ASSERT_FLOAT_EQ(value[i][j], exp_value[i][j]);
    }
  }
}

TEST(downpour_feature_value_accessor_test, test_show_click_score) {
  TableAccessorParameter parameter = gen_param();
  CtrCommonAccessor* acc = new CtrCommonAccessor();
  ASSERT_EQ(acc->configure(parameter), 0);
  ASSERT_EQ(acc->initialize(), 0);

  float show = 10;
  float click = 6;
  ASSERT_FLOAT_EQ(acc->show_click_score(show, click), 6.8);
}

TEST(downpour_feature_value_accessor_test, test_string_related) {
  TableAccessorParameter parameter = gen_param();
  CtrCommonAccessor* acc = new CtrCommonAccessor();
  ASSERT_EQ(acc->configure(parameter), 0);
  ASSERT_EQ(acc->initialize(), 0);

  const int field_size = 15;
  float* value = new float[field_size];
  for (auto i = 0u; i < field_size; ++i) {
    value[i] = i;
  }

  auto str = acc->parse_to_string(value, 0);

  VLOG(3) << str << std::endl;

  str = "0 1 2 3 4 5 6";
  ASSERT_NE(acc->parse_from_string(str, value), 0);
  // make sure init_zero=true

  for (auto i = 7; i < 15; ++i) {
    ASSERT_FLOAT_EQ(value[i], 0);
  }
}
}  // namespace distributed
}  // namespace paddle
