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

#include "paddle/fluid/distributed/ps/table/ctr_dymf_accessor.h"

#include <cmath>
#include <iostream>

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/ps/table/sparse_sgd_rule.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"

namespace paddle::distributed {
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseAdaGradSGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, StdAdaGradSGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseAdamSGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseNaiveSGDRule);

TableAccessorParameter gen_param() {
  TableAccessorParameter param;
  param.set_accessor_class("CtrDymfAccessor");
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

  return param;
}

TEST(downpour_feature_value_accessor_test, test_shrink) {
  TableAccessorParameter parameter = gen_param();
  CtrDymfAccessor* acc = new CtrDymfAccessor();
  ASSERT_EQ(acc->Configure(parameter), 0);
  ASSERT_EQ(acc->Initialize(), 0);

  VLOG(3) << "size of struct: " << acc->common_feature_value.embed_sgd_dim
          << " " << acc->common_feature_value.embedx_dim << " "
          << acc->common_feature_value.embedx_sgd_dim << " "
          << acc->common_feature_value.Dim() << "\n";

  float* value = new float[acc->GetAccessorInfo().dim];
  for (auto i = 0u; i < acc->GetAccessorInfo().dim; ++i) {
    value[i] = static_cast<float>(i) * 1.0;
  }
  ASSERT_TRUE(!acc->Shrink(value));

  // set unseen_days too long
  value[0] = 1000;
  // set delta score too small
  value[1] = 0.001;
  ASSERT_TRUE(acc->Shrink(value));
}

TEST(downpour_feature_value_accessor_test, test_save) {
  TableAccessorParameter parameter = gen_param();
  CtrDymfAccessor* acc = new CtrDymfAccessor();
  ASSERT_EQ(acc->Configure(parameter), 0);
  ASSERT_EQ(acc->Initialize(), 0);

  float* value = new float[acc->GetAccessorInfo().dim];
  for (auto i = 0u; i < acc->GetAccessorInfo().dim; ++i) {
    value[i] = static_cast<float>(i) * 1.0;
  }

  // save all feature
  ASSERT_TRUE(acc->Save(value, 0));

  // save delta feature
  ASSERT_TRUE(acc->Save(value, 1));

  // save base feature with time decay
  ASSERT_TRUE(acc->Save(value, 2));

  VLOG(3) << "test_save:";
  for (auto i = 0u; i < acc->GetAccessorInfo().dim; ++i) {
    VLOG(3) << value[i];
  }
}

TEST(downpour_feature_value_accessor_test, test_create) {
  TableAccessorParameter parameter = gen_param();
  CtrDymfAccessor* acc = new CtrDymfAccessor();
  ASSERT_EQ(acc->Configure(parameter), 0);
  ASSERT_EQ(acc->Initialize(), 0);

  const int field_size = 8 + 8;
  const int item_size = 10;

  float** value = new float*[item_size];
  for (auto i = 0u; i < item_size; ++i) {
    value[i] = new float[field_size];
  }
  ASSERT_EQ(acc->Create(value, item_size), 0);

  for (auto i = 0u; i < item_size; ++i) {
    for (auto j = 0u; j < field_size; ++j) {
      VLOG(3) << value[i][j] << " ";
      // ASSERT_FLOAT_EQ(value[i][j], 0);
    }
    VLOG(3) << "\n";
  }
}

TEST(downpour_feature_value_accessor_test, test_show_click_score) {
  TableAccessorParameter parameter = gen_param();
  CtrDymfAccessor* acc = new CtrDymfAccessor();
  ASSERT_EQ(acc->Configure(parameter), 0);
  ASSERT_EQ(acc->Initialize(), 0);

  float show = 10;
  float click = 6;
  ASSERT_FLOAT_EQ(acc->ShowClickScore(show, click), 6.8);
}

TEST(downpour_feature_value_accessor_test, test_string_related) {
  TableAccessorParameter parameter = gen_param();
  CtrDymfAccessor* acc = new CtrDymfAccessor();
  ASSERT_EQ(acc->Configure(parameter), 0);
  ASSERT_EQ(acc->Initialize(), 0);

  const int field_size = 16;
  float* value = new float[field_size];
  for (auto i = 0u; i < field_size; ++i) {
    value[i] = static_cast<float>(i);
  }

  auto str = acc->ParseToString(value, 0);

  VLOG(0) << "test_string_related" << str << std::endl;

  str = "0 1 2 3 4 5 6 7";
  ASSERT_NE(acc->ParseFromString(str, value), 0);
  // make sure init_zero=true
}
}  // namespace paddle::distributed
