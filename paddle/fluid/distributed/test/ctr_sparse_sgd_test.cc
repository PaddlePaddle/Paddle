/***************************************************************************
 *
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/
#include "paddle/fluid/distributed/table/ctr_sparse_sgd.h"
#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps.pb.h"

namespace paddle {
namespace distributed {

TEST(sparse_value_naive_sgd_test, init_and_update) {
  CtrSparseNaiveSGDRule rule;
  SparseCommonSGDRuleParameter param;
  param.set_name("naive");
  auto* naive_param = param.mutable_naive();
  naive_param->set_learning_rate(0.1);
  naive_param->set_initial_range(0.3);
  naive_param->add_weight_bounds(-10.0);
  naive_param->add_weight_bounds(10.0);

  rule.load_config(param, 10);

  // check init_value for zero
  int item_size = 10;
  float w[item_size];
  float grad[item_size];
  rule.init_value(w, w + 9, true);

  for (auto i = 0u; i < item_size; ++i) {
    ASSERT_FLOAT_EQ(w[i], 0);
  }

  // check init_value for random
  rule.init_value(w, w + 9, false);
  for (auto i = 0u; i < item_size; ++i) {
    ASSERT_TRUE(w[i] >= rule.min_bound() && w[i] <= rule.max_bound());
  }

  // check update_value for one field
  for (auto i = 0u; i < item_size; ++i) {
    w[i] = 0;
  }
  for (auto i = 0u; i < item_size; ++i) {
    grad[i] = (i + 1) * 1.0;
  }
  float label[] = {-0.100000, -0.200000, -0.300000, -0.400000, -0.500000,
                   -0.600000, -0.700000, -0.800000, -0.900000, -1.000000};
  const float* ptr_grad = grad;
  rule.update_value(w, w + 9, ptr_grad);

  for (auto i = 0u; i < item_size; ++i) {
    VLOG(3) << w[i] << "\n";
    ASSERT_FLOAT_EQ(w[i], label[i]);
  }
}

TEST(downpour_sparse_adagrad_test, test_init_and_update) {
  CtrSparseAdaGradSGDRule rule;
  SparseCommonSGDRuleParameter param;
  param.set_name("adagrad");
  auto* adagrad_param = param.mutable_adagrad();
  adagrad_param->set_learning_rate(0.1);
  adagrad_param->set_initial_g2sum(0.2);
  adagrad_param->set_initial_range(0.3);
  adagrad_param->add_weight_bounds(-10.0);
  adagrad_param->add_weight_bounds(10.0);

  rule.load_config(param, 10);

  // check init_value for zero
  int value_size = 11;
  int emb_size = 10;
  float w[value_size];

  rule.init_value(w, w + 10, true);

  for (auto i = 0u; i < emb_size; ++i) {
    ASSERT_FLOAT_EQ(w[i], 0);
  }
  ASSERT_FLOAT_EQ(w[emb_size], 0);

  // check init_value for random
  rule.init_value(w, w + 10, false);
  for (auto i = 0u; i < emb_size; ++i) {
    ASSERT_TRUE(w[i] >= rule.min_bound() && w[i] <= rule.max_bound());
  }
  ASSERT_FLOAT_EQ(w[emb_size], 0);

  // check update_value for one field
  for (auto i = 0u; i < emb_size; ++i) {
    w[i] = 0;
  }
  w[emb_size] = 0;
  float grad[emb_size];
  for (auto i = 0u; i < emb_size; ++i) {
    grad[i] = (i + 1) * 1.0;
  }

  const float* ptr_grad = grad;
  rule.update_value(w, w + 10, ptr_grad);
  float label[] = {-0.100000, -0.200000, -0.300000, -0.400000,
                   -0.500000, -0.600000, -0.700000, -0.800000,
                   -0.900000, -1.000000, 38.500000};
  for (auto i = 0u; i < value_size; ++i) {
    ASSERT_FLOAT_EQ(w[i], label[i]);
  }
}

TEST(downpour_sparse_adam_test, test_init_and_update) {
  const int embed_dim = 10;  // dims of parameters
  SparseCommonSGDRuleParameter param;
  param.set_name("adam");
  auto* adam_param = param.mutable_adam();
  adam_param->set_learning_rate(0.1);
  adam_param->set_initial_range(0.3);
  adam_param->set_beta1_decay_rate(0.9);
  adam_param->set_beta2_decay_rate(0.999);
  adam_param->set_ada_epsilon(1e-08);
  adam_param->add_weight_bounds(-10.0);
  adam_param->add_weight_bounds(10.0);

  ASSERT_FLOAT_EQ(param.adam().learning_rate(), 0.1);
  ASSERT_FLOAT_EQ(param.adam().initial_range(), 0.3);
  ASSERT_FLOAT_EQ(param.adam().beta1_decay_rate(), 0.9);
  ASSERT_FLOAT_EQ(param.adam().beta2_decay_rate(), 0.999);
  ASSERT_FLOAT_EQ(param.adam().ada_epsilon(), 1e-08);

  CtrSparseAdamSGDRule rule;

  rule.load_config(param, embed_dim);

  // check init_value for zero
  const int rule_dim =
      rule.dim();  // dims of gsum + g2sum + beta1_pow + beta2_pow in adam
  const int value_dim = embed_dim + rule_dim;  // total dims of w + rule
  float* value = new float[value_dim];
  rule.init_value(value, value + embed_dim, true);
  for (auto i = 0u; i < rule.beta1_pow_index(); ++i) {
    ASSERT_FLOAT_EQ(value[i], 0);
  }
  ASSERT_FLOAT_EQ(*(value + embed_dim + rule.beta1_pow_index()), 0.9);
  ASSERT_FLOAT_EQ(*(value + embed_dim + rule.beta2_pow_index()), 0.999);

  // check init_value for random
  rule.init_value(value, value + embed_dim, false);
  for (auto i = 0u; i < embed_dim; ++i) {
    ASSERT_TRUE(value[i] >= rule.min_bound() && value[i] <= rule.max_bound());
  }
  for (auto i = rule.gsum_index(); i < rule.beta1_pow_index(); ++i) {
    ASSERT_FLOAT_EQ(value[i + embed_dim], 0);
  }
  ASSERT_FLOAT_EQ(*(value + embed_dim + rule.beta1_pow_index()), 0.9);
  ASSERT_FLOAT_EQ(*(value + embed_dim + rule.beta2_pow_index()), 0.999);

  // check update_value
  rule.init_value(value, value + embed_dim, true);
  float* grad = new float[embed_dim];
  for (auto i = 0u; i < embed_dim; ++i) {
    grad[i] = (i + 1) * 1.0;
  }

  float label[] = {-0.0999999642,  -0.099999994, -0.099999994,  -0.099999994,
                   -0.099999994,   -0.099999994, -0.099999994,  -0.100000001,
                   -0.100000009,   -0.100000001, 0.100000024,   0.200000048,
                   0.300000072,    0.400000095,  0.500000119,   0.600000143,
                   0.700000167,    0.800000191,  0.900000215,   1.00000024,
                   0.000999987125, 0.0039999485, 0.00899988413, 0.015999794,
                   0.0249996781,   0.0359995365, 0.0489993691,  0.063999176,
                   0.0809989572,   0.0999987125, 0.809999943,   0.998001039};

  rule.update_value(value, value + embed_dim, grad);

  for (auto i = 0u; i < value_dim; ++i) {  // check update
    ASSERT_FLOAT_EQ(value[i], label[i]) << "i is " << i;
  }
}
}
}
