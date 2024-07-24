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

#include "paddle/fluid/distributed/ps/table/sparse_sgd_rule.h"

#include <cmath>
#include <iostream>

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"

namespace paddle::distributed {

TEST(sparse_value_naive_sgd_test, init_and_update) {
  SparseNaiveSGDRule rule;
  SparseCommonSGDRuleParameter param;
  param.set_name("naive");
  auto* naive_param = param.mutable_naive();
  naive_param->set_learning_rate(0.1);
  naive_param->set_initial_range(0.3);
  naive_param->add_weight_bounds(-10.0);
  naive_param->add_weight_bounds(10.0);

  rule.LoadConfig(param, 10);

  // check init_value for zero
  const int kItemSize = 10;
  float w[kItemSize];     // NOLINT
  float grad[kItemSize];  // NOLINT
  rule.InitValue(w, w + 9, true);

  for (float item : w) {
    ASSERT_FLOAT_EQ(item, 0);
  }

  // check init_value for random
  rule.InitValue(w, w + 9, false);
  for (float item : w) {
    ASSERT_TRUE(item >= rule.MinBound() && item <= rule.MaxBound());
  }

  // check update_value for one field
  for (auto& item : w) {
    item = 0;
  }
  for (auto i = 0u; i < kItemSize; ++i) {
    grad[i] = static_cast<float>(i + 1) * 1.0;
  }
  std::array<float, 10> label = {-0.100000,
                                 -0.200000,
                                 -0.300000,
                                 -0.400000,
                                 -0.500000,
                                 -0.600000,
                                 -0.700000,
                                 -0.800000,
                                 -0.900000,
                                 -1.000000};
  const float* ptr_grad = grad;
  rule.UpdateValue(w, w + 9, ptr_grad);

  for (auto i = 0u; i < kItemSize; ++i) {
    VLOG(3) << w[i] << "\n";
    ASSERT_FLOAT_EQ(w[i], label[i]);
  }
}

TEST(downpour_sparse_adagrad_test, test_init_and_update) {
  SparseAdaGradSGDRule rule;
  SparseCommonSGDRuleParameter param;
  param.set_name("adagrad");
  auto* adagrad_param = param.mutable_adagrad();
  adagrad_param->set_learning_rate(0.1);
  adagrad_param->set_initial_g2sum(0.2);
  adagrad_param->set_initial_range(0.3);
  adagrad_param->add_weight_bounds(-10.0);
  adagrad_param->add_weight_bounds(10.0);

  rule.LoadConfig(param, 10);

  // check init_value for zero
  const int kValueSize = 11;
  int kEmbSize = 10;
  float w[kValueSize];  // NOLINT

  rule.InitValue(w, w + 10, true);

  for (int i = 0; i < kEmbSize; ++i) {
    ASSERT_FLOAT_EQ(w[i], 0);
  }
  ASSERT_FLOAT_EQ(w[kEmbSize], 0);

  // check init_value for random
  rule.InitValue(w, w + 10, false);
  for (int i = 0; i < kEmbSize; ++i) {
    ASSERT_TRUE(w[i] >= rule.MinBound() && w[i] <= rule.MaxBound());
  }
  ASSERT_FLOAT_EQ(w[kEmbSize], 0);

  // check update_value for one field
  for (int i = 0; i < kEmbSize; ++i) {
    w[i] = 0;
  }
  w[kEmbSize] = 0;
  float grad[kEmbSize];  // NOLINT
  for (int i = 0; i < kEmbSize; ++i) {
    grad[i] = static_cast<float>(i + 1) * 1.0;
  }

  const float* ptr_grad = grad;
  rule.UpdateValue(w, w + 10, ptr_grad);
  std::array<float, 11> label = {-0.100000,
                                 -0.200000,
                                 -0.300000,
                                 -0.400000,
                                 -0.500000,
                                 -0.600000,
                                 -0.700000,
                                 -0.800000,
                                 -0.900000,
                                 -1.000000,
                                 38.500000};
  for (auto i = 0u; i < kValueSize; ++i) {
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

  SparseAdamSGDRule rule;

  rule.LoadConfig(param, embed_dim);

  // check init_value for zero
  // dims of gsum + g2sum + beta1_pow + beta2_pow in adam
  const int rule_dim = static_cast<int>(rule.Dim());
  const int value_dim = embed_dim + rule_dim;  // total dims of w + rule
  float* value = new float[value_dim];
  rule.InitValue(value, value + embed_dim, true);
  for (auto i = 0u; i < rule.Beta1PowIndex(); ++i) {
    ASSERT_FLOAT_EQ(value[i], 0);
  }
  ASSERT_FLOAT_EQ(*(value + embed_dim + rule.Beta1PowIndex()), 0.9);
  ASSERT_FLOAT_EQ(*(value + embed_dim + rule.Beta2PowIndex()), 0.999);

  // check init_value for random
  rule.InitValue(value, value + embed_dim, false);
  for (auto i = 0u; i < embed_dim; ++i) {
    ASSERT_TRUE(value[i] >= rule.MinBound() && value[i] <= rule.MaxBound());
  }
  for (auto i = rule.GSumIndex(); i < rule.Beta1PowIndex(); ++i) {
    ASSERT_FLOAT_EQ(value[i + embed_dim], 0);
  }
  ASSERT_FLOAT_EQ(*(value + embed_dim + rule.Beta1PowIndex()), 0.9);
  ASSERT_FLOAT_EQ(*(value + embed_dim + rule.Beta2PowIndex()), 0.999);

  // check update_value
  rule.InitValue(value, value + embed_dim, true);
  float* grad = new float[embed_dim];
  for (auto i = 0u; i < embed_dim; ++i) {
    grad[i] = static_cast<float>(i + 1) * 1.0;
  }

  std::array<float, 32> label = {
      -0.0999999642,  -0.099999994, -0.099999994,  -0.099999994, -0.099999994,
      -0.099999994,   -0.099999994, -0.100000001,  -0.100000009, -0.100000001,
      0.100000024,    0.200000048,  0.300000072,   0.400000095,  0.500000119,
      0.600000143,    0.700000167,  0.800000191,   0.900000215,  1.00000024,
      0.000999987125, 0.0039999485, 0.00899988413, 0.015999794,  0.0249996781,
      0.0359995365,   0.0489993691, 0.063999176,   0.0809989572, 0.0999987125,
      0.809999943,    0.998001039};

  rule.UpdateValue(value, value + embed_dim, grad);

  for (int i = 0; i < value_dim; ++i) {  // check update
    ASSERT_FLOAT_EQ(value[i], label[i]) << "i is " << i;
  }
}
}  // namespace paddle::distributed
