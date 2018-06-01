//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "parameter_optimizer.h"
#include <cmath>
#include <map>
#include <vector>
#include "gtest/gtest.h"
#include "lr_policy.h"

paddle::optimizer::Tensor* FillTensor(size_t size) {
  paddle::optimizer::Tensor* param = new paddle::optimizer::Tensor(size);
  paddle::optimizer::Tensor& p = *param;
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = (float)rand() / (float)RAND_MAX;
  }
  return param;
}

paddle::optimizer::Tensor* FixedTensor(size_t size) {
  paddle::optimizer::Tensor* param = new paddle::optimizer::Tensor(size);
  paddle::optimizer::Tensor& p = *param;
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = i;
  }
  return param;
}

class OptimizerTest : public testing::Test {
 public:
  virtual ~OptimizerTest() {}
  // init paddle::optimizer::Tensor shape
  const size_t kSize = 5;

  virtual void SetUp() {
    CreateSGD();
    CreateAdam();
  }
  virtual void TearDown() {}

  void CreateSGD() {
    paddle::optimizer::Tensor* parameter = FixedTensor(kSize);
    config_.set_optimizer(paddle::OptimizerConfig::SGD);
    config_.mutable_sgd()->set_momentum(0.0);
    config_.mutable_sgd()->set_decay(0.0);
    config_.mutable_sgd()->set_nesterov(false);
    config_.set_lr_policy(paddle::OptimizerConfig::Const);
    config_.mutable_const_lr()->set_learning_rate(0.1);
    std::string str = config_.SerializeAsString();
    paddle::optimizer::ParameterOptimizer* opt =
        paddle::optimizer::ParameterOptimizer::Create(str, parameter);
    opts_.push_back(opt);
  }

  void CreateAdam() {
    paddle::optimizer::Tensor* parameter = FixedTensor(kSize);
    config_.set_optimizer(paddle::OptimizerConfig::Adam);
    config_.mutable_adam()->set_beta_1(0.9);
    config_.mutable_adam()->set_beta_2(0.1);
    config_.mutable_adam()->set_epsilon(1e-3);
    config_.mutable_adam()->set_decay(0.0);
    config_.set_lr_policy(paddle::OptimizerConfig::Const);
    config_.mutable_const_lr()->set_learning_rate(0.1);
    std::string str = config_.SerializeAsString();
    paddle::optimizer::ParameterOptimizer* opt =
        paddle::optimizer::ParameterOptimizer::Create(str, parameter);
    opts_.push_back(opt);
  }

  void TestGetWeight() {
    paddle::optimizer::Tensor* p = FixedTensor(kSize);
    for (size_t i = 0; i < opts_.size(); ++i) {
      int s = 0;
      float* newp = (float*)opts_[i]->get_weight(&s);
      EXPECT_EQ(static_cast<size_t>(s), kSize);
      for (size_t j = 0; j < kSize; ++j) {
        EXPECT_EQ(newp[j], (*p)[j]);
      }
    }
  }

  void TestUpdate() {
    paddle::optimizer::Tensor* g = FixedTensor(kSize);
    for (size_t i = 0; i < opts_.size(); ++i) {
      opts_[i]->Update(g);
    }
  }

  void TestCheckPoint() {
    paddle::optimizer::Tensor* p = FixedTensor(kSize);
    for (size_t i = 0; i < opts_.size(); ++i) {
      auto state = opts_[i]->SerializeState();
      opts_[i]->DeserializeState(state);
      auto state1 = opts_[i]->SerializeState();
      opts_[i]->DeserializeState(state);
      EXPECT_EQ(state, state1);

      int s = 0;
      float* newp = (float*)opts_[i]->get_weight(&s);
      EXPECT_EQ(static_cast<size_t>(s), kSize);
      for (size_t j = 0; j < kSize; ++j) {
        EXPECT_EQ(newp[j], (*p)[j]);
      }
    }
  }

 private:
  std::vector<paddle::optimizer::ParameterOptimizer*> opts_;
  paddle::OptimizerConfig config_;
};

TEST_F(OptimizerTest, TestGetWeight) { TestGetWeight(); }

TEST_F(OptimizerTest, TestUpdate) { TestUpdate(); }

TEST_F(OptimizerTest, TestCheckPoint) { TestCheckPoint(); }
