#include "parameter_optimizer.h"
#include <cmath>
#include <map>
#include <vector>
#include "gtest/gtest.h"
#include "lr_policy.h"

using namespace paddle;
using namespace paddle::optimizer;

Tensor* FillTensor(size_t size) {
  Tensor* param = new Tensor(size);
  Tensor& p = *param;
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = (float)rand() / (float)RAND_MAX;
  }
  return param;
}

Tensor* FixedTensor(size_t size) {
  Tensor* param = new Tensor(size);
  Tensor& p = *param;
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = i;
  }
  return param;
}

class OptimizerTest : public testing::Test {
public:
  // init tensor shape
  const size_t kSize = 5;

  virtual void SetUp() {
    CreateSGD();
    CreateAdam();
  }
  virtual void TearDown() {}

  void CreateSGD() {
    Tensor* parameter = FixedTensor(kSize);
    config_.set_optimizer(OptimizerConfig::SGD);
    config_.mutable_sgd()->set_momentum(0.0);
    config_.mutable_sgd()->set_decay(0.0);
    config_.mutable_sgd()->set_nesterov(false);
    config_.set_lr_policy(OptimizerConfig::Const);
    config_.mutable_const_lr()->set_learning_rate(0.1);
    std::string str = config_.SerializeAsString();
    ParameterOptimizer* opt = ParameterOptimizer::Create(str, parameter);
    opts_.push_back(opt);
  }

  void CreateAdam() {
    Tensor* parameter = FixedTensor(kSize);
    config_.set_optimizer(OptimizerConfig::Adam);
    config_.mutable_adam()->set_beta_1(0.9);
    config_.mutable_adam()->set_beta_2(0.1);
    config_.mutable_adam()->set_epsilon(1e-3);
    config_.mutable_adam()->set_decay(0.0);
    config_.set_lr_policy(OptimizerConfig::Const);
    config_.mutable_const_lr()->set_learning_rate(0.1);
    std::string str = config_.SerializeAsString();
    ParameterOptimizer* opt = ParameterOptimizer::Create(str, parameter);
    opts_.push_back(opt);
  }

  void TestGetWeight() {
    Tensor* p = FixedTensor(kSize);
    for (size_t i = 0; i < opts_.size(); ++i) {
      int s = 0;
      float* newp = (float*)opts_[i]->get_weight(&s);
      for (size_t j = 0; j < kSize; ++j) {
        EXPECT_EQ(newp[j], (*p)[j]);
      }
    }
  }

  void TestUpdate() {
    Tensor* g = FixedTensor(kSize);
    for (size_t i = 0; i < opts_.size(); ++i) {
      opts_[i]->Update(g);
    }
  }

  void TestCheckPoint() {
    for (size_t i = 0; i < opts_.size(); ++i) {
      int state_len = 0;
      std::string state = opts_[i]->SerializeState(&state_len);
      opts_[i]->DeserializeState(state);
    }
  }

private:
  std::vector<ParameterOptimizer*> opts_;
  OptimizerConfig config_;
};

TEST_F(OptimizerTest, TestGetWeight) { TestGetWeight(); }

TEST_F(OptimizerTest, TestUpdate) { TestUpdate(); }

TEST_F(OptimizerTest, TestCheckPoint) { TestCheckPoint(); }

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
