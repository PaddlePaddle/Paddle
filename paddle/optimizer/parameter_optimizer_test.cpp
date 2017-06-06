#include "parameter_optimizer.h"
#include <cmath>
#include <tuple>
#include <vector>
#include "adadelta_optimizer.h"
#include "adagrad_optimizer.h"
#include "adam_optimizer.h"
#include "gtest/gtest.h"
#include "sgd_optimizer.h"
using namespace paddle;
using namespace paddle::optimizer;

Tensor* FillTensor(size_t size) {
  real* ptr = new real[size];
  Tensor* param = new Tensor(ptr, size);
  Tensor& p = *param;
  for (auto i = 0; i < p.size(); ++i) {
    p[i] = (float)rand() / (float)RAND_MAX;
  }
  return param;
}

Tensor* FixedTensor(size_t size) {
  real* ptr = new real[size];
  Tensor* param = new Tensor(ptr, size);
  Tensor& p = *param;
  for (auto i = 0; i < p.size(); ++i) {
    p[i] = i;
  }
  return param;
}

class OptimizerTest : public testing::Test {
public:
  // init tensor shape
  const size_t size = 5;

  virtual void SetUp() {
    CreateSGD();
    CreateAdam();
  }
  virtual void TearDown() {}

  void CreateSGD() {
    config.set_optimizer(OptimizerConfig::SGD);
    config.mutable_sgd()->set_momentum(0.0);
    config.mutable_sgd()->set_decay(0.0);
    config.mutable_sgd()->set_nesterov(false);
    config.set_lr_policy(OptimizerConfig::ConstLr);
    config.mutable_const_lr()->set_learning_rate(0.1);

    ParameterOptimizer* opt =
        ParameterOptimizer::Create(config.SerializeAsString());
    opts.push_back(opt);
  }

  void CreateAdam() {
    config.set_optimizer(OptimizerConfig::Adam);
    config.mutable_adam()->set_beta_1(0.9);
    config.mutable_adam()->set_beta_2(0.1);
    config.mutable_adam()->set_epsilon(1e-3);
    config.mutable_adam()->set_decay(0.0);
    config.set_lr_policy(OptimizerConfig::ConstLr);
    config.mutable_const_lr()->set_learning_rate(0.1);
    ParameterOptimizer* opt =
        ParameterOptimizer::Create(config.SerializeAsString());
    opts.push_back(opt);
  }
  void TestSetWeight() {
    Tensor* p = FillTensor(size);
    for (size_t i = 0; i < opts.size(); ++i) {
      opts[i]->set_weight(p);
    }
  }

  void TestGetWeight() {
    Tensor* p = FixedTensor(size);
    for (size_t i = 0; i < opts.size(); ++i) {
      opts[i]->set_weight(p);
    }
    for (size_t i = 0; i < opts.size(); ++i) {
      real* newp = (real*)opts[i]->get_weight();
      for (size_t j = 0; j < size; ++j) {
        EXPECT_EQ(newp[j], (*p)[j]);
      }
    }
  }
  void TestUpdate() {
    Tensor* g = FixedTensor(size);
    for (size_t i = 0; i < opts.size(); ++i) {
      opts[i]->Update(g);
    }
  }

private:
  std::vector<ParameterOptimizer*> opts;
  OptimizerConfig config;
};

TEST_F(OptimizerTest, test_set_get_weight) {
  TestSetWeight();
  TestGetWeight();
}
TEST_F(OptimizerTest, TestUpdate) { TestUpdate(); }

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
