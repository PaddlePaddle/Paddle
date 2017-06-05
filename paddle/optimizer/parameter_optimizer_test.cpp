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

Tensor* fill_n_Tensor(size_t size) {
  real* ptr = new real[size];
  Tensor* param = new Tensor(ptr, size);
  Tensor& p = *param;
  for (auto i = 0; i < p.size(); ++i) {
    p[i] = (float)rand() / (float)RAND_MAX;
  }
  return param;
}

Tensor* fix_n_Tensor(size_t size) {
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
    create_sgd();
    create_adam();
  }
  virtual void TearDown() {}

  void create_sgd() {
    config.set_optimizer_name("SGD");
    config.mutable_sgd()->set_momentum(0.0);
    config.mutable_sgd()->set_decay(0.0);
    config.mutable_sgd()->set_nesterov(false);
    config.set_lr_policy("ConstLr");
    config.mutable_const_lr()->set_learning_rate(0.1);

    ParameterOptimizer* opt =
        ParameterOptimizer::create(config.SerializeAsString());
    opts.push_back(opt);
  }

  void create_adam() {
    config.set_optimizer_name("Adam");
    config.mutable_adam()->set_beta_1(0.9);
    config.mutable_adam()->set_beta_2(0.1);
    config.mutable_adam()->set_epsilon(1e-3);
    config.mutable_adam()->set_decay(0.0);
    config.set_lr_policy("ConstLr");
    config.mutable_const_lr()->set_learning_rate(0.1);
    ParameterOptimizer* opt =
        ParameterOptimizer::create(config.SerializeAsString());
    opts.push_back(opt);
  }
  void test_set_weight() {
    Tensor* p = fill_n_Tensor(size);
    for (size_t i = 0; i < opts.size(); ++i) {
      opts[i]->set_weight(p);
    }
  }

  void test_get_weight() {
    Tensor* p = fix_n_Tensor(size);
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
  void test_update() {
    Tensor* g = fix_n_Tensor(size);
    for (size_t i = 0; i < opts.size(); ++i) {
      opts[i]->update(g);
    }
  }

private:
  std::vector<ParameterOptimizer*> opts;
  OptimizerConfig config;
};

TEST_F(OptimizerTest, test_set_get_weight) {
  test_set_weight();
  test_get_weight();
}
TEST_F(OptimizerTest, test_update) { test_update(); }

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
