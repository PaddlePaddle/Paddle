#include "paddle/fluid/inference/tensorrt/engine.h"

#include <gtest/testing.h>

class TensorrtEngineTest : public ::testing::Test {
 protected:
  void SetUp() override { engine_ = new TensorrtEngine(10, 1 << 20); }

  void TearDown() override { delete engine_; }

 private:
  TensorrtEngine* engine_;
};

TEST(TensorrtEngine, add_layer) {
  engine_.InitNetwork();
  const int size = 2;

  float raw_weight[size] = {0.1, 0.2};  // Weight in CPU memory.
  float raw_bias[size] = {0.3, 0.4};

  TensorrtEngine::Weight weight(data_type::kFLOAT, weight, size);
  engine_.DeclInput("x", data_type::kFLOAT, dim_type{1, 1, 1});
  TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, size, weight, bias)
  engine_.FreezeNetwork();
}
