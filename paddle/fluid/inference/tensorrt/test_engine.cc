#include "paddle/fluid/inference/tensorrt/engine.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace paddle {

class TensorrtEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, cudaStreamCreate(&stream_));
    engine_ = new TensorrtEngine(1, 1 << 10, &stream_);
    engine_->InitNetwork();
  }

  void TearDown() override {
    delete engine_;
    cudaStreamDestroy(stream_);
  }

 protected:
  TensorrtEngine* engine_;
  cudaStream_t stream_;
};

TEST_F(TensorrtEngineTest, add_layer) {
  const int size = 1;

  float raw_weight[size] = {2.};  // Weight in CPU memory.
  float raw_bias[size] = {3.};

  LOG(INFO) << "create weights";
  TensorrtEngine::Weight weight(TensorrtEngine::data_type::kFLOAT, raw_weight,
                                size);
  TensorrtEngine::Weight bias(TensorrtEngine::data_type::kFLOAT, raw_bias,
                              size);
  auto* x = engine_->DeclInput("x", TensorrtEngine::data_type::kFLOAT,
                               nvinfer1::DimsCHW{1, 1, 1});
  auto* fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *x, size,
                                        weight.get(), bias.get());

  engine_->DeclOutput(fc_layer, 0, "y");
  LOG(INFO) << "freeze network";
  engine_->FreezeNetwork();
}

}  // namespace paddle
