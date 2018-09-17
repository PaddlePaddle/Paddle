#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(dirname, "", "dirname to tests.");

namespace paddle {
namespace inference {

TEST(AnalysisPredictor, ZeroCopy) {
  AnalysisConfig config;
  config.model_dir = FLAGS_dirname + "/word2vec.inference.model";
  config.use_feed_fetch_ops = false;

  auto predictor =
      CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
          config);

  auto w0 = predictor->GetInputTensor("firstw");
  auto w1 = predictor->GetInputTensor("secondw");
  auto w2 = predictor->GetInputTensor("thirdw");
  auto w3 = predictor->GetInputTensor("forthw");

  w0->Reshape({4, 1});
  w1->Reshape({4, 1});
  w2->Reshape({4, 1});
  w3->Reshape({4, 1});

  auto* w0_data = w0->mutable_data<int64_t>(PaddlePlace::kCPU);
  auto* w1_data = w1->mutable_data<int64_t>(PaddlePlace::kCPU);
  auto* w2_data = w2->mutable_data<int64_t>(PaddlePlace::kCPU);
  auto* w3_data = w3->mutable_data<int64_t>(PaddlePlace::kCPU);

  for (int i = 0; i < 4; i++) {
    w0_data[i] = i;
    w1_data[i] = i;
    w2_data[i] = i;
    w3_data[i] = i;
  }

  predictor->ZeroCopyRun();

}

}  // namespace inference
}  // namespace paddle
